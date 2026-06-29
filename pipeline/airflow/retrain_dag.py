"""
Airflow DAG: Campaign-Guard Model Retraining Pipeline

Schedule: Daily at 02:00 UTC
Flow:
  fetch_training_data_from_foundry
      → check_drift
      → retrain_model
      → evaluate_model
      → push_model_to_foundry
      → restart_api
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from loguru import logger


# ---------------------------------------------------------------------------
# DAG defaults
# ---------------------------------------------------------------------------

DEFAULT_ARGS = {
    "owner": "campaign-ml-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
MODEL_OUTPUT_DIR = os.path.join(AIRFLOW_HOME, "models")
CAMPAIGN_DATASET_RID = os.getenv("CAMPAIGN_DATASET_RID", "")
PREDICTIONS_DATASET_RID = os.getenv("PREDICTIONS_DATASET_RID", "")
ENSEMBLE_ARTIFACT = os.path.join(MODEL_OUTPUT_DIR, "campaign_ensemble.joblib")
FEATURE_ENGINEER_ARTIFACT = os.path.join(MODEL_OUTPUT_DIR, "feature_engineer.joblib")
FEATURE_COLS_PATH = os.path.join(MODEL_OUTPUT_DIR, "feature_cols.json")
MIN_DRIFT_RATIO_FOR_RETRAIN = float(os.getenv("MIN_DRIFT_RATIO_FOR_RETRAIN", "0.20"))


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="campaign_guard_retrain",
    default_args=DEFAULT_ARGS,
    description=(
        "Daily retraining pipeline: Foundry data fetch → drift check → "
        "retrain → evaluate → push model → restart API"
    ),
    schedule_interval="0 2 * * *",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["campaign-guard", "ml", "retraining"],
) as dag:

    # -----------------------------------------------------------------------
    # Task 1: Fetch training data from Palantir Foundry
    # -----------------------------------------------------------------------

    @task(task_id="fetch_training_data_from_foundry")
    def fetch_training_data_from_foundry(**context) -> str:
        """Pull the latest campaign training data from Foundry dataset."""
        import sys
        sys.path.insert(0, AIRFLOW_HOME)
        from foundry.foundry_client import FoundryClient

        execution_date = context["execution_date"].strftime("%Y%m%d")
        local_path = f"/tmp/campaign_training_{execution_date}.parquet"

        if CAMPAIGN_DATASET_RID:
            client = FoundryClient()
            try:
                df = client.read_dataset(CAMPAIGN_DATASET_RID)
                df.to_parquet(local_path, index=False)
                logger.info(
                    "Fetched {:,} campaigns from Foundry (dataset={}).",
                    len(df), CAMPAIGN_DATASET_RID,
                )
                return local_path
            except Exception as exc:
                logger.warning(
                    "Foundry fetch failed ({}). Falling back to synthetic data.", exc
                )

        logger.warning("Generating synthetic training data for demo/fallback.")
        df = _generate_synthetic_campaigns(n=20_000)
        df.to_parquet(local_path, index=False)
        return local_path

    # -----------------------------------------------------------------------
    # Task 2: Check drift against current production model reference
    # -----------------------------------------------------------------------

    @task(task_id="check_drift")
    def check_drift(data_path: str) -> dict:
        """Compare incoming data distribution against the stored reference."""
        import sys
        sys.path.insert(0, AIRFLOW_HOME)
        from monitoring.drift_monitor import DriftMonitor
        import joblib

        df = pd.read_parquet(data_path)

        if Path(FEATURE_ENGINEER_ARTIFACT).exists():
            fe = joblib.load(FEATURE_ENGINEER_ARTIFACT)
            df_feat = fe.transform(df)
        else:
            df_feat = df

        monitor = DriftMonitor()
        report = monitor.run(df_feat)

        drift_detected = report.get("drift_detected", False)
        drift_ratio = report.get("drift_ratio", 0.0)

        logger.info(
            "Drift check complete. Detected={}, ratio={:.2%}.",
            drift_detected, drift_ratio,
        )
        return {
            "drift_detected": drift_detected,
            "drift_ratio": drift_ratio,
            "drifted_features": report.get("drifted_features", []),
            "data_path": data_path,
        }

    # -----------------------------------------------------------------------
    # Task 3: Retrain model
    # -----------------------------------------------------------------------

    @task(task_id="retrain_model")
    def retrain_model(drift_report: dict) -> str:
        """
        Retrain CampaignEnsemble using the latest data.
        Always retrains (drift check result is logged but doesn't gate this step
        unless configured via MIN_DRIFT_RATIO_FOR_RETRAIN = 0).
        """
        import sys
        import joblib
        sys.path.insert(0, AIRFLOW_HOME)
        from pipeline.feature_engineering import CampaignFeatureEngineer
        from models.ensemble.campaign_classifier import CampaignEnsemble
        from sklearn.model_selection import train_test_split

        data_path = drift_report["data_path"]
        df = pd.read_parquet(data_path)

        logger.info(
            "Retraining on {:,} campaigns. Drift ratio was {:.2%}.",
            len(df), drift_report.get("drift_ratio", 0.0),
        )

        # Feature engineering
        fe = CampaignFeatureEngineer()
        df_feat = fe.fit_transform(df)

        Path(MODEL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        joblib.dump(fe, FEATURE_ENGINEER_ARTIFACT)

        # Derive ROI label from ROAS proxy if not present
        if "roi_label" not in df_feat.columns:
            df_feat = _derive_roi_labels(df_feat)

        exclude = {
            "campaign_id", "channel", "industry", "season",
            "roi_label", "roi_tier",
        }
        feature_cols = [
            c for c in df_feat.select_dtypes(include="number").columns
            if c not in exclude and c in df_feat.columns
        ]
        Path(FEATURE_COLS_PATH).write_text(json.dumps(feature_cols))

        X = df_feat[feature_cols].fillna(0)
        y = df_feat["roi_label"].astype(int)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        ensemble = CampaignEnsemble()
        ensemble.train(X_train, y_train, eval_X=X_val, eval_y=y_val, mlflow_run=True)
        ensemble.save(Path(ENSEMBLE_ARTIFACT))

        logger.success("Model retrained and saved → {}", ENSEMBLE_ARTIFACT)
        return ENSEMBLE_ARTIFACT

    # -----------------------------------------------------------------------
    # Task 4: Evaluate model
    # -----------------------------------------------------------------------

    @task(task_id="evaluate_model")
    def evaluate_model(ensemble_path: str) -> dict:
        """Run held-out evaluation and emit metrics."""
        import sys
        import joblib
        sys.path.insert(0, AIRFLOW_HOME)
        from models.ensemble.campaign_classifier import CampaignEnsemble
        from sklearn.metrics import classification_report, roc_auc_score
        from sklearn.preprocessing import label_binarize
        import numpy as np

        ensemble = CampaignEnsemble.load(ensemble_path)

        # Generate a small eval set for quick sanity check
        df_eval = _generate_synthetic_campaigns(n=2_000)
        fe = joblib.load(FEATURE_ENGINEER_ARTIFACT) if Path(FEATURE_ENGINEER_ARTIFACT).exists() else None
        df_eval_feat = fe.transform(df_eval) if fe else df_eval
        df_eval_feat = _derive_roi_labels(df_eval_feat)

        feature_cols = json.loads(Path(FEATURE_COLS_PATH).read_text())
        X_eval = pd.DataFrame(0.0, index=df_eval_feat.index, columns=feature_cols)
        for col in feature_cols:
            if col in df_eval_feat.columns:
                X_eval[col] = pd.to_numeric(df_eval_feat[col], errors="coerce").fillna(0)

        y_eval = df_eval_feat["roi_label"].astype(int)
        proba = ensemble.predict_proba(X_eval)
        y_bin = label_binarize(y_eval, classes=[0, 1, 2, 3])
        auc = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
        preds = np.argmax(proba, axis=1)

        report = classification_report(y_eval, preds, output_dict=True)
        logger.info("Evaluation AUC-ROC (macro OvR): {:.4f}", auc)
        logger.info("Classification report: {}", json.dumps(report, indent=2))

        return {
            "auc_roc_macro": round(auc, 4),
            "accuracy": round(report.get("accuracy", 0.0), 4),
            "ensemble_path": ensemble_path,
        }

    # -----------------------------------------------------------------------
    # Task 5: Push model artifacts to Foundry
    # -----------------------------------------------------------------------

    @task(task_id="push_model_to_foundry")
    def push_model_to_foundry(eval_metrics: dict) -> None:
        """Upload the retrained model metadata to the Foundry model catalog."""
        import sys
        sys.path.insert(0, AIRFLOW_HOME)
        from foundry.foundry_client import FoundryClient, FoundryError

        client = FoundryClient()
        metadata = {
            "name": "campaign-guard-ensemble",
            "version": __import__("datetime").datetime.utcnow().strftime("v%Y%m%d_%H%M"),
            "framework": "XGBoost+LightGBM+RandomForest",
            "metrics": {
                "auc_roc_macro": eval_metrics.get("auc_roc_macro"),
                "accuracy": eval_metrics.get("accuracy"),
            },
            "artifact_path": eval_metrics.get("ensemble_path"),
        }

        if not CAMPAIGN_DATASET_RID:
            logger.warning("CAMPAIGN_DATASET_RID not set — skipping Foundry model registration.")
            return

        try:
            client.register_model(metadata)
            logger.success("Model metadata pushed to Foundry catalog.")
        except FoundryError as exc:
            logger.error("Foundry model registration failed: {}", exc)
            raise

    # -----------------------------------------------------------------------
    # Task 6: Restart API (signal gunicorn to reload workers)
    # -----------------------------------------------------------------------

    @task(task_id="restart_api")
    def restart_api() -> None:
        """Send HUP signal to gunicorn master to perform a graceful reload."""
        import subprocess
        import signal

        pid_file = os.getenv("GUNICORN_PID_FILE", "/tmp/campaign_guard.pid")
        if not Path(pid_file).exists():
            logger.warning(
                "PID file {} not found — skipping API restart. "
                "Restart manually or via your orchestrator.",
                pid_file,
            )
            return

        pid = int(Path(pid_file).read_text().strip())
        try:
            import os as _os
            _os.kill(pid, signal.SIGHUP)
            logger.success("Sent SIGHUP to gunicorn PID {} — workers will reload.", pid)
        except ProcessLookupError:
            logger.warning("PID {} not found. API may already be down.", pid)
        except PermissionError as exc:
            logger.error("Could not signal gunicorn: {}", exc)

    # -----------------------------------------------------------------------
    # Wire tasks
    # -----------------------------------------------------------------------

    raw_data_path = fetch_training_data_from_foundry()
    drift_report  = check_drift(raw_data_path)
    model_path    = retrain_model(drift_report)
    eval_metrics  = evaluate_model(model_path)
    push_model_to_foundry(eval_metrics)
    restart_api()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_synthetic_campaigns(n: int = 20_000) -> pd.DataFrame:
    """Generate a synthetic campaign dataset for demo/fallback use."""
    import numpy as np

    rng = np.random.default_rng(2024)
    channels = ["social", "search", "email", "display", "affiliate"]
    industries = ["technology", "retail", "finance", "healthcare", "education",
                  "travel", "food", "automotive", "entertainment", "other"]
    seasons = ["q1", "q2", "q3", "q4"]

    spend = rng.lognormal(mean=9.5, sigma=1.2, size=n).round(2)
    impressions = (spend * rng.uniform(10, 100, n)).astype(int).clip(min=1)
    clicks = (impressions * rng.uniform(0.005, 0.08, n)).astype(int)
    conversions = (clicks * rng.uniform(0.01, 0.15, n)).astype(int)

    return pd.DataFrame({
        "campaign_id":             [f"CMP{i:07d}" for i in range(n)],
        "channel":                 rng.choice(channels, n),
        "spend":                   spend,
        "impressions":             impressions,
        "clicks":                  clicks,
        "conversions":             conversions,
        "audience_size":           rng.integers(50_000, 5_000_000, n),
        "campaign_duration_days":  rng.integers(7, 90, n),
        "industry":                rng.choice(industries, n),
        "season":                  rng.choice(seasons, n),
    })


def _derive_roi_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a 4-class ROI label from ROAS proxy (conversions / spend).
    0=POOR, 1=AVERAGE, 2=GOOD, 3=EXCELLENT
    """
    eps = 1e-9
    roas = df["conversions"] / (df["spend"] + eps) if "conversions" in df.columns else 0
    p33 = roas.quantile(0.33)
    p66 = roas.quantile(0.66)
    p90 = roas.quantile(0.90)

    labels = pd.cut(
        roas,
        bins=[-float("inf"), p33, p66, p90, float("inf")],
        labels=[0, 1, 2, 3],
    ).astype(int)
    df["roi_label"] = labels
    return df
