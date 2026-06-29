"""
Supervised campaign ROI classification ensemble.
XGBoost + LightGBM + Random Forest with soft-voting and calibrated probabilities.
SHAP values computed per-prediction for local explainability.
ROI tiers: POOR (0) / AVERAGE (1) / GOOD (2) / EXCELLENT (3)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path
from loguru import logger
from typing import Optional

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn


ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

ROI_TIERS = {0: "POOR", 1: "AVERAGE", 2: "GOOD", 3: "EXCELLENT"}
ROI_TIERS_INV = {v: k for k, v in ROI_TIERS.items()}


class CampaignEnsemble:
    """
    Three-model soft-voting ensemble for campaign ROI tier classification.
    Handles SMOTE oversampling for class imbalance, isotonic calibration,
    and SHAP explanations via TreeExplainer on the XGBoost base estimator.
    """

    def __init__(
        self,
        xgb_params: Optional[dict] = None,
        lgb_params: Optional[dict] = None,
        rf_params: Optional[dict] = None,
        calibration_method: str = "isotonic",
        random_state: int = 42,
    ):
        self.random_state = random_state
        self.calibration_method = calibration_method
        self.feature_names: list[str] = []

        xgb_defaults = dict(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
        lgb_defaults = dict(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multiclass",
            num_class=4,
            is_unbalance=True,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        rf_defaults = dict(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

        self.xgb_model = xgb.XGBClassifier(**(xgb_params or xgb_defaults))
        self.lgb_model = lgb.LGBMClassifier(**(lgb_params or lgb_defaults))
        self.rf_model = RandomForestClassifier(**(rf_params or rf_defaults))

        self.ensemble: Optional[CalibratedClassifierCV] = None
        self.scaler = StandardScaler()
        self._shap_explainer: Optional[shap.TreeExplainer] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_X: Optional[pd.DataFrame] = None,
        eval_y: Optional[pd.Series] = None,
        mlflow_run: bool = True,
    ) -> "CampaignEnsemble":
        """Fit ensemble on training data with SMOTE balancing and calibration."""
        self.feature_names = list(X.columns)
        X_arr = self.scaler.fit_transform(X.values.astype(float))
        y_arr = y.values.astype(int)

        unique, counts = np.unique(y_arr, return_counts=True)
        dist = dict(zip([ROI_TIERS[u] for u in unique], counts.tolist()))
        logger.info(
            "Applying SMOTE to balance {:,} samples. Class distribution: {}.",
            len(y_arr), dist,
        )

        # SMOTE requires at least k_neighbors samples per minority class
        min_count = min(counts)
        k_neighbors = min(5, min_count - 1) if min_count > 1 else 1
        smote = SMOTE(
            sampling_strategy="not majority",
            k_neighbors=k_neighbors,
            random_state=self.random_state,
        )
        X_res, y_res = smote.fit_resample(X_arr, y_arr)
        logger.info("Post-SMOTE: {:,} samples.", len(y_res))

        voter = VotingClassifier(
            estimators=[
                ("xgb", self.xgb_model),
                ("lgb", self.lgb_model),
                ("rf", self.rf_model),
            ],
            voting="soft",
            weights=[0.4, 0.4, 0.2],
        )
        self.ensemble = CalibratedClassifierCV(
            voter,
            method=self.calibration_method,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state),
        )

        if mlflow_run:
            mlflow_uri = __import__("os").getenv("MLFLOW_TRACKING_URI", "")
            if mlflow_uri:
                mlflow.set_tracking_uri(mlflow_uri)
            with mlflow.start_run(run_name="campaign_ensemble"):
                self.ensemble.fit(X_res, y_res)
                self._log_metrics(X_res, y_res, eval_X, eval_y)
        else:
            self.ensemble.fit(X_res, y_res)

        # Build SHAP explainer on the first XGBoost base estimator
        try:
            base_xgb = self.ensemble.calibrated_classifiers_[0].estimator.named_estimators_["xgb"]
            self._shap_explainer = shap.TreeExplainer(base_xgb)
            logger.success("SHAP TreeExplainer initialised.")
        except Exception as exc:
            logger.warning("Could not build SHAP explainer: {}", exc)

        self._fitted = True
        logger.success("CampaignEnsemble fitted.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probability matrix of shape (n_samples, 4)."""
        self._check_fitted()
        return self.ensemble.predict_proba(self.scaler.transform(X.values.astype(float)))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted class indices (0–3)."""
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_roi_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return a scalar ROI score in [0, 1] computed as the
        probability-weighted average of class indices / 3.
        """
        proba = self.predict_proba(X)
        weights = np.arange(4, dtype=float) / 3.0
        return (proba * weights).sum(axis=1)

    def explain(self, X: pd.DataFrame, max_display: int = 10) -> list[dict]:
        """Return SHAP values and top-k feature contributions for each row."""
        if self._shap_explainer is None:
            return [{"error": "SHAP explainer not available."}] * len(X)
        X_scaled = self.scaler.transform(X.values.astype(float))
        raw_sv = self._shap_explainer.shap_values(X_scaled)
        # For multiclass XGB, shap_values returns list of arrays (one per class)
        # Use class with highest mean absolute shap across all classes
        if isinstance(raw_sv, list):
            sv = np.stack(raw_sv, axis=0).mean(axis=0)  # (n_samples, n_features)
        else:
            sv = raw_sv

        results = []
        for i in range(len(X)):
            top_idx = np.argsort(np.abs(sv[i]))[::-1][:max_display]
            base_val = self._shap_explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = float(np.mean(base_val))
            results.append(
                {
                    "shap_features": {
                        self.feature_names[j]: float(sv[i][j]) for j in top_idx
                    },
                    "base_value": float(base_val),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path else ARTIFACT_DIR / "campaign_ensemble.joblib"
        joblib.dump(self, path)
        logger.info("CampaignEnsemble saved → {}", path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CampaignEnsemble":
        path = Path(path) if path else ARTIFACT_DIR / "campaign_ensemble.joblib"
        obj = joblib.load(path)
        logger.info("CampaignEnsemble loaded ← {}", path)
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_metrics(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_X: Optional[pd.DataFrame],
        eval_y: Optional[pd.Series],
    ) -> None:
        classes = np.arange(4)
        train_proba = self.ensemble.predict_proba(X_train)
        y_bin = label_binarize(y_train, classes=classes)
        mlflow.log_metric(
            "train_roc_auc_ovr",
            roc_auc_score(y_bin, train_proba, multi_class="ovr", average="macro"),
        )
        if eval_X is not None and eval_y is not None:
            eval_X_scaled = self.scaler.transform(eval_X.values.astype(float))
            eval_proba = self.ensemble.predict_proba(eval_X_scaled)
            eval_pred = np.argmax(eval_proba, axis=1)
            eval_y_arr = eval_y.values.astype(int)
            eval_bin = label_binarize(eval_y_arr, classes=classes)
            auc = roc_auc_score(eval_bin, eval_proba, multi_class="ovr", average="macro")
            mlflow.log_metric("val_roc_auc_ovr", auc)
            logger.info("Val macro AUC-ROC (OvR): {:.4f}", auc)
            logger.info(
                "Val classification report:\n{}",
                classification_report(eval_y_arr, eval_pred, target_names=list(ROI_TIERS.values())),
            )

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call train() on CampaignEnsemble before inference.")
