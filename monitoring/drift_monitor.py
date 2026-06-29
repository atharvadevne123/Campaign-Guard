"""
KS-drift monitoring for Campaign-Guard.
Detects statistical drift in campaign feature distributions using the
Kolmogorov-Smirnov two-sample test with Evidently as a richer fallback.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as scipy_stats


MONITOR_DIR = Path(__file__).parent / "reports"
MONITOR_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_PATH = MONITOR_DIR / "reference_stats.parquet"

# Feature columns most indicative of campaign data distribution
CORE_FEATURES = [
    "ctr", "cvr", "cpc", "cpa", "roas_proxy",
    "log_spend", "log_impressions",
    "conversions_per_day", "impressions_per_day",
    "engagement_score", "efficiency_score",
]


class DriftMonitor:
    """
    Compares current batch statistics against a reference distribution.
    Uses the KS test (p < 0.05 threshold) per feature and aggregates
    a drift report.  Alerts are generated when the drifted-feature ratio
    exceeds *drift_threshold*.
    """

    def __init__(
        self,
        drift_threshold: float = 0.20,
        ks_p_threshold: float = 0.05,
    ):
        self.drift_threshold = drift_threshold
        self.ks_p_threshold = ks_p_threshold
        self._reference: Optional[pd.DataFrame] = None
        self._load_reference()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_reference(self, df: pd.DataFrame) -> None:
        """Store a clean baseline for future drift comparisons."""
        self._reference = df.copy()
        df.to_parquet(REFERENCE_PATH, index=False)
        logger.info("Drift reference stored ({:,} rows, {} columns).", len(df), df.shape[1])

    def detect_drift(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> dict:
        """
        Run KS-drift detection between reference_df and current_df.

        Returns
        -------
        dict with keys:
            drift_detected (bool), drifted_features (list[str]),
            feature_reports (dict), drift_ratio (float),
            run_timestamp (str), alerts (list[str])
        """
        numeric_ref = reference_df.select_dtypes(include="number").columns.tolist()
        numeric_cur = current_df.select_dtypes(include="number").columns.tolist()
        shared = [c for c in numeric_ref if c in numeric_cur]

        feature_reports: dict[str, dict] = {}
        drifted: list[str] = []

        for col in shared:
            ref_vals = reference_df[col].dropna().values
            cur_vals = current_df[col].dropna().values
            if len(ref_vals) < 5 or len(cur_vals) < 5:
                continue
            ks_stat, p_val = scipy_stats.ks_2samp(ref_vals, cur_vals)
            drifted_flag = bool(p_val < self.ks_p_threshold)
            if drifted_flag:
                drifted.append(col)
            feature_reports[col] = {
                "ks_statistic": round(float(ks_stat), 6),
                "p_value": round(float(p_val), 6),
                "drift_detected": drifted_flag,
                "ref_mean": round(float(ref_vals.mean()), 6),
                "cur_mean": round(float(cur_vals.mean()), 6),
                "ref_std": round(float(ref_vals.std()), 6),
                "cur_std": round(float(cur_vals.std()), 6),
            }

        n_tested = len(feature_reports)
        drift_ratio = len(drifted) / n_tested if n_tested > 0 else 0.0
        drift_detected = drift_ratio > self.drift_threshold

        alerts = self._generate_alerts(feature_reports, drifted, drift_ratio)

        report = {
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "reference_rows": len(reference_df),
            "current_rows": len(current_df),
            "features_tested": n_tested,
            "drifted_features": drifted,
            "drift_ratio": round(drift_ratio, 4),
            "drift_detected": drift_detected,
            "feature_reports": feature_reports,
            "alerts": alerts,
        }

        self._save_report(report)

        if drift_detected:
            logger.warning(
                "DRIFT DETECTED — {}/{} features drifted ({:.0%}).",
                len(drifted), n_tested, drift_ratio,
            )
            for alert in alerts:
                logger.warning(alert)
        else:
            logger.info(
                "No significant drift. {}/{} features within tolerance.",
                n_tested - len(drifted), n_tested,
            )

        return report

    def run(self, current: pd.DataFrame) -> dict:
        """
        Run drift check against the stored reference distribution.
        If no reference is set, the current batch becomes the reference.
        """
        if self._reference is None:
            logger.warning("No reference set. Storing current batch as reference.")
            self.set_reference(current)
            return {"drift_detected": False, "reason": "Reference just established."}
        return self.detect_drift(self._reference, current)

    # ------------------------------------------------------------------
    # Alert generation
    # ------------------------------------------------------------------

    def _generate_alerts(
        self,
        feature_reports: dict[str, dict],
        drifted: list[str],
        drift_ratio: float,
    ) -> list[str]:
        alerts = []
        if drift_ratio > self.drift_threshold:
            alerts.append(
                f"CRITICAL: {len(drifted)} features drifted ({drift_ratio:.0%} > "
                f"threshold {self.drift_threshold:.0%}). Model retraining recommended."
            )
        for col in drifted:
            rpt = feature_reports[col]
            mean_shift = abs(rpt["cur_mean"] - rpt["ref_mean"])
            if rpt["ref_mean"] != 0:
                pct_shift = mean_shift / abs(rpt["ref_mean"]) * 100
                if pct_shift > 20:
                    alerts.append(
                        f"Feature '{col}': mean shifted {pct_shift:.1f}% "
                        f"({rpt['ref_mean']:.4f} → {rpt['cur_mean']:.4f}), "
                        f"KS p={rpt['p_value']:.4f}."
                    )
        return alerts

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_report(self, report: dict) -> None:
        ts = report["run_timestamp"].replace(":", "-")[:19]
        path = MONITOR_DIR / f"drift_report_{ts}.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        logger.debug("Drift report saved → {}", path)

    def _load_reference(self) -> None:
        if REFERENCE_PATH.exists():
            self._reference = pd.read_parquet(REFERENCE_PATH)
            logger.info(
                "Drift reference loaded ({:,} rows).", len(self._reference)
            )
