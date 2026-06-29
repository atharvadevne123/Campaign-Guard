"""
Campaign feature engineering for ROI prediction.
Generates efficiency ratios, seasonal features, channel encodings,
interaction terms, and velocity/efficiency metrics from raw campaign records.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


CHANNELS = ["social", "search", "email", "display", "affiliate"]
SEASONS = ["q1", "q2", "q3", "q4"]
INDUSTRIES = [
    "technology", "retail", "finance", "healthcare", "education",
    "travel", "food", "automotive", "entertainment", "other",
]


class CampaignFeatureEngineer:
    """
    Transforms raw campaign records into an ML-ready feature matrix.

    Features produced
    -----------------
    * Efficiency ratios  : CTR, CVR, CPC, CPA, ROAS proxy
    * Channel encoding   : one-hot for social/search/email/display/affiliate
    * Seasonal encoding  : one-hot for Q1-Q4
    * Industry encoding  : frequency-based + one-hot top-N
    * Interaction terms  : channel × season, spend × duration
    * Velocity/utility   : spend per day, impressions per dollar, conversions per day
    * Derived scores     : engagement_score, efficiency_score
    """

    def __init__(self):
        self._fitted = False
        self._industry_freq: dict[str, float] = {}
        self._spend_stats: dict[str, float] = {}
        self._impressions_stats: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "CampaignFeatureEngineer":
        """Learn global statistics for normalisation from training data."""
        if "industry" in df.columns:
            freq = df["industry"].fillna("other").value_counts(normalize=True)
            self._industry_freq = freq.to_dict()

        for col, store in [("spend", "_spend_stats"), ("impressions", "_impressions_stats")]:
            if col in df.columns:
                setattr(self, store, {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std() + 1e-9),
                    "p90": float(df[col].quantile(0.90)),
                    "p99": float(df[col].quantile(0.99)),
                })

        self._fitted = True
        logger.info("CampaignFeatureEngineer fitted on {:,} records.", len(df))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations; returns enriched DataFrame."""
        if not self._fitted:
            raise RuntimeError("Call fit() or fit_transform() before transform().")
        df = df.copy()
        df = self._safe_numeric(df)
        df = self._efficiency_ratios(df)
        df = self._channel_encoding(df)
        df = self._seasonal_encoding(df)
        df = self._industry_encoding(df)
        df = self._interaction_features(df)
        df = self._velocity_features(df)
        df = self._composite_scores(df)
        logger.debug(
            "CampaignFeatureEngineer: {:,} records → {} features.", len(df), df.shape[1]
        )
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    def _safe_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce required numeric columns and fill missing with 0."""
        for col in ["spend", "impressions", "clicks", "conversions",
                    "audience_size", "campaign_duration_days"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df

    def _efficiency_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """CTR, CVR, CPC, CPA, ROAS proxy."""
        eps = 1e-9

        # Click-through rate
        df["ctr"] = df["clicks"] / (df["impressions"] + eps)

        # Conversion rate (clicks → conversions)
        df["cvr"] = df["conversions"] / (df["clicks"] + eps)

        # Cost per click
        df["cpc"] = df["spend"] / (df["clicks"] + eps)

        # Cost per acquisition
        df["cpa"] = df["spend"] / (df["conversions"] + eps)

        # ROAS proxy: conversions × assumed value per conversion / spend
        # We use a unit revenue of 1 so the signal is proportional
        df["roas_proxy"] = df["conversions"] / (df["spend"] + eps)

        # Log-transform spend and impressions to reduce skew
        df["log_spend"] = np.log1p(df["spend"])
        df["log_impressions"] = np.log1p(df["impressions"])

        # Normalised spend vs training distribution
        stats = self._spend_stats
        if stats:
            df["spend_zscore"] = (df["spend"] - stats["mean"]) / (stats["std"] + 1e-9)
            df["spend_above_p90"] = (df["spend"] > stats["p90"]).astype(int)
            df["spend_above_p99"] = (df["spend"] > stats["p99"]).astype(int)

        return df

    def _channel_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode marketing channel."""
        if "channel" not in df.columns:
            for ch in CHANNELS:
                df[f"channel_{ch}"] = 0
            return df
        ch_series = df["channel"].str.lower().fillna("unknown")
        for ch in CHANNELS:
            df[f"channel_{ch}"] = (ch_series == ch).astype(int)
        return df

    def _seasonal_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode season (q1–q4)."""
        if "season" not in df.columns:
            for s in SEASONS:
                df[f"season_{s}"] = 0
            return df
        season_series = df["season"].str.lower().fillna("q1")
        for s in SEASONS:
            df[f"season_{s}"] = (season_series == s).astype(int)
        return df

    def _industry_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Frequency-encode industry + one-hot top industries."""
        if "industry" not in df.columns:
            df["industry_freq"] = 0.0
            for ind in INDUSTRIES:
                df[f"industry_{ind}"] = 0
            return df

        ind_series = df["industry"].str.lower().fillna("other")
        df["industry_freq"] = ind_series.map(self._industry_freq).fillna(0.0)
        for ind in INDUSTRIES:
            df[f"industry_{ind}"] = (ind_series == ind).astype(int)
        return df

    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Channel × season + spend × duration interaction terms."""
        for ch in CHANNELS:
            for s in SEASONS:
                ch_col = f"channel_{ch}"
                s_col = f"season_{s}"
                if ch_col in df.columns and s_col in df.columns:
                    df[f"inter_{ch}_{s}"] = df[ch_col] * df[s_col]

        # Spend × duration
        if "campaign_duration_days" in df.columns:
            df["spend_x_duration"] = df["spend"] * df["campaign_duration_days"].clip(lower=1)
            df["daily_budget"] = df["spend"] / df["campaign_duration_days"].clip(lower=1)
        else:
            df["spend_x_duration"] = df["spend"]
            df["daily_budget"] = df["spend"]

        # CTR × CVR composite
        df["ctr_x_cvr"] = df.get("ctr", 0) * df.get("cvr", 0)

        return df

    def _velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-day and per-dollar efficiency metrics."""
        dur = df["campaign_duration_days"].clip(lower=1) if "campaign_duration_days" in df.columns else pd.Series(1, index=df.index)

        df["impressions_per_day"] = df["impressions"] / dur
        df["clicks_per_day"] = df["clicks"] / dur
        df["conversions_per_day"] = df["conversions"] / dur
        df["spend_per_day"] = df["spend"] / dur
        df["impressions_per_dollar"] = df["impressions"] / (df["spend"] + 1e-9)
        df["conversions_per_dollar"] = df["conversions"] / (df["spend"] + 1e-9)

        # Audience penetration rate
        if "audience_size" in df.columns:
            df["audience_penetration"] = df["impressions"] / (df["audience_size"] + 1e-9)
        else:
            df["audience_penetration"] = 0.0

        return df

    def _composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engagement score and efficiency score as aggregate signals."""
        # Engagement: normalised blend of CTR and CVR
        ctr = df.get("ctr", pd.Series(0, index=df.index))
        cvr = df.get("cvr", pd.Series(0, index=df.index))
        df["engagement_score"] = (0.4 * ctr + 0.6 * cvr).clip(upper=1.0)

        # Efficiency: conversions per unit spend (log-scaled)
        cpd = df.get("conversions_per_dollar", pd.Series(0, index=df.index))
        df["efficiency_score"] = np.log1p(cpd)

        return df
