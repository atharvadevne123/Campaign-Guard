"""
Shapley value-based channel attribution model.

Treats marketing channels as players in a cooperative game where the
grand coalition is the full campaign portfolio.  Marginal contributions
are averaged over all permutations (exact Shapley) when the channel
count is small, or approximated via Monte-Carlo sampling for large sets.
"""

from __future__ import annotations

import itertools
import math
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# Maximum number of channels for exact (factorial) Shapley computation.
# Above this threshold we fall back to Monte-Carlo approximation.
EXACT_THRESHOLD = 8


class AttributionModel:
    """
    Shapley-value channel attribution.

    Usage
    -----
    model = AttributionModel()
    attribution = model.attribute(campaigns_df)
    # → {"social": 0.35, "search": 0.42, "email": 0.18, ...}
    """

    def __init__(
        self,
        value_col: str = "conversions",
        weight_col: Optional[str] = "spend",
        n_mc_samples: int = 2_000,
        random_state: int = 42,
    ):
        self.value_col = value_col
        self.weight_col = weight_col
        self.n_mc_samples = n_mc_samples
        self.rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attribute(self, campaigns_df: pd.DataFrame) -> dict[str, float]:
        """
        Compute Shapley attribution for each channel present in *campaigns_df*.

        Parameters
        ----------
        campaigns_df : DataFrame with at least ``channel`` and ``value_col`` columns.

        Returns
        -------
        dict  channel → attributed value (sums to grand coalition value).
        """
        if "channel" not in campaigns_df.columns:
            raise ValueError("campaigns_df must contain a 'channel' column.")
        if self.value_col not in campaigns_df.columns:
            raise ValueError(f"campaigns_df must contain '{self.value_col}' column.")

        channels = sorted(campaigns_df["channel"].dropna().unique().tolist())
        n = len(channels)
        if n == 0:
            return {}

        logger.info(
            "Computing Shapley attribution for {} channels via {}.",
            n,
            "exact enumeration" if n <= EXACT_THRESHOLD else "Monte-Carlo ({} samples)".format(
                self.n_mc_samples
            ),
        )

        def coalition_value(subset: tuple[str, ...]) -> float:
            return self._coalition_value(campaigns_df, set(subset))

        if n <= EXACT_THRESHOLD:
            shapley = self._exact_shapley(channels, coalition_value)
        else:
            shapley = self._mc_shapley(channels, coalition_value)

        # Normalise so attribution sums to grand coalition value
        grand_value = coalition_value(tuple(channels))
        total_sv = sum(shapley.values())
        if total_sv > 0:
            scale = grand_value / total_sv
            shapley = {ch: round(v * scale, 6) for ch, v in shapley.items()}

        logger.info("Attribution complete. Grand value: {:.4f}", grand_value)
        return shapley

    def marginal_contributions(self, campaigns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame with the marginal contribution of each channel
        when added last to the grand coalition.
        """
        channels = sorted(campaigns_df["channel"].dropna().unique().tolist())
        grand_value = self._coalition_value(campaigns_df, set(channels))
        rows = []
        for ch in channels:
            without = self._coalition_value(campaigns_df, set(channels) - {ch})
            rows.append(
                {
                    "channel": ch,
                    "grand_coalition_value": grand_value,
                    "value_without_channel": without,
                    "marginal_contribution": grand_value - without,
                }
            )
        return pd.DataFrame(rows).sort_values("marginal_contribution", ascending=False)

    # ------------------------------------------------------------------
    # Shapley implementations
    # ------------------------------------------------------------------

    def _exact_shapley(
        self,
        channels: list[str],
        coalition_value: callable,
    ) -> dict[str, float]:
        """Exact Shapley via summation over all permutations."""
        n = len(channels)
        sv: dict[str, float] = {ch: 0.0 for ch in channels}
        n_perms = math.factorial(n)

        for perm in itertools.permutations(channels):
            for i, ch in enumerate(perm):
                subset_with = tuple(perm[: i + 1])
                subset_without = tuple(perm[:i])
                marginal = coalition_value(subset_with) - coalition_value(subset_without)
                sv[ch] += marginal / n_perms

        return sv

    def _mc_shapley(
        self,
        channels: list[str],
        coalition_value: callable,
    ) -> dict[str, float]:
        """Monte-Carlo Shapley approximation via random permutation sampling."""
        n = len(channels)
        sv: dict[str, float] = {ch: 0.0 for ch in channels}
        channels_arr = np.array(channels)

        for _ in range(self.n_mc_samples):
            perm = self.rng.permutation(n)
            for i, idx in enumerate(perm):
                ch = channels_arr[idx]
                subset_with = tuple(channels_arr[perm[: i + 1]])
                subset_without = tuple(channels_arr[perm[:i]])
                marginal = coalition_value(subset_with) - coalition_value(subset_without)
                sv[ch] += marginal / self.n_mc_samples

        return sv

    # ------------------------------------------------------------------
    # Coalition value function
    # ------------------------------------------------------------------

    def _coalition_value(self, df: pd.DataFrame, subset: set[str]) -> float:
        """
        Value of a coalition of channels is the total *value_col* produced
        by those channels, optionally ROAS-weighted by *weight_col*.
        """
        if not subset:
            return 0.0
        mask = df["channel"].isin(subset)
        sub_df = df[mask]
        if sub_df.empty:
            return 0.0

        total_value = sub_df[self.value_col].sum()

        if self.weight_col and self.weight_col in sub_df.columns:
            total_spend = sub_df[self.weight_col].sum()
            if total_spend > 0:
                # Value as ROAS-like efficiency metric
                return float(total_value / total_spend * total_spend)
            return 0.0

        return float(total_value)
