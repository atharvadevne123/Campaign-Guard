from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.feature_engineering import CampaignFeatureEngineer


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "campaign_id": "CMP-001",
            "channel": "social",
            "spend": 15000.0,
            "impressions": 500000,
            "clicks": 12000,
            "conversions": 350,
            "audience_size": 2000000,
            "campaign_duration_days": 30,
            "industry": "technology",
            "season": "q4",
        },
        {
            "campaign_id": "CMP-002",
            "channel": "search",
            "spend": 8000.0,
            "impressions": 200000,
            "clicks": 6000,
            "conversions": 200,
            "audience_size": 1000000,
            "campaign_duration_days": 14,
            "industry": "retail",
            "season": "q1",
        },
    ])


class TestCampaignFeatureEngineer:
    def test_fit_transform_returns_dataframe(self, sample_df):
        fe = CampaignFeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_ctr_feature_created(self, sample_df):
        fe = CampaignFeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert "ctr" in result.columns
        assert (result["ctr"] >= 0).all()

    def test_cvr_feature_created(self, sample_df):
        fe = CampaignFeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert "cvr" in result.columns

    def test_cpc_feature_created(self, sample_df):
        fe = CampaignFeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert "cpc" in result.columns

    def test_cpa_feature_created(self, sample_df):
        fe = CampaignFeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert "cpa" in result.columns

    def test_no_nans_in_output(self, sample_df):
        fe = CampaignFeatureEngineer()
        result = fe.fit_transform(sample_df)
        numeric_cols = result.select_dtypes(include="number").columns
        assert not result[numeric_cols].isnull().any().any()

    def test_transform_without_fit_raises(self, sample_df):
        fe = CampaignFeatureEngineer()
        with pytest.raises(RuntimeError):
            fe.transform(sample_df)

    def test_fit_then_transform(self, sample_df):
        fe = CampaignFeatureEngineer()
        fe.fit(sample_df)
        result = fe.transform(sample_df)
        assert len(result) == 2

    def test_channel_encoding_present(self, sample_df):
        fe = CampaignFeatureEngineer()
        result = fe.fit_transform(sample_df)
        channel_cols = [c for c in result.columns if "channel_" in c]
        assert len(channel_cols) > 0

    def test_season_encoding_present(self, sample_df):
        fe = CampaignFeatureEngineer()
        result = fe.fit_transform(sample_df)
        season_cols = [c for c in result.columns if "season_" in c]
        assert len(season_cols) > 0
