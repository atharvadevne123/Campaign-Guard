from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def client():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from api.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def patch_models(monkeypatch):
    import api.app as app_module

    mock_ensemble = MagicMock()
    mock_ensemble.predict_roi_score.return_value = [0.75]
    mock_ensemble.explain.return_value = [{"shap_features": {"ctr": 0.12, "cvr": 0.09}}]

    mock_attr = MagicMock()
    mock_attr.attribute.return_value = {"social": 0.45, "search": 0.30, "email": 0.25}
    mock_attr.marginal_contributions.return_value = __import__("pandas").DataFrame(
        [{"channel": "social", "marginal_value": 0.45}]
    )
    mock_attr.value_col = "conversions"

    mock_fe = MagicMock()
    mock_fe.transform.side_effect = lambda df: df

    monkeypatch.setattr(app_module, "_ensemble", mock_ensemble)
    monkeypatch.setattr(app_module, "_attribution_model", mock_attr)
    monkeypatch.setattr(app_module, "_feature_engineer", mock_fe)
    monkeypatch.setattr(app_module, "_feature_cols", [])
    return mock_ensemble


@pytest.fixture
def valid_campaign():
    return {
        "campaign_id": "CMP-TEST-001",
        "channel": "social",
        "spend": 15000.0,
        "impressions": 500000,
        "clicks": 12000,
        "conversions": 350,
        "audience_size": 2000000,
        "campaign_duration_days": 30,
        "industry": "technology",
        "season": "q4",
    }
