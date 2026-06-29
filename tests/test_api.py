from __future__ import annotations

import json


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "ok"
        assert "models_loaded" in data

    def test_health_models_not_loaded(self, client):
        r = client.get("/health")
        assert r.get_json()["models_loaded"] is False


class TestModelInfo:
    def test_model_info_structure(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200
        data = r.get_json()
        for key in ("ensemble_loaded", "attribution_loaded", "feature_count", "version"):
            assert key in data

    def test_model_info_version(self, client):
        r = client.get("/model/info")
        assert r.get_json()["version"] == "1.0.0"

    def test_model_info_roi_tiers(self, client):
        data = client.get("/model/info").get_json()
        assert set(data["roi_tiers"]) == {"POOR", "AVERAGE", "GOOD", "EXCELLENT"}


class TestPredict:
    def test_predict_valid(self, client, patch_models, valid_campaign):
        r = client.post("/predict", data=json.dumps(valid_campaign),
                        content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert "roi_score" in data
        assert "roi_tier" in data
        assert "budget_recommendation" in data
        assert 0.0 <= data["roi_score"] <= 1.0
        assert data["roi_tier"] in ("EXCELLENT", "GOOD", "AVERAGE", "POOR")

    def test_predict_missing_campaign_id(self, client, patch_models):
        r = client.post("/predict",
                        data=json.dumps({"spend": 1000.0, "impressions": 10000, "clicks": 500, "conversions": 10}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_predict_missing_spend(self, client, patch_models, valid_campaign):
        payload = {k: v for k, v in valid_campaign.items() if k != "spend"}
        r = client.post("/predict", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_predict_spend_zero_rejected(self, client, patch_models, valid_campaign):
        payload = {**valid_campaign, "spend": 0.0}
        r = client.post("/predict", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_predict_impressions_zero_rejected(self, client, patch_models, valid_campaign):
        payload = {**valid_campaign, "impressions": 0}
        r = client.post("/predict", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_predict_invalid_season(self, client, patch_models, valid_campaign):
        payload = {**valid_campaign, "season": "q5"}
        r = client.post("/predict", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_predict_response_has_request_id(self, client, patch_models, valid_campaign):
        r = client.post("/predict", data=json.dumps(valid_campaign), content_type="application/json")
        assert "X-Request-ID" in r.headers

    def test_predict_empty_body_returns_400(self, client, patch_models):
        r = client.post("/predict", data=json.dumps({}), content_type="application/json")
        assert r.status_code == 400

    def test_predict_duration_out_of_range(self, client, patch_models, valid_campaign):
        payload = {**valid_campaign, "campaign_duration_days": 400}
        r = client.post("/predict", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400


class TestAttribution:
    def test_attribution_returns_channel_breakdown(self, client, patch_models, valid_campaign):
        payload = {"campaigns": [valid_campaign, {**valid_campaign, "channel": "search"}],
                   "value_col": "conversions"}
        r = client.post("/attribution", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert "attribution" in data
        assert "channel_count" in data

    def test_attribution_missing_campaigns(self, client, patch_models):
        r = client.post("/attribution", data=json.dumps({}), content_type="application/json")
        assert r.status_code == 400

    def test_attribution_empty_campaigns(self, client, patch_models):
        r = client.post("/attribution", data=json.dumps({"campaigns": []}),
                        content_type="application/json")
        assert r.status_code == 400


class TestBatch:
    def test_batch_predict_valid(self, client, patch_models, valid_campaign):
        payload = {"campaigns": [valid_campaign, {**valid_campaign, "campaign_id": "CMP-002"}]}
        r = client.post("/predict/batch", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert data["count"] == 2
        assert len(data["results"]) == 2

    def test_batch_too_large_rejected(self, client, patch_models, valid_campaign):
        payload = {"campaigns": [valid_campaign] * 501}
        r = client.post("/predict/batch", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_batch_missing_key(self, client, patch_models):
        r = client.post("/predict/batch", data=json.dumps({}), content_type="application/json")
        assert r.status_code == 400
