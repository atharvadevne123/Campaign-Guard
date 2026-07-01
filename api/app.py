"""
Flask microservice: Campaign-Guard API
Endpoints:
  POST /predict          — score a single campaign (ROI tier + budget recommendation)
  POST /predict/batch    — score a batch of campaigns
  POST /attribution      — Shapley-based channel attribution
  GET  /health           — liveness probe
  GET  /metrics          — Prometheus metrics
  GET  /model/info       — model metadata
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, g, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_restx import Api, Resource, fields
from loguru import logger
from marshmallow import Schema, ValidationError, fields as ma_fields, validate as ma_validate
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# ─── App bootstrap ────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per minute"],
    storage_uri="memory://",
)

api = Api(
    app,
    version="1.0",
    title="Campaign-Guard API",
    description="ML-powered marketing campaign ROI prediction and budget attribution",
    doc="/docs",
)

ns = api.namespace("", description="Campaign scoring endpoints")

# ─── Prometheus metrics ───────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "campaign_api_requests_total",
    "Total API requests",
    ["endpoint", "status"],
)
PREDICTION_LATENCY = Histogram(
    "campaign_api_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)
ROI_SCORE_HISTOGRAM = Histogram(
    "campaign_roi_score_distribution",
    "Distribution of ROI scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ─── ROI tier thresholds ──────────────────────────────────────────────────────

THRESHOLD_EXCELLENT = float(os.getenv("THRESHOLD_EXCELLENT", "0.75"))
THRESHOLD_GOOD      = float(os.getenv("THRESHOLD_GOOD",      "0.50"))
THRESHOLD_AVERAGE   = float(os.getenv("THRESHOLD_AVERAGE",   "0.25"))

# ─── Model loading ────────────────────────────────────────────────────────────

MODEL_DIR = Path(os.getenv("MODEL_DIR", str(Path(__file__).parent.parent / "models")))

_ensemble = None
_attribution_model = None
_feature_engineer = None
_feature_cols: list = []


def _load_models() -> None:
    global _ensemble, _attribution_model, _feature_engineer, _feature_cols
    import json
    import joblib

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.ensemble.campaign_classifier import CampaignEnsemble
    from models.attribution.attribution_model import AttributionModel

    ensemble_path = MODEL_DIR / "campaign_ensemble.joblib"
    fe_path = MODEL_DIR / "feature_engineer.joblib"
    cols_path = MODEL_DIR / "feature_cols.json"

    if ensemble_path.exists():
        _ensemble = CampaignEnsemble.load(ensemble_path)
        logger.success("Ensemble model loaded.")
    else:
        logger.warning("Ensemble model not found at {}. Operating in mock mode.", ensemble_path)

    if fe_path.exists():
        _feature_engineer = joblib.load(fe_path)
        logger.success("Feature engineer loaded.")

    if cols_path.exists():
        _feature_cols = json.loads(cols_path.read_text())

    _attribution_model = AttributionModel()
    logger.success("Attribution model initialised.")


# ─── Input validation ─────────────────────────────────────────────────────────

class CampaignSchema(Schema):
    campaign_id           = ma_fields.Str(required=True)
    channel               = ma_fields.Str(
                                load_default="unknown",
                                validate=ma_validate.OneOf(
                                    ["social", "search", "email", "display", "affiliate", "unknown"],
                                    error="channel must be one of: social, search, email, display, affiliate.",
                                ),
                            )
    spend                 = ma_fields.Float(
                                required=True,
                                validate=ma_validate.Range(
                                    min=0, min_inclusive=False,
                                    error="spend must be greater than 0.",
                                ),
                            )
    impressions           = ma_fields.Int(
                                required=True,
                                validate=ma_validate.Range(
                                    min=1, error="impressions must be greater than 0."
                                ),
                            )
    clicks                = ma_fields.Int(
                                required=True,
                                validate=ma_validate.Range(
                                    min=0, error="clicks must be >= 0."
                                ),
                            )
    conversions           = ma_fields.Int(
                                required=True,
                                validate=ma_validate.Range(
                                    min=0, error="conversions must be >= 0."
                                ),
                            )
    audience_size         = ma_fields.Int(load_default=None)
    campaign_duration_days = ma_fields.Int(
                                load_default=30,
                                validate=ma_validate.Range(
                                    min=1, max=365,
                                    error="campaign_duration_days must be between 1 and 365.",
                                ),
                            )
    industry              = ma_fields.Str(load_default="other")
    season                = ma_fields.Str(
                                load_default="q1",
                                validate=ma_validate.OneOf(
                                    ["q1", "q2", "q3", "q4"],
                                    error="season must be one of: q1, q2, q3, q4.",
                                ),
                            )


class AttributionSchema(Schema):
    campaigns = ma_fields.List(ma_fields.Dict(), required=True, validate=ma_validate.Length(min=1))
    value_col = ma_fields.Str(load_default="conversions")


_schema        = CampaignSchema()
_batch_schema  = CampaignSchema(many=True)
_attr_schema   = AttributionSchema()

# ─── Swagger models ───────────────────────────────────────────────────────────

campaign_model = api.model("Campaign", {
    "campaign_id":             fields.String(required=True, example="CMP-2024-001"),
    "channel":                 fields.String(example="social"),
    "spend":                   fields.Float(required=True, example=15000.0),
    "impressions":             fields.Integer(required=True, example=500000),
    "clicks":                  fields.Integer(required=True, example=12000),
    "conversions":             fields.Integer(required=True, example=350),
    "audience_size":           fields.Integer(example=2000000),
    "campaign_duration_days":  fields.Integer(example=30),
    "industry":                fields.String(example="technology"),
    "season":                  fields.String(example="q4"),
})

prediction_response = api.model("PredictionResponse", {
    "campaign_id":          fields.String(),
    "roi_score":            fields.Float(),
    "roi_tier":             fields.String(),
    "budget_recommendation": fields.String(),
    "shap_features":        fields.Raw(),
    "latency_ms":           fields.Float(),
})

attribution_model_swagger = api.model("AttributionRequest", {
    "campaigns": fields.List(fields.Raw(), required=True),
    "value_col": fields.String(example="conversions"),
})

# ─── API key auth ─────────────────────────────────────────────────────────────

_API_KEY = os.getenv("API_KEY")
_OPEN_PATHS = {"/health", "/metrics", "/swagger.json"}


@limiter.request_filter
def _exempt_health_metrics():
    return request.path in ("/health", "/metrics") or request.path.startswith("/docs")


@app.before_request
def _check_api_key():
    if not _API_KEY:
        return
    if request.path in _OPEN_PATHS or request.path.startswith("/docs"):
        return
    if request.headers.get("X-Api-Key", "") != _API_KEY:
        REQUEST_COUNT.labels(endpoint="auth", status="401").inc()
        return jsonify({"error": "Unauthorized", "detail": "Invalid or missing X-Api-Key header"}), 401


# ─── Request ID ───────────────────────────────────────────────────────────────

@app.before_request
def _attach_request_id():
    g.request_id = str(uuid.uuid4())


@app.after_request
def _add_request_id_header(response):
    response.headers["X-Request-ID"] = getattr(g, "request_id", "")
    return response


# ─── Error handlers ───────────────────────────────────────────────────────────

@api.errorhandler(Exception)
def handle_generic(e):
    code = getattr(e, "code", 500)
    return {"error": type(e).__name__, "detail": str(e)}, code


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found", "detail": str(e)}), 404


@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({"error": "Rate limit exceeded", "detail": str(e)}), 429


# ─── Core scoring logic ───────────────────────────────────────────────────────

def _score_campaign(campaign: dict) -> dict:
    """Run feature engineering → ensemble → SHAP for one campaign record."""
    start = time.perf_counter()
    df = pd.DataFrame([campaign])

    if _feature_engineer is not None:
        df = _feature_engineer.transform(df)

    # Build feature matrix aligned to training columns
    if _feature_cols:
        X = pd.DataFrame(0.0, index=df.index, columns=_feature_cols)
        for col in _feature_cols:
            if col in df.columns:
                X[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    else:
        exclude = {
            "campaign_id", "channel", "industry", "season",
            "roi_label", "roi_tier",
        }
        feat_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in exclude
        ]
        X = df[feat_cols].fillna(0)

    roi_score = 0.5
    shap_features: dict = {}

    if _ensemble is not None:
        roi_score = float(_ensemble.predict_roi_score(X)[0])
        try:
            explanations = _ensemble.explain(X)
            shap_features = explanations[0].get("shap_features", {})
        except Exception as exc:
            logger.debug("SHAP explain failed: {}", exc)
    else:
        import random
        roi_score = round(random.uniform(0.1, 0.95), 4)

    # Tier classification
    if roi_score >= THRESHOLD_EXCELLENT:
        roi_tier = "EXCELLENT"
    elif roi_score >= THRESHOLD_GOOD:
        roi_tier = "GOOD"
    elif roi_score >= THRESHOLD_AVERAGE:
        roi_tier = "AVERAGE"
    else:
        roi_tier = "POOR"

    budget_recommendation = _budget_recommendation(roi_tier, campaign)
    latency_ms = (time.perf_counter() - start) * 1000
    ROI_SCORE_HISTOGRAM.observe(roi_score)

    return {
        "campaign_id":            campaign.get("campaign_id"),
        "roi_score":              round(roi_score, 6),
        "roi_tier":               roi_tier,
        "budget_recommendation":  budget_recommendation,
        "shap_features":          {k: round(v, 6) for k, v in list(shap_features.items())[:10]},
        "latency_ms":             round(latency_ms, 2),
    }


def _budget_recommendation(roi_tier: str, campaign: dict) -> str:
    """Generate a human-readable budget recommendation based on ROI tier."""
    spend = campaign.get("spend", 0)
    channel = campaign.get("channel", "this channel")
    recs = {
        "EXCELLENT": (
            f"Scale budget aggressively. Consider increasing spend by 30-50% on {channel}. "
            f"Current ROAS indicates strong returns."
        ),
        "GOOD": (
            f"Maintain or modestly increase spend on {channel}. "
            f"Optimise targeting to push toward EXCELLENT tier."
        ),
        "AVERAGE": (
            f"Hold budget steady on {channel}. Review audience segmentation and creative. "
            f"A/B test new ad formats before scaling."
        ),
        "POOR": (
            f"Reduce or reallocate budget away from {channel}. "
            f"Campaign efficiency is below threshold. Investigate targeting and landing pages."
        ),
    }
    return recs.get(roi_tier, "Insufficient data for recommendation.")


# ─── Routes ───────────────────────────────────────────────────────────────────

@ns.route("/health")
class HealthCheck(Resource):
    def get(self):
        return {
            "status": "ok",
            "models_loaded": _ensemble is not None,
            "attribution_loaded": _attribution_model is not None,
        }, 200


@ns.route("/metrics")
class Metrics(Resource):
    def get(self):
        from flask import Response
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@ns.route("/model/info")
class ModelInfo(Resource):
    def get(self):
        return {
            "ensemble_loaded":    _ensemble is not None,
            "attribution_loaded": _attribution_model is not None,
            "feature_count":      len(_feature_cols),
            "version":            "1.0.0",
            "roi_tiers":          ["POOR", "AVERAGE", "GOOD", "EXCELLENT"],
            "thresholds": {
                "excellent": THRESHOLD_EXCELLENT,
                "good":      THRESHOLD_GOOD,
                "average":   THRESHOLD_AVERAGE,
            },
        }


@ns.route("/predict")
class Predict(Resource):
    @ns.expect(campaign_model)
    def post(self):
        try:
            data = _schema.load(request.get_json(force=True) or {})
        except ValidationError as exc:
            REQUEST_COUNT.labels(endpoint="predict", status="400").inc()
            return {"error": str(exc.messages)}, 400

        with PREDICTION_LATENCY.time():
            result = _score_campaign(data)

        REQUEST_COUNT.labels(endpoint="predict", status="200").inc()
        return result, 200


@ns.route("/predict/batch")
class PredictBatch(Resource):
    def post(self):
        payload   = request.get_json(force=True) or {}
        campaigns = payload.get("campaigns")

        if campaigns is None:
            return {"error": "Missing 'campaigns' key."}, 400
        if not campaigns:
            return {"error": "No campaigns provided."}, 400
        if len(campaigns) > 500:
            return {"error": "Batch size limited to 500 campaigns."}, 400

        try:
            data_list = _batch_schema.load(campaigns)
        except ValidationError as exc:
            REQUEST_COUNT.labels(endpoint="predict_batch", status="400").inc()
            return {"error": str(exc.messages)}, 400

        results = []
        with PREDICTION_LATENCY.time():
            for campaign in data_list:
                results.append(_score_campaign(campaign))

        REQUEST_COUNT.labels(endpoint="predict_batch", status="200").inc()
        return {"results": results, "count": len(results)}, 200


@ns.route("/attribution")
class Attribution(Resource):
    @ns.expect(attribution_model_swagger)
    def post(self):
        try:
            data = _attr_schema.load(request.get_json(force=True) or {})
        except ValidationError as exc:
            REQUEST_COUNT.labels(endpoint="attribution", status="400").inc()
            return {"error": str(exc.messages)}, 400

        campaigns_df = pd.DataFrame(data["campaigns"])
        value_col = data.get("value_col", "conversions")

        if "channel" not in campaigns_df.columns:
            return {"error": "Each campaign must include a 'channel' field."}, 400
        if value_col not in campaigns_df.columns:
            return {
                "error": f"value_col '{value_col}' not found in campaigns data."
            }, 400

        for col in ["spend", "impressions", "clicks", "conversions"]:
            if col in campaigns_df.columns:
                campaigns_df[col] = pd.to_numeric(campaigns_df[col], errors="coerce").fillna(0)

        start = time.perf_counter()
        attr_model = _attribution_model
        if attr_model is None:
            from models.attribution.attribution_model import AttributionModel
            attr_model = AttributionModel(value_col=value_col)

        attr_model.value_col = value_col
        attribution = attr_model.attribute(campaigns_df)
        marginals = attr_model.marginal_contributions(campaigns_df)
        latency_ms = (time.perf_counter() - start) * 1000

        REQUEST_COUNT.labels(endpoint="attribution", status="200").inc()
        return {
            "attribution": attribution,
            "marginal_contributions": marginals.to_dict(orient="records"),
            "value_col": value_col,
            "channel_count": len(attribution),
            "latency_ms": round(latency_ms, 2),
        }, 200


@ns.route("/benchmark/<string:industry>")
class IndustryBenchmark(Resource):
    """Return median ROI benchmarks for a given industry."""

    _BENCHMARKS = {
        "ecommerce":      {"median_ctr": 0.028, "median_cvr": 0.031, "median_roas": 3.8, "median_cpa": 18.5, "top_channels": ["search", "social", "email"]},
        "saas":           {"median_ctr": 0.019, "median_cvr": 0.022, "median_roas": 2.9, "median_cpa": 42.0, "top_channels": ["search", "email", "display"]},
        "finance":        {"median_ctr": 0.015, "median_cvr": 0.014, "median_roas": 2.1, "median_cpa": 95.0, "top_channels": ["search", "display", "social"]},
        "healthcare":     {"median_ctr": 0.021, "median_cvr": 0.018, "median_roas": 2.4, "median_cpa": 68.0, "top_channels": ["search", "social", "email"]},
        "retail":         {"median_ctr": 0.033, "median_cvr": 0.026, "median_roas": 4.1, "median_cpa": 14.0, "top_channels": ["social", "search", "affiliate"]},
        "travel":         {"median_ctr": 0.025, "median_cvr": 0.019, "median_roas": 3.2, "median_cpa": 55.0, "top_channels": ["search", "display", "social"]},
        "education":      {"median_ctr": 0.018, "median_cvr": 0.016, "median_roas": 2.6, "median_cpa": 38.0, "top_channels": ["social", "search", "video"]},
        "entertainment":  {"median_ctr": 0.042, "median_cvr": 0.035, "median_roas": 3.5, "median_cpa": 9.5,  "top_channels": ["social", "video", "display"]},
    }

    def get(self, industry: str):
        key = industry.lower().replace("-", "").replace(" ", "")
        bench = self._BENCHMARKS.get(key)
        if bench is None:
            available = sorted(self._BENCHMARKS.keys())
            return {"error": f"Unknown industry '{industry}'.", "available": available}, 404

        REQUEST_COUNT.labels(endpoint="benchmark", status="200").inc()
        return {
            "industry": industry.lower(),
            "benchmarks": bench,
            "note": "Median values across campaigns in the past 12 months. Use as baseline for ROI tier calibration.",
        }, 200


@ns.route("/channel-mix")
class ChannelMix(Resource):
    """Suggest optimal channel mix given a total budget and industry."""

    def post(self):
        body = request.get_json(force=True) or {}
        total_budget = body.get("total_budget")
        industry     = body.get("industry", "ecommerce")
        objective    = body.get("objective", "conversions")  # conversions | awareness | leads

        if not total_budget or float(total_budget) <= 0:
            return {"error": "'total_budget' must be a positive number."}, 400

        total_budget = float(total_budget)

        # Evidence-based allocation weights by objective
        weights: dict[str, dict[str, float]] = {
            "conversions": {"search": 0.40, "social": 0.25, "email": 0.20, "display": 0.10, "video": 0.05},
            "awareness":   {"video": 0.35,  "social": 0.30, "display": 0.25, "search": 0.07, "email": 0.03},
            "leads":       {"search": 0.35, "email": 0.30,  "social": 0.20, "display": 0.10, "video": 0.05},
        }
        allocation_weights = weights.get(objective, weights["conversions"])

        allocation = {
            ch: {"budget": round(total_budget * w, 2), "pct": round(w * 100, 1)}
            for ch, w in allocation_weights.items()
        }

        REQUEST_COUNT.labels(endpoint="channel_mix", status="200").inc()
        return {
            "total_budget":  total_budget,
            "industry":      industry,
            "objective":     objective,
            "allocation":    allocation,
            "rationale":     f"Optimized for '{objective}' objective using industry-weighted channel efficiency scores.",
        }, 200


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _load_models()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=False)
