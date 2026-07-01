# Campaign-Guard

[![CI](https://github.com/atharvadevne123/Campaign-Guard/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvadevne123/Campaign-Guard/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Palantir Foundry](https://img.shields.io/badge/Palantir-Foundry-orange)](https://www.palantir.com/platforms/foundry/)

> ML-powered marketing campaign ROI prediction, budget attribution, and channel optimization API — production-ready with Palantir Foundry integration.

## Overview

Campaign-Guard scores incoming marketing campaigns in real time, returning a predicted ROI score, tier classification, budget recommendation, and Shapley-value attribution across channels. It is built around a soft-voting ensemble of XGBoost, LightGBM, and Random Forest calibrated with isotonic regression, wrapped in a Flask-RESTX REST API with full Swagger UI documentation.

## Features

- **ROI Prediction** — probability-calibrated score (0–1) with tier labels: `EXCELLENT / GOOD / AVERAGE / POOR`
- **Budget Recommendation** — data-driven spend guidance per campaign
- **Channel Attribution** — Shapley-value decomposition across `search / social / email / display / video / affiliate`
- **Batch Endpoint** — score up to 100 campaigns per request
- **SHAP Explanations** — per-prediction feature importances for model transparency
- **KS-Drift Monitoring** — automated covariate drift detection via Evidently
- **Palantir Foundry Integration** — bidirectional dataset sync (Parquet), model registry, and prediction logging
- **Automated Retraining** — Apache Airflow DAG: fetch → drift check → retrain → evaluate → push to Foundry → restart API
- **Prometheus Metrics** — `campaign_predictions_total`, `prediction_latency_seconds`, `roi_score_histogram`
- **MLflow Tracking** — experiment metadata, feature importances, calibration curves

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | Flask 3, Flask-RESTX, Gunicorn |
| ML Models | XGBoost 2, LightGBM 4, scikit-learn (Random Forest, isotonic calibration) |
| Explainability | SHAP |
| Data | pandas, NumPy |
| Validation | marshmallow |
| Rate Limiting | Flask-Limiter |
| Drift Monitoring | Evidently |
| Experiment Tracking | MLflow |
| Orchestration | Apache Airflow 2 |
| Data Platform | Palantir Foundry REST API (Parquet, transaction writes) |
| Observability | Prometheus, prometheus-client |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Containerisation | Docker, docker-compose |
| Testing | pytest, pytest-mock |
| Runtime | Python 3.11 |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Score a single campaign |
| `POST` | `/predict/batch` | Score up to 100 campaigns |
| `POST` | `/attribution` | Shapley channel attribution |
| `GET` | `/benchmark/<industry>` | Industry ROI benchmarks (CTR, CVR, ROAS, CPA) |
| `POST` | `/channel-mix` | Suggest optimal budget allocation by objective |
| `GET` | `/health` | Liveness probe |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/model/info` | Model version and feature metadata |
| `GET` | `/docs` | Swagger UI |

### POST `/predict` — Request

```json
{
  "campaign_id": "CAMP-2024-001",
  "channel": "social",
  "spend": 15000.0,
  "impressions": 500000,
  "clicks": 12000,
  "conversions": 320,
  "audience_size": 1200000,
  "campaign_duration_days": 14,
  "industry": "ecommerce",
  "season": "Q4"
}
```

### POST `/predict` — Response

```json
{
  "campaign_id": "CAMP-2024-001",
  "roi_score": 0.847,
  "roi_tier": "EXCELLENT",
  "budget_recommendation": 18500.0,
  "shap_features": {
    "ctr": 0.142,
    "cvr": 0.118,
    "spend": -0.063
  },
  "request_id": "req-abc123",
  "latency_ms": 18.4
}
```

## Project Structure

```
Campaign-Guard/
├── api/
│   ├── app.py               # Flask-RESTX application
│   └── wsgi.py
├── foundry/
│   └── foundry_client.py    # Palantir Foundry REST client
├── models/
│   ├── ensemble/
│   │   └── campaign_classifier.py   # XGBoost + LightGBM + RF ensemble
│   └── attribution/
│       └── attribution_model.py     # Shapley channel attribution
├── pipeline/
│   ├── feature_engineering.py       # CTR, CVR, CPC, CPA, interactions
│   └── airflow/
│       └── retrain_dag.py           # Daily retraining DAG
├── monitoring/
│   └── drift_monitor.py             # KS-drift detection
├── scripts/
│   └── train.py                     # Synthetic data generation + training
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   └── test_feature_engineering.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Palantir Foundry Integration

Campaign-Guard uses the Foundry REST API for:

- **Dataset Upload** — training data pushed as Parquet with transaction-based writes
- **Dataset Download** — latest labelled campaigns pulled for retraining
- **Model Registry** — ensemble artifacts registered with version and metrics
- **Prediction Logging** — every scored campaign is appended to the predictions dataset

Configure via `.env`:

```env
FOUNDRY_HOST=https://your-instance.palantirfoundry.com
FOUNDRY_TOKEN=your-bearer-token
CAMPAIGN_DATASET_RID=ri.foundry.main.dataset.xxxxxxxx
PREDICTIONS_DATASET_RID=ri.foundry.main.dataset.yyyyyyyy
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model (generates synthetic data)
python scripts/train.py

# Start API
gunicorn -b 0.0.0.0:8000 api.wsgi:app

# Or with Docker
docker-compose -f docker/docker-compose.yml up
```

## Running Tests

```bash
pytest tests/ -v
```

All 30 tests pass covering health, model info, predict, attribution, batch, and feature engineering.

## Airflow DAG

The `campaign_retrain` DAG runs daily at 02:00 UTC:

```
fetch_training_data_from_foundry
    → check_drift
    → retrain_model
    → evaluate_model
    → push_model_to_foundry
    → restart_api
```
