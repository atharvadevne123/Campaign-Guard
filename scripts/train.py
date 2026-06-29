"""
Train Campaign-Guard ensemble from scratch.
Usage: python scripts/train.py [--data-path PATH] [--model-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.ensemble.campaign_classifier import CampaignEnsemble
from pipeline.feature_engineering import CampaignFeatureEngineer
from loguru import logger


def generate_synthetic_data(n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    channels = ["social", "search", "email", "display", "affiliate"]
    industries = ["technology", "retail", "finance", "healthcare", "travel"]
    seasons = ["q1", "q2", "q3", "q4"]

    spend = rng.uniform(500, 50000, n)
    impressions = (spend * rng.uniform(10, 50, n)).astype(int)
    clicks = (impressions * rng.uniform(0.005, 0.08, n)).astype(int)
    conversions = (clicks * rng.uniform(0.01, 0.15, n)).astype(int)

    df = pd.DataFrame({
        "campaign_id":            [f"CMP-{i:05d}" for i in range(n)],
        "channel":                rng.choice(channels, n),
        "spend":                  spend,
        "impressions":            impressions,
        "clicks":                 clicks,
        "conversions":            conversions,
        "audience_size":          rng.integers(10_000, 5_000_000, n),
        "campaign_duration_days": rng.integers(1, 90, n),
        "industry":               rng.choice(industries, n),
        "season":                 rng.choice(seasons, n),
    })

    roi = (conversions / (spend + 1e-9)) * 1000
    df["roi_label"] = (roi > roi.quantile(0.5)).astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(description="Train Campaign-Guard ensemble")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to training data CSV/Parquet (default: synthetic)")
    parser.add_argument("--model-dir", type=str, default=str(ROOT / "models"),
                        help="Directory to save trained models")
    parser.add_argument("--n-synthetic", type=int, default=5000,
                        help="Number of synthetic samples if no data-path given")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate data
    if args.data_path:
        path = Path(args.data_path)
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        logger.info("Loaded {:,} rows from {}.", len(df), path)
    else:
        logger.info("No data path given. Generating {:,} synthetic campaigns.", args.n_synthetic)
        df = generate_synthetic_data(args.n_synthetic)

    label_col = "roi_label"
    if label_col not in df.columns:
        logger.warning("Label column '{}' not found. Deriving from conversions/spend ratio.", label_col)
        roi = df.get("conversions", 0) / (df.get("spend", 1) + 1e-9)
        df[label_col] = (roi > roi.quantile(0.5)).astype(int)

    # Feature engineering
    fe = CampaignFeatureEngineer()
    df_feat = fe.fit_transform(df)

    exclude = {label_col, "campaign_id", "channel", "industry", "season", "roi_tier"}
    feat_cols = [c for c in df_feat.select_dtypes(include="number").columns if c not in exclude]
    X = df_feat[feat_cols].fillna(0)
    y = df_feat[label_col].astype(int)

    logger.info("Training on {:,} samples with {} features.", len(X), len(feat_cols))

    # Train ensemble
    model = CampaignEnsemble()
    metrics = model.train(X, y)

    # Save artifacts
    model.save(model_dir / "campaign_ensemble.joblib")
    joblib.dump(fe, model_dir / "feature_engineer.joblib")
    (model_dir / "feature_cols.json").write_text(json.dumps(feat_cols))

    logger.success("Training complete. Artifacts saved to {}.", model_dir)
    logger.info("Metrics: {}", metrics)


if __name__ == "__main__":
    main()
