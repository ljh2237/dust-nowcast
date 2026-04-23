from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from src.evaluation.metrics import binary_metrics, classification_metrics, regression_metrics


def evaluate_saved(config: Dict, model_name: str = "dustriskformer") -> Dict:
    results_dir = Path(config["paths"]["results_dir"])
    pred_path = results_dir / f"predictions_{model_name}.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}. Run training first.")

    df = pd.read_csv(pred_path)
    out = {
        "model": model_name,
        "regression": regression_metrics(df["y_wind_true"].values, df["y_wind_pred"].values),
        "risk_classification": classification_metrics(df["y_risk_true"].values, df["y_risk_pred"].values),
        "warning_binary": binary_metrics(df["y_warn_true"].values, df["y_warn_pred"].values, df["y_warn_prob"].values),
    }

    with (results_dir / f"metrics_eval_{model_name}.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out
