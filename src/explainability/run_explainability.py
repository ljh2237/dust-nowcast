from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation.plots import save_attention_heatmap, save_feature_importance


def run_explainability(results_dir: str | Path, processed_dir: str | Path) -> dict:
    results_dir = Path(results_dir)
    processed_dir = Path(processed_dir)
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)

    out = {}

    t_attn_path = results_dir / "temporal_attention.npy"
    if t_attn_path.exists():
        t_attn = np.load(t_attn_path)
        # [B*N, T, T] in this implementation; take first sample
        if t_attn.ndim == 3:
            mat = t_attn[0]
            save_attention_heatmap(mat, results_dir / "plots" / "temporal_attention_heatmap.png", "Temporal Attention")
            out["temporal_attention_plot"] = str(results_dir / "plots" / "temporal_attention_heatmap.png")

    g_attn_path = results_dir / "graph_attention.npy"
    if g_attn_path.exists():
        g_attn = np.load(g_attn_path)
        # [B, N, N]
        if g_attn.ndim == 3:
            mat = g_attn[0]
            save_attention_heatmap(mat, results_dir / "plots" / "graph_attention_heatmap.png", "Graph Attention")
            out["graph_attention_plot"] = str(results_dir / "plots" / "graph_attention_heatmap.png")

    rf_reg_path = results_dir / "rf_reg.pkl"
    tab_path = processed_dir / "dataset_tabular.parquet"
    if rf_reg_path.exists() and tab_path.exists():
        rf_reg = joblib.load(rf_reg_path)
        tab = pd.read_parquet(tab_path)
        feats = [c for c in tab.columns if c.startswith("last_") or c.startswith("mean_") or c.startswith("std_") or c.startswith("static_")]
        # MultiOutputRegressor(RandomForestRegressor)
        est = rf_reg.estimators_[0]
        importances = est.feature_importances_
        save_feature_importance(feats, importances, results_dir / "plots" / "rf_feature_importance.png", "RF Feature Importance")
        out["rf_feature_importance_plot"] = str(results_dir / "plots" / "rf_feature_importance.png")

    with (results_dir / "explainability_summary.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return out
