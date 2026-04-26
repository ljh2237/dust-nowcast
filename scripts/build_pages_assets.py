from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

def _safe_float(x: float | int | np.floating | np.integer) -> float:
    return float(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))


def _build_explainability(root: Path, data_meta: dict) -> dict:
    stations = data_meta["stations"]
    station_names = [s["station_name"] for s in stations]
    seq_len = int(data_meta.get("seq_len", 24))

    temporal = np.load(root / "results/temporal_attention.npy", allow_pickle=True)
    graph = np.load(root / "results/graph_attention.npy", allow_pickle=True)
    tab = pd.read_parquet(root / "data/processed/dataset_tabular.parquet")

    # Temporal: average attention over batch/query -> key-step importance
    t_imp = temporal.mean(axis=(0, 1))
    t_imp = t_imp / np.clip(t_imp.sum(), 1e-9, None)
    top_t_idx = np.argsort(t_imp)[::-1][:6]
    temporal_top = [
        {
            "step": int(i),
            "window_label": f"T-{max(seq_len - 1 - int(i), 0)}h",
            "score": _safe_float(t_imp[i]),
        }
        for i in top_t_idx
    ]

    # Spatial: average outgoing weights as contribution proxy
    g = graph.mean(axis=0)  # [N, N]
    out_contrib = g.mean(axis=1)
    out_contrib = out_contrib / np.clip(out_contrib.sum(), 1e-9, None)
    top_s_idx = np.argsort(out_contrib)[::-1][:5]
    spatial_top = [
        {
            "station_name": station_names[int(i)],
            "score": _safe_float(out_contrib[i]),
        }
        for i in top_s_idx
    ]

    # Variable: robust and reproducible proxy via abs Spearman corr to warn label
    candidate_cols = [c for c in tab.columns if c.startswith("last_") or c.startswith("mean_")]
    target = tab["y_warn_3h"].astype(float)
    pairs = []
    for c in candidate_cols:
        v = tab[c].astype(float)
        if v.std() < 1e-8:
            continue
        corr = v.corr(target, method="spearman")
        if pd.isna(corr):
            continue
        pairs.append((c, abs(float(corr))))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top_pairs = pairs[:8]
    s = sum(p[1] for p in top_pairs) or 1.0
    variable_top = [{"feature": k, "score": float(v / s)} for k, v in top_pairs]

    return {
        "temporal_top_windows": temporal_top,
        "spatial_top_stations": spatial_top,
        "variable_top_features": variable_top,
        "summary": {
            "temporal_focus_share_top3": _safe_float(sum(x["score"] for x in temporal_top[:3])),
            "spatial_focus_share_top3": _safe_float(sum(x["score"] for x in spatial_top[:3])),
            "variable_focus_share_top3": _safe_float(sum(x["score"] for x in variable_top[:3])),
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_meta = json.loads((root / "data/processed/dataset_meta.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "results/metrics.json").read_text(encoding="utf-8"))
    event_summary = json.loads((root / "results/event_metrics_summary.json").read_text(encoding="utf-8"))
    opt_csv = root / "results/optimization/experiment_summary.csv"
    optimization_rows = []
    recommend = {
        "research_enhanced": "dustriskformer",
        "business_lite": "attn_tcn_lstm",
        "notes": "默认按业务指标与模型复杂度综合推荐。",
    }
    if opt_csv.exists():
        opt_df = pd.read_csv(opt_csv)
        optimization_rows = opt_df.fillna("").to_dict(orient="records")
        valid = opt_df[opt_df["warn_pr_auc"].notna()].copy()
        if not valid.empty:
            # Prefer strict overall test experiments when choosing final display model
            strict_overall = valid[~valid["experiment"].str.contains("spring_holdout", case=False, na=False)]
            cand = strict_overall if not strict_overall.empty else valid
            cand = cand.copy()
            cand["biz_score"] = 0.45 * cand["risk_f1"] + 0.20 * cand["warn_f1"] + 0.35 * cand["warn_pr_auc"]
            top_research = cand.sort_values("biz_score", ascending=False).iloc[0]
            recommend["research_enhanced"] = str(top_research["experiment"])

            lite = valid[valid["experiment"].str.contains("business_lite", case=False, na=False)]
            if not lite.empty:
                recommend["business_lite"] = str(lite.iloc[0]["experiment"])
            else:
                top_lite = valid.sort_values(["warn_f1", "risk_f1"], ascending=False).iloc[0]
                recommend["business_lite"] = str(top_lite["experiment"])

    base_snapshot = root / "results/optimization/exp_base_drf_predictions_detailed.csv"
    detailed = pd.read_csv(base_snapshot if base_snapshot.exists() else root / "results/predictions_detailed_dustriskformer.csv")
    station_metrics = pd.read_csv(root / "results/station_level_metrics.csv")
    event_metrics = pd.read_csv(root / "results/event_level_metrics.csv")

    # latest snapshot by station+horizon
    latest_idx = detailed["sample_idx"].max()
    latest = detailed[detailed["sample_idx"] == latest_idx].copy()

    # historical replay per station/horizon (last 180 points)
    replay = {}
    for sid in sorted(detailed["station_id"].unique()):
        replay[sid] = {}
        for h in [1, 3, 6]:
            s = detailed[(detailed["station_id"] == sid) & (detailed["horizon_hour"] == h)].sort_values("sample_idx").tail(180)
            replay[sid][str(h)] = {
                "sample_idx": s["sample_idx"].astype(int).tolist(),
                "y_wind_true": s["y_wind_true"].astype(float).tolist(),
                "y_wind_pred": s["y_wind_pred"].astype(float).tolist(),
                "y_warn_true": s["y_warn_true"].astype(int).tolist(),
                "y_warn_prob": s["y_warn_prob"].astype(float).tolist(),
            }

    # base profiles for interactive browser-side demo predictor
    base_profiles = {}
    for sid in sorted(latest["station_id"].unique()):
        part = latest[latest["station_id"] == sid].sort_values("horizon_hour")
        base_profiles[sid] = {
            "horizons": part["horizon_hour"].astype(int).tolist(),
            "wind": part["y_wind_pred"].astype(float).tolist(),
            "warn_prob": part["y_warn_prob"].astype(float).tolist(),
            "risk": part["y_risk_pred"].astype(int).tolist(),
        }

    stations = []
    for s in data_meta["stations"]:
        sid = s["station_id"]
        sm = station_metrics[station_metrics["station_id"] == sid]
        em = event_metrics[event_metrics["station_id"] == sid]
        snap3 = latest[(latest["station_id"] == sid) & (latest["horizon_hour"] == 3)]
        stations.append(
            {
                "station_id": sid,
                "station_name": s["station_name"],
                "lat": float(s["lat"]),
                "lon": float(s["lon"]),
                "risk_3h": int(snap3["y_risk_pred"].iloc[0]) if not snap3.empty else 0,
                "warn_prob_3h": float(snap3["y_warn_prob"].iloc[0]) if not snap3.empty else 0.0,
                "mae": float(sm["mae"].iloc[0]) if not sm.empty else None,
                "event_hit_rate": float(em["event_hit_rate"].iloc[0]) if not em.empty else None,
            }
        )

    out = {
        "product": {
            "name": "机制约束下的多源时空融合沙尘暴短临预测系统",
            "region_cn": "中国西北（河西走廊-宁夏北部-陕北西段）",
            "horizons": [1, 3, 6],
            "tasks": ["wind", "risk", "warning_probability", "explainability"],
            "target_users": ["政府", "交通", "农业", "园区", "企业"],
        },
        "metrics": metrics,
        "event_summary": event_summary,
        "optimization": {
            "experiments": optimization_rows,
            "recommendation": recommend,
            "strict_eval_note": "春季专项优先采用严格口径：原始 test 内筛选或季节留出评估，避免泄漏。",
        },
        "stations": stations,
        "base_profiles": base_profiles,
        "replay": replay,
        "risk_labels": ["低", "中", "高", "极高"],
        "sensitivity": {
            "wind_gain": [0.32, 0.28, 0.24],
            "rh_penalty": [0.018, 0.016, 0.014],
        },
        "explainability": _build_explainability(root, data_meta),
    }

    out_path = root / "docs/assets/demo_data.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
