from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.evaluation.metrics import binary_metrics, classification_metrics, regression_metrics


def _extract_events(labels: np.ndarray) -> List[tuple[int, int]]:
    events = []
    in_evt = False
    start = 0
    for i, v in enumerate(labels):
        if v == 1 and not in_evt:
            in_evt = True
            start = i
        if v == 0 and in_evt:
            events.append((start, i - 1))
            in_evt = False
    if in_evt:
        events.append((start, len(labels) - 1))
    return events


def _event_metrics_for_series(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    true_events = _extract_events(y_true.astype(int))
    pred_events = _extract_events(y_pred.astype(int))

    if not true_events:
        return {
            "event_hit_rate": 0.0,
            "onset_lead_error": 0.0,
            "duration_error": 0.0,
            "crossing_f1": float(binary_metrics(y_true, y_pred)["f1"]),
        }

    hits = 0
    onset_errs = []
    dur_errs = []
    for ts, te in true_events:
        overlaps = [(ps, pe) for ps, pe in pred_events if not (pe < ts or ps > te)]
        if overlaps:
            hits += 1
            ps, pe = overlaps[0]
            onset_errs.append(abs(ps - ts))
            dur_errs.append(abs((pe - ps + 1) - (te - ts + 1)))

    return {
        "event_hit_rate": float(hits / max(len(true_events), 1)),
        "onset_lead_error": float(np.mean(onset_errs) if onset_errs else 999.0),
        "duration_error": float(np.mean(dur_errs) if dur_errs else 999.0),
        "crossing_f1": float(binary_metrics(y_true, y_pred)["f1"]),
    }


def export_product_reports(results_dir: str | Path, model_name: str = "dustriskformer") -> Dict:
    results_dir = Path(results_dir)
    pred_path = results_dir / f"predictions_detailed_{model_name}.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing {pred_path}")

    df = pd.read_csv(pred_path)

    station_rows = []
    event_rows = []
    for sid, g in df.groupby("station_id"):
        station_rows.append(
            {
                "station_id": sid,
                **regression_metrics(g["y_wind_true"].values, g["y_wind_pred"].values),
                **{f"risk_{k}": v for k, v in classification_metrics(g["y_risk_true"].values, g["y_risk_pred"].values).items()},
                **{f"warn_{k}": v for k, v in binary_metrics(g["y_warn_true"].values, g["y_warn_pred"].values, g["y_warn_prob"].values).items()},
            }
        )

        e = _event_metrics_for_series(g.sort_values(["sample_idx", "horizon_hour"])["y_warn_true"].values, g.sort_values(["sample_idx", "horizon_hour"])["y_warn_pred"].values)
        e["station_id"] = sid
        event_rows.append(e)

    station_df = pd.DataFrame(station_rows).sort_values("station_id")
    event_df = pd.DataFrame(event_rows).sort_values("station_id")

    station_df.to_csv(results_dir / "station_level_metrics.csv", index=False)
    event_df.to_csv(results_dir / "event_level_metrics.csv", index=False)

    summary = {
        "model": model_name,
        "event_level_macro": {
            "event_hit_rate": float(event_df["event_hit_rate"].mean()),
            "onset_lead_error": float(event_df["onset_lead_error"].replace(999.0, np.nan).mean()),
            "duration_error": float(event_df["duration_error"].replace(999.0, np.nan).mean()),
            "crossing_f1": float(event_df["crossing_f1"].mean()),
        },
    }

    with (results_dir / "event_metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (results_dir / "result_summary.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Result Summary\n\n"
            f"- Model: {model_name}\n"
            f"- Event Hit Rate (macro): {summary['event_level_macro']['event_hit_rate']:.4f}\n"
            f"- Onset Lead Error (macro): {summary['event_level_macro']['onset_lead_error']:.4f}\n"
            f"- Duration Error (macro): {summary['event_level_macro']['duration_error']:.4f}\n"
            f"- Crossing F1 (macro): {summary['event_level_macro']['crossing_f1']:.4f}\n"
        )

    return summary
