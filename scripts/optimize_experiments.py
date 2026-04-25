from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training.trainer import train_deep_model
from src.utils.config import load_config
from src.utils.seed import set_seed


def _mk_cfg(base: dict, *, epochs: int, loss: dict | None = None) -> dict:
    cfg = copy.deepcopy(base)
    cfg["training"]["epochs"] = int(epochs)
    if loss is not None:
        cfg["loss"] = loss
    return cfg


def _flatten(exp: str, model: str, m: dict) -> dict:
    s = m.get("subset_eval", {})
    return {
        "experiment": exp,
        "model": model,
        "mae": m["regression"]["mae"],
        "rmse": m["regression"]["rmse"],
        "r2": m["regression"]["r2"],
        "risk_f1": m["risk_classification"]["f1"],
        "warn_f1": m["warning_binary"]["f1"],
        "warn_roc_auc": m["warning_binary"]["roc_auc"],
        "warn_pr_auc": m["warning_binary"]["pr_auc"],
        "high_risk_recall_all": s.get("all", {}).get("high_risk_recall"),
        "spring_warn_f1": s.get("spring_3_5", {}).get("warning", {}).get("f1"),
        "spring_high_risk_recall": s.get("spring_3_5", {}).get("high_risk_recall"),
        "highrisk_subset_warn_f1": s.get("high_risk_subset", {}).get("warning", {}).get("f1"),
    }


def main() -> None:
    base = load_config("configs/default.yaml")
    set_seed(int(base["project"]["seed"]))
    out_dir = Path(base["paths"]["results_dir"]) / "optimization"
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "name": "exp_base_drf",
            "model": "dustriskformer",
            "cfg": _mk_cfg(base, epochs=8, loss={"alpha": 1.0, "beta": 1.0, "gamma": 1.0}),
        },
        {
            "name": "exp_weighted_drf",
            "model": "dustriskformer",
            "cfg": _mk_cfg(
                base,
                epochs=8,
                loss={
                    "alpha": 0.7,
                    "beta": 1.6,
                    "gamma": 1.8,
                    "use_balanced_risk_weights": True,
                    "warn_pos_weight": 1.8,
                    "warn_focal_gamma": 1.2,
                },
            ),
        },
        {
            "name": "exp_weighted_focal_drf",
            "model": "dustriskformer",
            "cfg": _mk_cfg(
                base,
                epochs=10,
                loss={
                    "alpha": 0.6,
                    "beta": 1.8,
                    "gamma": 2.0,
                    "use_balanced_risk_weights": True,
                    "warn_pos_weight": 2.2,
                    "warn_focal_gamma": 1.8,
                },
            ),
        },
        {
            "name": "exp_business_lite_attn_tcn_lstm",
            "model": "attn_tcn_lstm",
            "cfg": _mk_cfg(
                base,
                epochs=10,
                loss={
                    "alpha": 0.7,
                    "beta": 1.6,
                    "gamma": 1.8,
                    "use_balanced_risk_weights": True,
                    "warn_pos_weight": 1.6,
                    "warn_focal_gamma": 1.0,
                },
            ),
        },
        {
            "name": "exp_weighted_focal_drf_spring_eval",
            "model": "dustriskformer",
            "cfg": _mk_cfg(
                {
                    **base,
                    "evaluation": {"test_subset": "spring"},
                },
                epochs=8,
                loss={
                    "alpha": 0.6,
                    "beta": 1.8,
                    "gamma": 2.0,
                    "use_balanced_risk_weights": True,
                    "warn_pos_weight": 2.2,
                    "warn_focal_gamma": 1.8,
                },
            ),
        },
    ]

    rows = []
    raw = {}
    for exp in experiments:
        print(f"\n=== running {exp['name']} ({exp['model']}) ===")
        metrics = train_deep_model(exp["cfg"], exp["model"])
        raw[exp["name"]] = metrics
        rows.append(_flatten(exp["name"], exp["model"], metrics))

        # keep per-experiment snapshot
        with (out_dir / f"{exp['name']}_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # copy detailed predictions snapshot
        src_pred = Path(exp["cfg"]["paths"]["results_dir"]) / f"predictions_detailed_{exp['model']}.csv"
        if src_pred.exists():
            dst = out_dir / f"{exp['name']}_predictions_detailed.csv"
            pd.read_csv(src_pred).to_csv(dst, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "experiment_summary.csv", index=False)
    with (out_dir / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)

    # leaderboard by business-oriented metrics
    biz = df[["experiment", "model", "risk_f1", "warn_f1", "warn_pr_auc", "spring_warn_f1", "spring_high_risk_recall"]].copy()
    biz = biz.sort_values(["spring_warn_f1", "warn_pr_auc", "risk_f1"], ascending=False)
    biz.to_csv(out_dir / "business_leaderboard.csv", index=False)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    x = range(len(df))
    ax.plot(x, df["risk_f1"], marker="o", label="Risk-F1")
    ax.plot(x, df["warn_f1"], marker="o", label="Warn-F1")
    ax.plot(x, df["warn_pr_auc"], marker="o", label="Warn PR-AUC")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["experiment"], rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("CN-NW Optimization: Business Metrics")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "experiment_business_metrics.png", dpi=180)
    plt.close(fig)

    # markdown summary
    md = [
        "# 模型优化与必要性总结",
        "",
        "## 定位",
        "- 本项目是中国西北区域级 0-6 小时短临增强系统，不替代国家/省级业务预报。",
        "- 价值在于最后一公里辅助决策增强：局地高风险识别、快速集成、可解释输出。",
        "",
        "## 实验结论（自动生成）",
        "```text",
        biz.head(10).to_string(index=False),
        "```",
        "",
        "## 建议双方案",
        "- 研究增强版：加权多任务 DustRiskFormer（提升高风险与预警识别）。",
        "- 业务落地版：Attention-TCN-LSTM（轻量、稳定、易讲解）。",
    ]
    (out_dir / "模型优化与必要性总结.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\nSaved optimization artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
