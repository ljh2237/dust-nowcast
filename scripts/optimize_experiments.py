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


def _mk_cfg(base: dict, *, epochs: int, loss: dict | None = None, eval_subset: str | None = None) -> dict:
    cfg = copy.deepcopy(base)
    cfg["training"]["epochs"] = int(epochs)
    if loss is not None:
        cfg["loss"] = loss
    if eval_subset:
        cfg["evaluation"] = {"test_subset": eval_subset}
    return cfg


def _flatten(exp: str, model: str, m: dict) -> dict:
    if "error" in m:
        return {
            "experiment": exp,
            "model": model,
            "mae": None,
            "rmse": None,
            "r2": None,
            "risk_f1": None,
            "warn_f1": None,
            "warn_roc_auc": None,
            "warn_pr_auc": None,
            "all_high_risk_recall": None,
            "spring_n": None,
            "spring_warn_f1": None,
            "spring_high_risk_recall": None,
            "highrisk_n": None,
            "highrisk_warn_f1": None,
        }
    s = m.get("subset_eval", {})
    all_s = s.get("all", {})
    sp_s = s.get("spring_3_5", {})
    hr_s = s.get("high_risk_subset", {})
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
        "all_high_risk_recall": all_s.get("high_risk_recall"),
        "spring_n": sp_s.get("n"),
        "spring_warn_f1": (sp_s.get("warning") or {}).get("f1"),
        "spring_high_risk_recall": sp_s.get("high_risk_recall"),
        "highrisk_n": hr_s.get("n"),
        "highrisk_warn_f1": (hr_s.get("warning") or {}).get("f1"),
    }


def _write_summary_markdown(out_dir: Path, df: pd.DataFrame) -> None:
    leaderboard = df.sort_values(["warn_pr_auc", "risk_f1"], ascending=False).copy()
    spring_rows = df[df["experiment"].str.contains("spring", case=False, na=False)].copy()
    best_global = leaderboard.iloc[0]
    best_biz = df.sort_values(["risk_f1", "warn_f1", "warn_pr_auc"], ascending=False).iloc[0]

    lines = [
        "# 模型优化与必要性总结（最终版）",
        "",
        "## 项目定位",
        "- 本系统定位为中国西北区域级 0-6 小时短临预测增强系统，不替代国家/省级业务预报。",
        "- 价值在于最后一公里辅助决策增强：局地高风险识别、预警触发判断、可解释输出、轻量部署。",
        "",
        "## 严格评估口径",
        "- 默认总体指标：按原始时间切分的 test 集评估。",
        "- 春季专项（严格子集）：仅在原始 test 集内筛选 3/4/5 月样本，避免泄漏。",
        "- 春季专项（严格留出）：训练/验证剔除春季，春季作为独立测试集（更苛刻）。",
        "",
        "## 关键结论",
        f"- 答辩主方案（研究增强版）建议：`{best_global['experiment']}`，PR-AUC={best_global['warn_pr_auc']:.3f}，Risk-F1={best_global['risk_f1']:.3f}。",
        f"- 业务落地方案建议：`{best_biz['experiment']}`，Warning-F1={best_biz['warn_f1']:.3f}，结构更轻、更稳定、便于部署。",
        "",
        "## 结果表（节选）",
        "```text",
        leaderboard[
            [
                "experiment",
                "model",
                "mae",
                "risk_f1",
                "warn_f1",
                "warn_roc_auc",
                "warn_pr_auc",
                "all_high_risk_recall",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## 春季专项说明",
        "```text",
        (spring_rows.to_string(index=False) if len(spring_rows) else "无春季专项实验记录"),
        "```",
        "",
        "## 必要性表达（可用于答辩）",
        "- 我们不与国家级系统做全面对抗，而是在中国西北重点站点与高发季节提供局地增强能力。",
        "- 在 0-6 小时短临、风险等级识别、预警触发判断等限定任务下，本系统具备明确业务价值。",
    ]
    (out_dir / "模型优化与必要性总结（最终版）.md").write_text("\n".join(lines), encoding="utf-8")


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
            "name": "exp_weighted_focal_drf_spring_from_test_strict",
            "model": "dustriskformer",
            "cfg": _mk_cfg(
                base,
                epochs=8,
                eval_subset="spring_strict_from_test",
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
            "name": "exp_weighted_focal_drf_spring_holdout_strict",
            "model": "dustriskformer",
            "cfg": _mk_cfg(
                base,
                epochs=8,
                eval_subset="spring_holdout_strict",
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
        try:
            metrics = train_deep_model(exp["cfg"], exp["model"])
        except Exception as e:
            metrics = {"model": exp["model"], "error": str(e)}
            print(f"[WARN] {exp['name']} failed: {e}")
        raw[exp["name"]] = metrics
        rows.append(_flatten(exp["name"], exp["model"], metrics))

        with (out_dir / f"{exp['name']}_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        src_pred = Path(exp["cfg"]["paths"]["results_dir"]) / f"predictions_detailed_{exp['model']}.csv"
        if src_pred.exists():
            dst = out_dir / f"{exp['name']}_predictions_detailed.csv"
            pd.read_csv(src_pred).to_csv(dst, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "experiment_summary.csv", index=False)
    with (out_dir / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    leaderboard = df[df["warn_pr_auc"].notna()].sort_values(["warn_pr_auc", "risk_f1", "warn_f1"], ascending=False)
    leaderboard.to_csv(out_dir / "business_leaderboard.csv", index=False)

    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))
    x = range(len(df))
    ax.plot(x, df["risk_f1"], marker="o", label="Risk-F1")
    ax.plot(x, df["warn_f1"], marker="o", label="Warn-F1")
    ax.plot(x, df["warn_pr_auc"], marker="o", label="Warn PR-AUC")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["experiment"], rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("CN-NW Optimization Experiments (Strict Evaluation Included)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "experiment_business_metrics.png", dpi=180)
    plt.close(fig)

    _write_summary_markdown(out_dir, df)
    print(f"\nSaved optimization artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
