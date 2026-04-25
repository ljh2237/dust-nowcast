from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training.trainer import train_deep_model, train_ml_baselines
from src.evaluation.product_reports import export_product_reports
from src.utils.config import load_config
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--models",
        type=str,
        default="dustriskformer,lstm,cnn_lstm",
        help="comma-separated deep models to train: dustriskformer,lstm,cnn_lstm,attn_tcn_lstm",
    )
    parser.add_argument("--skip_ml", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    set_seed(int(cfg["project"]["seed"]))

    out = {}
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        out[m] = train_deep_model(cfg, m)

    if not args.skip_ml:
        out["ml_baselines"] = train_ml_baselines(cfg)

    # Export standard product report if dustriskformer exists
    if (Path(cfg["paths"]["results_dir"]) / "predictions_detailed_dustriskformer.csv").exists():
        out["product_reports"] = export_product_reports(cfg["paths"]["results_dir"], model_name="dustriskformer")

    Path(cfg["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    with (Path(cfg["paths"]["results_dir"]) / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Training finished. Metrics saved to results/metrics.json")


if __name__ == "__main__":
    main()
