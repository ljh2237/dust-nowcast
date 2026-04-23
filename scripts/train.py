from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training.trainer import train_deep_model, train_ml_baselines
from src.utils.config import load_config
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    set_seed(int(cfg["project"]["seed"]))

    out = {}
    out["dustriskformer"] = train_deep_model(cfg, "dustriskformer")
    out["lstm"] = train_deep_model(cfg, "lstm")
    out["cnn_lstm"] = train_deep_model(cfg, "cnn_lstm")
    out["ml_baselines"] = train_ml_baselines(cfg)

    Path(cfg["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    with (Path(cfg["paths"]["results_dir"]) / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Training finished. Metrics saved to results/metrics.json")


if __name__ == "__main__":
    main()
