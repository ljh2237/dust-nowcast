from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataset_builder import build_processed_dataset
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    build_processed_dataset(cfg, cfg["paths"]["raw_dir"], cfg["paths"]["processed_dir"])
    print("Dataset build finished.")


if __name__ == "__main__":
    main()
