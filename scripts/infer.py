from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.inference.predictor import DustPredictor
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--station-id", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    predictor = DustPredictor(cfg)
    out = predictor.predict_single(args.station_id)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
