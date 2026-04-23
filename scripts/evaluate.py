from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluate_saved import evaluate_saved
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="dustriskformer")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = evaluate_saved(cfg, args.model)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
