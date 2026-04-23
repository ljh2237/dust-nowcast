from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.models.dustriskformer import DustRiskFormer


class DustPredictor:
    def __init__(self, config: Dict):
        self.config = config
        self.processed_dir = Path(config["paths"]["processed_dir"])
        self.results_dir = Path(config["paths"]["results_dir"])

        self.dataset = np.load(self.processed_dir / "dataset_tensors.npz")
        with (self.processed_dir / "dataset_meta.json").open("r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.stations = self.meta["stations"]
        self.station_ids = [s["station_id"] for s in self.stations]
        self.horizons = self.meta["horizons"]
        self.risk_num_classes = int(self.meta.get("risk_num_classes", 3))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        in_dim = self.dataset["X"].shape[-1]
        static_dim = self.dataset["X_static"].shape[-1]
        h = len(self.horizons)
        self.model = DustRiskFormer(
            in_dim=in_dim,
            static_dim=static_dim,
            hidden_dim=int(config["model"]["hidden_dim"]),
            num_heads=int(config["model"]["num_heads"]),
            horizons=h,
            num_risk_classes=self.risk_num_classes,
            dropout=float(config["model"]["dropout"]),
        ).to(self.device)
        ckpt = Path(config["paths"]["checkpoints_dir"]) / "dustriskformer_best.pt"
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()

        self.X = self.dataset["X"]
        self.X_static = torch.tensor(self.dataset["X_static"], dtype=torch.float32, device=self.device)
        self.adj = torch.tensor(self.dataset["adj"], dtype=torch.float32, device=self.device)

    def _station_idx(self, station_id: str) -> int:
        if station_id not in self.station_ids:
            raise ValueError(f"Unknown station_id={station_id}. Allowed: {self.station_ids}")
        return self.station_ids.index(station_id)

    def predict_single(self, station_id: str, feature_overrides: Optional[Dict[str, float]] = None) -> Dict:
        si = self._station_idx(station_id)
        x = self.X[-1:].copy()
        if feature_overrides:
            feat_names = self.meta["features"]
            for k, v in feature_overrides.items():
                if k in feat_names:
                    fi = feat_names.index(k)
                    x[0, -1, si, fi] = float(v)

        with torch.no_grad():
            out = self.model(
                torch.tensor(x, dtype=torch.float32, device=self.device),
                self.X_static,
                self.adj,
            )

        wind = out["wind"].cpu().numpy()[0, si, :]
        risk_logits = out["risk_logits"].cpu().numpy()[0, si, :, :]
        risk_class = risk_logits.argmax(axis=-1)
        warn_prob = torch.sigmoid(out["warn_logit"]).cpu().numpy()[0, si, :]

        horizon_results = []
        for i, h in enumerate(self.horizons):
            horizon_results.append(
                {
                    "horizon_hour": int(h),
                    "wind_speed_pred": float(wind[i]),
                    "risk_level": int(risk_class[i]),
                    "warning_probability": float(warn_prob[i]),
                }
            )

        graph_attn = out["graph_attention"].cpu().numpy()[0, si, :]
        top_idx = np.argsort(graph_attn)[::-1][:3]
        key_nodes = [self.station_ids[i] for i in top_idx]

        return {
            "station_id": station_id,
            "results": horizon_results,
            "key_influential_stations": key_nodes,
        }

    def predict_batch(self, station_ids: List[str]) -> Dict:
        return {sid: self.predict_single(sid) for sid in station_ids}
