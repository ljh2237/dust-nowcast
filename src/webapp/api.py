from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.inference.predictor import DustPredictor
from src.utils.config import load_config


class PredictSingleRequest(BaseModel):
    station_id: str
    feature_overrides: Optional[Dict[str, float]] = None


class PredictBatchRequest(BaseModel):
    station_ids: List[str]


app = FastAPI(title="中国西北沙尘短临预测 API", version="0.2.0")
_config = load_config("configs/default.yaml")
_predictor = DustPredictor(_config)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/stations")
def stations() -> Dict:
    return {"stations": _predictor.stations}


@app.get("/metrics")
def metrics() -> Dict:
    p = Path(_config["paths"]["results_dir"]) / "metrics_dustriskformer.json"
    if p.exists():
        import json

        return json.loads(p.read_text(encoding="utf-8"))
    return {"message": "metrics file not found, run training first"}


@app.post("/predict/single")
def predict_single(req: PredictSingleRequest) -> Dict:
    return _predictor.predict_single(req.station_id, req.feature_overrides)


@app.post("/predict/batch")
def predict_batch(req: PredictBatchRequest) -> Dict:
    return _predictor.predict_batch(req.station_ids)
