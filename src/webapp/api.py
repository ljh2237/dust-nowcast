from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.inference.predictor import DustPredictor
from src.utils.config import load_config


class PredictSingleRequest(BaseModel):
    station_id: str
    feature_overrides: Optional[Dict[str, float]] = None
    mc_samples: Optional[int] = 20


class PredictBatchRequest(BaseModel):
    station_ids: List[str]


app = FastAPI(title="中国西北沙尘短临预测 API", version="0.2.0")
_config = load_config("configs/default.yaml")
_predictor = DustPredictor(_config)


@app.exception_handler(Exception)
def handle_exception(_, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc), "type": exc.__class__.__name__})


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> Dict:
    return {
        "product": "机制约束下的多源时空融合沙尘暴短临预测系统",
        "region": _config["region"].get("display_name_cn", _config["region"].get("display_name", _config["region"]["name"])),
        "horizons": _config["dataset"]["horizons"],
        "tasks": ["wind_regression", "risk_classification", "warning_probability", "explainability"],
        "target_users": ["government", "traffic", "agriculture", "industrial_park", "enterprise"],
    }


@app.get("/stations")
def stations() -> Dict:
    return {"stations": _predictor.stations}


@app.get("/metadata")
def metadata() -> Dict:
    return {
        "project": _config["project"]["name"],
        "region": _config["region"],
        "dataset": {"seq_len": _config["dataset"]["seq_len"], "horizons": _config["dataset"]["horizons"]},
    }


@app.get("/metrics")
def metrics() -> Dict:
    p = Path(_config["paths"]["results_dir"]) / "metrics_dustriskformer.json"
    if p.exists():
        import json

        return json.loads(p.read_text(encoding="utf-8"))
    return {"message": "metrics file not found, run training first"}


@app.post("/predict/single")
def predict_single(req: PredictSingleRequest) -> Dict:
    try:
        return _predictor.predict_single(req.station_id, req.feature_overrides, mc_samples=int(req.mc_samples or 20))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch")
def predict_batch(req: PredictBatchRequest) -> Dict:
    try:
        return _predictor.predict_batch(req.station_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
