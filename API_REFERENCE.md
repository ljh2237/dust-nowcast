# API Reference

Base URL (local): `http://127.0.0.1:8000`

## GET /
Product metadata and positioning.

## GET /health
Health check.

## GET /stations
Return station list.

## GET /metadata
Return config-level metadata.

## GET /metrics
Return latest model metrics.

## POST /predict/single
Request:
```json
{
  "station_id": "53772099999_Yulin",
  "feature_overrides": {"wind_speed": 8.0, "relative_humidity": 30.0},
  "mc_samples": 20
}
```
Response includes:
- wind speed forecast (1/3/6h)
- risk level
- warning probability
- p10/p90 uncertainty intervals
- key influential stations

## POST /predict/batch
Request:
```json
{
  "station_ids": ["53772099999_Yulin", "53614099999_Yinchuan"]
}
```
