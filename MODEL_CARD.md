# Model Card

## Model Name
DustRiskFormer-v2 (productized CN-NW variant)

## Scope
- Region: China Northwest (Hexi Corridor - Northern Ningxia - Western Shaanbei)
- Horizon: 1h / 3h / 6h
- Tasks:
  - Wind speed regression
  - Dust risk classification (4 classes)
  - Warning probability
  - Explainability outputs

## Inputs
- Dynamic station sequence features
- Background reanalysis/proxy features
- Static geographic/source features

## Outputs
- `wind_speed_pred`
- `risk_level`
- `warning_probability`
- `wind_speed_p10/p90` and `warning_probability_p10/p90` via MC Dropout

## Training Objective
- Multi-task loss: regression + risk classification + threshold crossing BCE

## Intended Use
- Last-mile decision support for government/traffic/agriculture/parks/enterprise.

## Not Intended Use
- Not a replacement of national/provincial operational forecasting.

## Risks and Limitations
- Partial station fallback to Open-Meteo in data-sparse conditions.
- Proxy labels are not official manually labeled dust-event records.

## Performance Snapshot (latest run)
See `results/metrics_dustriskformer.json` and `results/metrics.json`.
