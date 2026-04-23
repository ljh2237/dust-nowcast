# Changelog

## 2026-04-23 - Productization Upgrade
- Upgraded Streamlit app to product pages:
  - 首页/数据说明/模型说明/实时预测/区域地图/历史回放/模型结果/可解释性/API页/部署页
- Added API metadata root and robust error responses.
- Added MC Dropout uncertainty output in inference.
- Added detailed prediction export and product reports:
  - `predictions_detailed_*.csv`
  - `station_level_metrics.csv`
  - `event_level_metrics.csv`
  - `event_metrics_summary.json`
  - `result_summary.md`
- Added deployment artifacts:
  - `Dockerfile`, `docker-compose.yml`, `render.yaml`, `Procfile`, `runtime.txt`, `Makefile`, `app.py`
- Added deliverable docs:
  - `DEPLOYMENT.md`, `MODEL_CARD.md`, `DATA_CARD.md`, `PRODUCT_OVERVIEW.md`, `API_REFERENCE.md`
