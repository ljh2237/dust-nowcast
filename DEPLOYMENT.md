# Deployment Guide

## Product
中国西北沙尘暴风速短时临近预测系统（0-6小时）

## 1. Local Docker Deployment
```bash
docker compose up --build
```
- Web: http://127.0.0.1:8501
- API: http://127.0.0.1:8000/docs

## 2. Render (Recommended Public Demo)
### Files already prepared
- `Dockerfile`
- `render.yaml`
- `Procfile`

### Fastest steps (10-20 min)
1. Push repository to GitHub.
2. Login [Render](https://render.com) and create **Web Service** from this repo.
3. Environment: Docker.
4. Auto-detected command from Dockerfile.
5. Wait for build and open generated public URL.

Expected homepage should display:
- 中国西北
- 0-6 小时短临预测
- 风速 + 风险等级 + 预警概率 + 可解释性

## 3. Streamlit Community Cloud
1. Push repo to GitHub.
2. Create app from `src/webapp/streamlit_app.py`.
3. Ensure `requirements.txt` is used.
4. Deploy and get public URL.

## 4. Hugging Face Spaces (Streamlit)
1. Create Streamlit Space.
2. Upload this repo (or mirror).
3. Entry uses `app.py`.
4. Start Space and verify UI.

## 5. Environment Variables
Current version can run with defaults. Optional vars:
- `API_URL` (if frontend points to external API)
- `PYTHONUNBUFFERED=1`

## 6. Health Checks
- API: `GET /health`
- Web: `GET /`

## 7. Update Deployment
1. Re-train and refresh `results/` artifacts.
2. Commit changes.
3. Push to deployment branch.
4. Platform auto-redeploys.
