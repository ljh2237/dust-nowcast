from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
import streamlit as st

from src.inference.predictor import DustPredictor
from src.utils.config import load_config


API_URL = "http://127.0.0.1:8000"
RESULTS_DIR = Path("results")
CONFIG = load_config("configs/default.yaml")

st.set_page_config(page_title="中国西北沙尘短临预测产品", layout="wide")
st.title("机制约束下的多源时空融合沙尘暴短临预测系统")


@st.cache_resource
def _local_predictor() -> DustPredictor:
    return DustPredictor(CONFIG)


def _get_stations() -> List[Dict]:
    try:
        r = requests.get(f"{API_URL}/stations", timeout=10)
        r.raise_for_status()
        return r.json().get("stations", [])
    except Exception:
        return _local_predictor().stations


def _predict_single(station_id: str, feature_overrides: Dict[str, float], mc_samples: int = 20) -> Dict:
    payload = {"station_id": station_id, "feature_overrides": feature_overrides, "mc_samples": mc_samples}
    try:
        r = requests.post(f"{API_URL}/predict/single", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return _local_predictor().predict_single(station_id, feature_overrides=feature_overrides, mc_samples=mc_samples)


def _load_json(path: Path) -> Dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


page = st.sidebar.radio(
    "产品导航",
    [
        "首页",
        "数据说明",
        "模型说明",
        "实时/准实时预测",
        "区域地图",
        "历史回放",
        "模型结果",
        "可解释性",
        "API 页",
        "部署页",
    ],
)

if page == "首页":
    st.subheader("中国西北 0-6 小时短时临近预测产品")
    st.markdown(
        """
- 区域：河西走廊-宁夏北部-陕北西段
- 时效：0-6 小时（可扩展 12 小时滚动）
- 输出：风速 + 风险等级 + 预警概率 + 可解释性
- 服务对象：政府、交通、农业、园区、企业
- 角色：国家/省级业务预报的“最后一公里”增强
        """
    )
    st.info("限定条件：适用于具备站点观测 + 背景场 + 静态因子的区域，不替代国家级业务系统。")

elif page == "数据说明":
    st.subheader("多源数据与标签体系")
    st.markdown(
        """
- 动态站点：NOAA 中国站点（优先）+ Open-Meteo 回退补强
- 背景场：Open-Meteo Archive（风场、湿度、气压、土壤湿度等）
- 静态因子：海拔、地形起伏、沙源距离、沙源接近度
- 标签：风速回归 + 4级风险 + 预警概率 + 事件代理标签
        """
    )
    m = _load_json(Path("data/raw/download_manifest.json"))
    if m:
        st.json(m)

elif page == "模型说明":
    st.subheader("模型架构与机制约束")
    st.markdown(
        """
- 时空融合：Temporal Attention + Graph Attention
- 多任务：风速回归、风险等级、预警概率
- 不确定性：MC Dropout 置信区间输出（P10-P90）
- 解释：时间注意力、空间注意力、特征重要性

理论机制层：
- 风场主导 + 干湿条件 + 沙源接近度 + 季节性因子

业务机制层：
- 阈值跨越识别（预警触发）
- 事件命中率与起报提前量评估
        """
    )

elif page == "实时/准实时预测":
    st.subheader("单站点实时/准实时预测")
    stations = _get_stations()
    if not stations:
        st.error("未找到站点列表，请先运行数据构建与训练。")
    else:
        station_map = {f"{s['station_name']} ({s['station_id']})": s["station_id"] for s in stations}
        station_key = st.selectbox("站点", list(station_map.keys()), index=0)
        station_id = station_map[station_key]

        col1, col2, col3 = st.columns(3)
        with col1:
            wind_override = st.number_input("当前风速输入(m/s)", min_value=0.0, value=7.0, step=0.5)
        with col2:
            rh_override = st.number_input("当前相对湿度(可选)", min_value=0.0, max_value=100.0, value=35.0, step=1.0)
        with col3:
            mc_samples = st.slider("不确定性采样次数", 10, 50, 20, 5)

        if st.button("开始预测"):
            out = _predict_single(station_id, {"wind_speed": wind_override, "relative_humidity": rh_override}, mc_samples)
            st.json(out)
            if "results" in out:
                st.dataframe(pd.DataFrame(out["results"]))

elif page == "区域地图":
    st.subheader("中国西北站点风险地图")
    stations = _get_stations()
    if stations:
        rows = []
        for s in stations:
            pred = _predict_single(s["station_id"], {}, mc_samples=15)
            h3 = next((r for r in pred.get("results", []) if int(r.get("horizon_hour", -1)) == 3), None)
            rows.append(
                {
                    "station_id": s["station_id"],
                    "station_name": s["station_name"],
                    "lat": s["lat"],
                    "lon": s["lon"],
                    "risk_3h": h3["risk_level"] if h3 else 0,
                    "warn_prob_3h": h3["warning_probability"] if h3 else 0.0,
                }
            )
        df = pd.DataFrame(rows)
        st.map(df.rename(columns={"lat": "latitude", "lon": "longitude"}))
        st.dataframe(df.sort_values("risk_3h", ascending=False))

elif page == "历史回放":
    st.subheader("历史回放与事件对比")
    detailed = RESULTS_DIR / "predictions_detailed_dustriskformer.csv"
    if detailed.exists():
        df = pd.read_csv(detailed)
        station_ids = sorted(df["station_id"].unique().tolist())
        sid = st.selectbox("选择站点", station_ids)
        h = st.selectbox("选择预测时效", sorted(df["horizon_hour"].unique().tolist()))
        n = st.slider("回放长度", 50, 500, 200, 50)
        s = df[(df["station_id"] == sid) & (df["horizon_hour"] == h)].sort_values("sample_idx").tail(n)
        st.line_chart(s[["y_wind_true", "y_wind_pred"]].reset_index(drop=True))
        st.line_chart(s[["y_warn_true", "y_warn_prob"]].reset_index(drop=True))
    else:
        st.warning("未找到历史回放文件，请先训练模型。")

elif page == "模型结果":
    st.subheader("主模型与 Baseline 对比")
    for fn in ["metrics.json", "event_metrics_summary.json", "metrics_baselines_ml.json"]:
        p = RESULTS_DIR / fn
        if p.exists():
            st.markdown(f"**{fn}**")
            st.json(_load_json(p))

    station_metrics = RESULTS_DIR / "station_level_metrics.csv"
    event_metrics = RESULTS_DIR / "event_level_metrics.csv"
    if station_metrics.exists():
        st.markdown("**station_level_metrics.csv**")
        st.dataframe(pd.read_csv(station_metrics))
    if event_metrics.exists():
        st.markdown("**event_level_metrics.csv**")
        st.dataframe(pd.read_csv(event_metrics))

elif page == "可解释性":
    st.subheader("三层解释体系")
    st.markdown("1) 变量级 2) 时间级 3) 空间级")
    ep = _load_json(RESULTS_DIR / "explainability_summary.json")
    if ep:
        st.json(ep)
        for _, p in ep.items():
            if Path(p).exists():
                st.image(p)

elif page == "API 页":
    st.subheader("在线 API 使用说明")
    st.code(
        """GET /health
GET /stations
GET /metadata
POST /predict/single
POST /predict/batch
"""
    )
    st.markdown("FastAPI 文档：`http://127.0.0.1:8000/docs`")
    st.json(
        {
            "sample_request": {
                "station_id": "53772099999_Yulin",
                "feature_overrides": {"wind_speed": 8.0, "relative_humidity": 30.0},
                "mc_samples": 20,
            }
        }
    )

elif page == "部署页":
    st.subheader("部署与运维")
    st.markdown(
        """
- Docker: `docker compose up --build`
- Streamlit Cloud / Render / HuggingFace Spaces 均提供配置模板
- 详细步骤见 `DEPLOYMENT.md`
        """
    )
