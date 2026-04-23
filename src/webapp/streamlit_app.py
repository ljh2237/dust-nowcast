from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"
RESULTS_DIR = Path("results")

st.set_page_config(page_title="DustRiskFormer", layout="wide")
st.title("基于多源时空深度学习的沙尘暴风速临近预测系统")

page = st.sidebar.radio(
    "页面导航",
    ["首页", "数据说明", "训练结果", "单点预测", "批量预测", "可解释性", "可视化结果"],
)


if page == "首页":
    st.markdown(
        """
本系统定位于**区域级 0-6 小时短时临近预测**，输出：
- 1h / 3h / 6h 风速预测
- 沙尘风险等级（0/1/2）
- 达到预警阈值概率
- 解释性结果（注意力与特征重要性）
        """
    )

elif page == "数据说明":
    st.subheader("数据源")
    st.write("1. Meteostat 地面站小时观测")
    st.write("2. Open-Meteo Archive 背景场（近似再分析）")
    st.write("3. Open-Meteo Elevation 静态地理因子")
    m = Path("data/raw/download_manifest.json")
    if m.exists():
        st.json(json.loads(m.read_text(encoding="utf-8")))

elif page == "训练结果":
    st.subheader("主模型指标")
    mp = RESULTS_DIR / "metrics_dustriskformer.json"
    if mp.exists():
        st.json(json.loads(mp.read_text(encoding="utf-8")))
    bp = RESULTS_DIR / "metrics_baselines_ml.json"
    if bp.exists():
        st.subheader("ML Baselines 指标")
        st.json(json.loads(bp.read_text(encoding="utf-8")))

elif page == "单点预测":
    st.subheader("单点预测")
    stations_resp = requests.get(f"{API_URL}/stations", timeout=20).json()
    station_ids = [s["station_id"] for s in stations_resp["stations"]]
    station_id = st.selectbox("站点", station_ids)
    wind_override = st.number_input("可选: 当前风速覆盖值 (m/s)", min_value=0.0, value=0.0, step=0.5)

    if st.button("预测"):
        payload = {"station_id": station_id, "feature_overrides": {"wind_speed": wind_override}}
        r = requests.post(f"{API_URL}/predict/single", json=payload, timeout=60)
        st.json(r.json())

elif page == "批量预测":
    st.subheader("批量预测")
    stations_resp = requests.get(f"{API_URL}/stations", timeout=20).json()
    station_ids = [s["station_id"] for s in stations_resp["stations"]]
    selected = st.multiselect("选择站点", station_ids, default=station_ids[:2])
    if st.button("批量预测"):
        r = requests.post(f"{API_URL}/predict/batch", json={"station_ids": selected}, timeout=60)
        st.json(r.json())

elif page == "可解释性":
    st.subheader("可解释性输出")
    ep = RESULTS_DIR / "explainability_summary.json"
    if ep.exists():
        data = json.loads(ep.read_text(encoding="utf-8"))
        st.json(data)
        for _, p in data.items():
            if Path(p).exists():
                st.image(p)
    else:
        st.warning("未找到解释性文件，请先运行 explainability")

elif page == "可视化结果":
    st.subheader("图表结果")
    plot_dir = RESULTS_DIR / "plots"
    if plot_dir.exists():
        for p in sorted(plot_dir.glob("*.png")):
            st.markdown(f"**{p.name}**")
            st.image(str(p))
    pred_path = RESULTS_DIR / "predictions_dustriskformer.csv"
    if pred_path.exists():
        st.subheader("预测样例")
        st.dataframe(pd.read_csv(pred_path).head(50))
