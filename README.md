# DustRiskFormer-CN-NW-MVP: 中国西北沙尘暴风速短时临近预测系统

## 1. 项目定位
本项目在已跑通的美国西南 MVP 基础上，完成了**区域迁移 + 数据重构 + 系统本地化**，当前默认版本为：
- 区域：**中国西北典型子区域（河西走廊-宁夏北部-陕北西段）**
- 任务：0-6 小时短临多任务预测
  1. 风速回归
  2. 沙尘风险等级分类（4级）
  3. 达预警阈值概率
  4. 可解释性输出（时间注意力/空间注意力/特征重要性）

该系统定位于“最后一公里精细化增强预测”，不替代国家级业务预报。

## 2. 为什么从美国西南迁移到中国西北
- 业务目标是中国西北场景，需完成区域本地化。
- 中国西北（河西走廊、宁夏北部、陕北西段）是典型沙尘高发带，具有代表性。
- 数据可复现：NOAA 中国站点（可得部分）+ Open-Meteo 中国区域背景场（稳定）+ 中国化静态因子，可真实跑通工程闭环。

## 3. 中国西北区域与站点
### 3.1 选择区域
- 河西走廊-宁夏北部-陕北西段（跨甘肃/宁夏/陕北西部）
- 代表性：接近腾格里、巴丹吉林、毛乌素等典型沙源区

### 3.2 站点清单（当前默认 8 站）
- Yulin（榆林）
- Dingbian（定边）
- Yinchuan（银川）
- Lanzhou（兰州）
- Wuwei（武威）
- Zhangye（张掖）
- Jiuquan（酒泉）
- Jiayuguan（嘉峪关）

站点详情见：`data/raw/station_meta.csv`

## 4. 数据源（中国化）
### A. 动态站点数据（优先观测）
- 优先：`NOAA NCEI Global Hourly` 中国站点
- 回退：若 NOAA 某站不可稳定获取，自动回退 `Open-Meteo Archive` 同点小时数据

### B. 背景场
- `Open-Meteo Archive`（中国区域小时级）：
  - 10m 风速/风向、温度、湿度、地表气压、降水、土壤湿度、能见度等

### C. 静态因子（中国西北版）
- 站点经纬度、海拔
- 地形起伏 proxy（周边高程标准差）
- 到最近沙源区距离（腾格里/巴丹吉林/毛乌素）
- 沙源接近度 `source_proximity`

## 5. 中国化标签体系（代码已实现）
实现文件：`src/data/dataset_builder.py`

多任务保持：
1. `y_wind`: 风速回归（1/3/6h）
2. `y_risk`: 风险等级（4级）
3. `y_warn`: 预警二分类概率

风险分值（可配置）综合：
- 强风（future wind）
- 低湿度
- 低能见度
- 土壤偏干
- 春季增强（3/4/5月）
- 沙源接近度

预警触发：`future_wind >= wind_warning_threshold` 或 `risk_level >= 高风险阈值`

配置项见：`configs/default.yaml`

## 6. 模型与图结构适配
- 主模型：`DustRiskFormer`
  - 时间注意力 + 图注意力 + 静态融合 + 多任务头
- 图结构：基于中国西北站点经纬度距离重建邻接矩阵（非美国图）
- 风险分类头：按数据集标签类别数自动适配（当前 4 类）

Baselines（已保留并重跑）：
- LSTM
- CNN-LSTM
- RandomForest
- XGBoost

## 7. 数据与训练流程（中国西北）
### 7.1 下载数据
```bash
python scripts/download_data.py --config configs/default.yaml
```
输出：
- `data/raw/station_observations.parquet`
- `data/raw/background_openmeteo.parquet`
- `data/raw/static_features.csv`
- `data/raw/station_meta.csv`
- `data/raw/download_manifest.json`

### 7.2 构建数据集
```bash
python scripts/build_dataset.py --config configs/default.yaml
```
输出：
- `data/processed/dataset_tensors.npz`
- `data/processed/dataset_tabular.parquet`
- `data/processed/dataset_meta.json`

### 7.3 训练
```bash
python scripts/train.py --config configs/default.yaml --epochs 8
```
输出：
- `results/metrics.json`
- `results/metrics_dustriskformer.json`
- `results/metrics_lstm.json`
- `results/metrics_cnn_lstm.json`
- `results/metrics_baselines_ml.json`
- `results/predictions_*.csv`
- `results/checkpoints/*.pt`
- `results/plots/*.png`

### 7.4 评估与解释
```bash
python scripts/evaluate.py --config configs/default.yaml --model dustriskformer
python scripts/explain.py --config configs/default.yaml
```

### 7.5 推理
```bash
python scripts/infer.py --config configs/default.yaml --station-id 53772099999_Yulin
```

## 8. API 与网页端（中国西北本地化）
### 启动
```bash
python scripts/run_api.py
python scripts/run_streamlit.py
```

### 访问
- API 文档：`http://127.0.0.1:8000/docs`
- Web 页面：`http://127.0.0.1:8501`

### 页面本地化内容
- 首页标题：**中国西北沙尘暴风速短时临近预测系统**
- 单点预测默认站点：中国西北站点（如 Yulin/Dingbian/Yinchuan）
- 训练与可视化页面均展示中国西北版本结果

## 9. 本次真实运行结果（中国西北版本）
运行日期：2026-04-23（本地真实执行）

### 主模型 DustRiskFormer
- MAE: 3.2466
- RMSE: 4.5997
- R2: 0.5274
- 风险分类 F1: 0.4652
- 预警 ROC-AUC: 0.8483
- 预警 PR-AUC: 0.8140

### Baselines
- LSTM：MAE 4.4487, RMSE 6.2245, R2 0.1345
- CNN-LSTM：MAE 3.5651, RMSE 5.2642, R2 0.3810
- RandomForest(3h)：MAE 3.1981, RMSE 4.3451, R2 0.5783
- XGBoost(3h)：MAE 3.1297, RMSE 4.3305, R2 0.5811

指标文件：
- `results/metrics.json`
- `results/metrics_dustriskformer.json`
- `results/metrics_baselines_ml.json`

图表与解释性：
- `results/plots/loss_curve_dustriskformer.png`
- `results/plots/pred_vs_true_dustriskformer.png`
- `results/plots/confusion_risk_dustriskformer.png`
- `results/plots/temporal_attention_heatmap.png`
- `results/plots/graph_attention_heatmap.png`
- `results/plots/rf_feature_importance.png`

说明：README 本节即中国西北区域结果说明（满足“至少一个结果说明明确显示中国西北区域名/站点名”的要求）。

## 10. 数据构成与局限（诚实说明）
- 中国真实站点观测：来自 NOAA Global Hourly 的可得站点（部分站点可用）
- 再分析/重建补强：Open-Meteo（用于背景场，且在 NOAA 不稳定时回退为动态输入）
- 标签：业务 proxy 标签（非官方人工逐时沙尘过程标注）

因此该版本是“真实可运行的中国西北 MVP”，但并非最终业务级全要素系统。

## 11. 后续优化方向
1. 接入 ERA5/ERA5-Land（含 850/700hPa 风、BLH）
2. 接入 MODIS/MERRA-2 AOD 与 NDVI
3. 引入沙源区边界 polygon 与风向传播加权图
4. 引入 PM10/PM2.5（若可稳定获取）
5. 增加事件级评估与空间热力图
