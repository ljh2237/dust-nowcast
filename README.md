# DustRiskFormer-MVP: 基于多源时空深度学习的沙尘暴风速临近预测系统

## 1. 项目定位与目标
本项目是**区域级、0-6 小时短时临近预测系统**，面向沙尘高发区域的“最后一公里增强预测”，不替代国家级业务预报。

输出目标（1h / 3h / 6h）：
1. 风速回归值
2. 沙尘风险等级分类（0/1/2）
3. 达到预警阈值概率
4. 可解释性结果（时间注意力、空间注意力、特征重要性）

## 2. 区域与数据源选择（MVP）
### 区域选择
- **美国西南（US Southwest）**：Phoenix / Tucson / El Paso / Albuquerque / Las Vegas
- 原因：沙尘与强风现象较明显，公开数据接口稳定，便于真实复现与自动化下载。
- 方法可迁移到中国西北（只需替换站点与数据接口）。

### 数据源
1. `NOAA NCEI Global Hourly` 小时站点数据（动态站点时序）
2. `Open-Meteo Archive API` 小时背景场（近似再分析/NWP代理）
3. `Open-Meteo Elevation API` 静态地理因子
4. 预留 ERA5/CDS 扩展接口（`.env.example`）

## 3. 标签构建规则（代码实现）
代码位置：`src/data/dataset_builder.py`

- 任务1：风速回归 `y_wind(h)`
- 任务2：风险等级 `y_risk(h)`，由 dust proxy score 构造：
  - `score = 0.55*norm(future_wind) + 0.2*(RH<30) + 0.15*(visibility<5000m) + 0.1*(soil_moisture<0.15)`
  - 阈值：`[0.35, 0.65]` 映射到 `0/1/2`
- 任务3：预警二分类 `y_warn(h)`
  - `future_wind >= 12m/s` 或 `risk_level >= 2`

> 如果某站点缺少 visibility/soil moisture，代码会自动使用背景场并插值补全。

## 4. 模型方案
主模型：`DustRiskFormer`（`src/models/dustriskformer.py`）
- 时间分支：Multihead Temporal Attention
- 空间分支：Graph Attention（站点图）
- 多模态融合：站点动态 + 背景场 + 静态地理
- 多任务输出：风速回归 + 风险等级 + 预警概率
- 可解释性：输出 temporal/graph attention

Baselines：
1. LSTM
2. CNN-LSTM
3. XGBoost（若安装可用）
4. RandomForest

## 5. 项目目录结构
```text
project_root/
  README.md
  requirements.txt
  environment.yml
  .env.example
  configs/
    default.yaml
  data/
    raw/
    interim/
    processed/
  notebooks/
  scripts/
    download_data.py
    build_dataset.py
    train.py
    evaluate.py
    infer.py
    explain.py
    run_api.py
    run_streamlit.py
  src/
    data/
      downloader.py
      dataset_builder.py
    features/
    models/
      dustriskformer.py
      baselines.py
    training/
      datasets.py
      trainer.py
    evaluation/
      metrics.py
      plots.py
      evaluate_saved.py
    inference/
      predictor.py
    explainability/
      run_explainability.py
    webapp/
      api.py
      streamlit_app.py
    utils/
      config.py
      seed.py
  results/
  tests/
```

## 6. 一键运行（MVP）
### 6.1 安装依赖
```bash
pip install -r requirements.txt
```

### 6.2 下载真实数据
```bash
python scripts/download_data.py --config configs/default.yaml
```
输出：
- `data/raw/station_observations.parquet`
- `data/raw/background_openmeteo.parquet`
- `data/raw/static_features.csv`
- `data/raw/station_meta.csv`
- `data/raw/download_manifest.json`

### 6.3 构建训练数据
```bash
python scripts/build_dataset.py --config configs/default.yaml
```
输出：
- `data/processed/dataset_tensors.npz`
- `data/processed/dataset_tabular.parquet`
- `data/processed/dataset_meta.json`

### 6.4 训练与评估
```bash
python scripts/train.py --config configs/default.yaml
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

### 6.5 解释性输出
```bash
python scripts/explain.py --config configs/default.yaml
```
输出：
- `results/explainability_summary.json`
- `results/plots/temporal_attention_heatmap.png`
- `results/plots/graph_attention_heatmap.png`
- `results/plots/rf_feature_importance.png`

### 6.6 推理
```bash
python scripts/infer.py --station-id <station_id> --config configs/default.yaml
```

### 6.7 启动 API + 网页
终端1：
```bash
python scripts/run_api.py
```
终端2：
```bash
python scripts/run_streamlit.py
```
访问：
- API 文档: `http://127.0.0.1:8000/docs`
- Web 页面: `http://127.0.0.1:8501`

## 7. 评估指标
- 回归：MAE / RMSE / R2
- 多分类：Accuracy / Precision / Recall / F1
- 预警二分类：Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC
- 混淆矩阵与预测曲线见 `results/plots`

## 10. 已执行实验结果（真实运行，2026-04-23）
运行命令：
```bash
python scripts/download_data.py --config configs/default.yaml
python scripts/build_dataset.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml --epochs 8
python scripts/explain.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml --model dustriskformer
```

关键结果（见 `results/metrics*.json`）：
- `DustRiskFormer`：MAE=3.6625, RMSE=5.2545, R2=0.2465, Risk-F1=0.5527, Warning-ROC-AUC=0.7755
- `LSTM`：MAE=4.1955, RMSE=5.6329, R2=0.1341, Risk-F1=0.4409
- `CNN-LSTM`：MAE=3.6941, RMSE=5.1621, R2=0.2728, Risk-F1=0.5914
- `XGBoost (3h)`：MAE=3.5037, RMSE=4.7465, R2=0.3850
- `RandomForest (3h)`：MAE=3.5177, RMSE=4.7645, R2=0.3803

图表路径：
- `results/plots/loss_curve_dustriskformer.png`
- `results/plots/pred_vs_true_dustriskformer.png`
- `results/plots/confusion_risk_dustriskformer.png`
- `results/plots/temporal_attention_heatmap.png`
- `results/plots/graph_attention_heatmap.png`
- `results/plots/rf_feature_importance.png`

## 8. 已知局限
1. MVP 背景场使用 Open-Meteo（可视作再分析代理），非完整 ERA5 全层场。
2. 遥感 AOD/NDVI 在本版为扩展接口，未纳入主训练流。
3. 标签使用 dust-event proxy，不等同于官方沙尘过程人工标注。

## 9. 第二阶段扩展方向
1. 接入 ERA5（u/v 10m, 850/700hPa, BLH, soil moisture）
2. 接入 MODIS/MERRA-2 AOD 与 NDVI
3. 引入更强图时空模型（TFT/PatchTST + GATv2）
4. 增加事件级评估与区域热力图
5. 增加私有化部署（Docker + CI/CD + Model Registry）
