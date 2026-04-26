# 机制约束下的多源时空融合沙尘暴短临预测系统（中国西北）

## 产品定位
本系统是面向中国西北重点沙源区及下游影响走廊的**区域级 0-6 小时短临预测产品**，输出：
- 风速预测（1h/3h/6h）
- 风险等级（4级）
- 预警概率
- 可解释性结果
- 不确定性区间（P10-P90）

服务对象：政府、交通、农业、园区、企业。  
系统角色：国家/省级业务预报的最后一公里增强，不替代业务预报。

## 区域与站点
- 区域：河西走廊-宁夏北部-陕北西段
- 站点：Yulin、Dingbian、Yinchuan、Lanzhou、Wuwei、Zhangye、Jiuquan、Jiayuguan

## 数据源与标签
- 站点观测：NOAA Global Hourly（优先）
- 回退/补强：Open-Meteo Archive（当 NOAA 不稳定时）
- 静态因子：海拔、地形起伏、沙源距离、沙源接近度
- 标签：风速、风险等级、预警阈值跨越；并输出事件级统计（命中率、起报提前误差、持续时长误差）

## 架构能力
- 时空融合：Temporal Attention + Graph Attention
- 多任务输出：回归 + 分类 + 概率
- 可解释性：时间/空间注意力 + 特征重要性
- 不确定性：MC Dropout 区间输出

## 模型优化（严格评估版）
- 研究增强版：`DustRiskFormer`（多任务重加权 + focal + 类别平衡）
- 业务落地版：`Attention-TCN-LSTM`（轻量、稳定、易部署）
- 严格评估口径：
  - 总体：原始 test 集
  - 春季专项（严格）：原始 test 集内筛选春季子集（若为空会明确标注）
  - 春季留出（严格）：训练/验证剔除春季，春季独立测试
- 优化结果文件：
  - `results/optimization/experiment_summary.csv`
  - `results/optimization/business_leaderboard.csv`
  - `results/optimization/模型优化与必要性总结（最终版）.md`

结论表达（答辩可用）：
在“中国西北重点沙源区及其下游走廊、0-6 小时短临、预警触发与高风险识别”限定条件下，
区域专项模型具备明确业务价值；系统定位为最后一公里辅助决策增强，不替代国家级业务预报。

## 产品页面
Streamlit 产品端已包含：
- 首页
- 数据说明页
- 模型说明页
- 实时/准实时预测页
- 区域地图页
- 历史回放页
- 模型结果页
- 可解释性页
- API 页
- 部署页

## 快速运行
```bash
pip install -r requirements.txt
python scripts/download_data.py --config configs/default.yaml
python scripts/build_dataset.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml --epochs 8
python scripts/explain.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml --model dustriskformer
```

## 启动服务
```bash
python scripts/run_api.py
python scripts/run_streamlit.py
```
- API: http://127.0.0.1:8000/docs
- Web: http://127.0.0.1:8501

## 一键容器运行
```bash
docker compose up --build
```

## 公开部署
优先推荐 Render / Streamlit Cloud / Hugging Face Spaces。  
详细步骤见：[DEPLOYMENT.md](./DEPLOYMENT.md)

## GitHub Pages 产品官网
- 官网地址（仓库开启 Pages 后）：`https://ljh2237.github.io/dust-nowcast/`
- 官网代码目录：`docs/`
- 运行时联动配置：`docs/assets/site_config.json`
  - `online_app_url`：在线体验入口
  - `api_doc_url`：公网 API 文档入口
  - `service_status`：首页服务状态提示

## 结果产物
- `results/metrics.json`
- `results/event_metrics_summary.json`
- `results/station_level_metrics.csv`
- `results/event_level_metrics.csv`
- `results/predictions_detailed_dustriskformer.csv`
- `results/explainability_summary.json`
- `results/plots/*.png`

## 文档清单
- [PRODUCT_OVERVIEW.md](./PRODUCT_OVERVIEW.md)
- [DATA_CARD.md](./DATA_CARD.md)
- [MODEL_CARD.md](./MODEL_CARD.md)
- [API_REFERENCE.md](./API_REFERENCE.md)
- [DEPLOYMENT.md](./DEPLOYMENT.md)
- [CHANGELOG.md](./CHANGELOG.md)
