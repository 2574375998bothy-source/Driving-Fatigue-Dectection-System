# 疲劳驾驶检测系统技术方案

## 项目定位

学生项目，面向答辩展示，核心目标：可运行的 demo + 可解释的模型决策可视化。

---

## 技术栈总览

| 模块 | 技术选型 |
|------|----------|
| 关键点提取 | MediaPipe Face Mesh (468点) |
| 特征计算 | 自实现 EAR / MAR / pitch / yaw |
| 时序建模 | Keras LSTM (32 units) |
| 可解释性 | SHAP DeepExplainer |
| 数据集 | NTHU Drowsy Driver Dataset / YawDD |
| 部署目标 | 边缘端（CPU推理，ARM兼容） |

---

## 目录结构建议

```
fatigue_detection/
├── data/
│   ├── raw/               # 原始视频
│   ├── features/          # 提取后的 CSV 特征
│   └── windows/           # 滑窗后的 npy 数组
├── src/
│   ├── extract_features.py   # MediaPipe 关键点 → EAR/MAR/pose
│   ├── build_dataset.py      # 滑窗切分 + 标注
│   ├── train.py              # LSTM 训练
│   ├── explain.py            # SHAP 可视化
│   └── inference.py          # 实时推理 pipeline
├── models/
│   └── lstm_fatigue.h5
└── README.md
```
