# 驾驶员疲劳检测系统 (Driver Fatigue Detection System)

这是一个基于 Python, OpenCV, MediaPipe 和 Keras 的驾驶员疲劳检测系统。系统包含了带有 Tkinter 界面的前端客户端以及可直接本地运行的实时检测脚本。

## 系统功能

- 👤 **身份验证 (Face Verification)**: 采集驾驶员面部信息，识别是否为授权用户。
- 👁 **眼睛纵横比 (EAR) 分析**: 实时监测眼睛闭合情况，识别是否打瞌睡。
- 👄 **嘴巴纵横比 (MAR) 分析**: 检测打哈欠行为，提前预警疲劳。
- 🧠 **头部姿态异常检测**: 监控驾驶员是否转移视线或低头。
- 🚨 **防遮挡防盗报警**: 摄像头被遮挡或检测到未授权面部时触发本地报警。

---

## 安装与配置

### 1. 环境依赖

系统要求：Python 3.8 或以上版本，并且需要一个可用的摄像头（USB 摄像头或笔记本自带摄像头）。

初始化虚拟环境并安装依赖：

```bash
# 创建并激活虚拟环境 (Windows)
python -m venv venv
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置本地 `.keras` 模型路径

系统部分核心脚本（用于进行图像分类或特征提取）需要加载预训练好的 `.keras` 模型文件（如 `eye_classifier.keras`）。请确保你拥有该模型文件，并在以下脚本中修改相应的模型路径，使其指向你本地存放 `.keras` 模型的绝对或相对路径：

- **`scr/integrated_system.py`**:
  打开此文件，找到约第 113 行附近的 `model_path`，将其修改为你的模型路径：
  ```python
  model_path = Path(r"D:\你的路径\eye_classifier.keras")
  ```
- **`scr/evaluate_system.py`**:
  打开此文件，找到约第 110 行附近的 `model_path`，修改路径。
- **`scr/evaluate_eye_cnn.py`**:
  打开此文件，找到约第 23 行的 `MODEL_PATH` 并修改。
- **`scr/live_fatigue_demo.py`**:
  此脚本默认去 `项目根目录/outputs/eye_classifier.keras` 读取模型。您可以将模型放置在 `outputs/` 文件夹下，或者修改脚本中约第 166 行的 `model_path`。

---

## 系统使用说明

根据您的使用需求，系统提供了不同的运行方式：

### 方式一：运行带有 UI 界面的前端客户端程序

该方式会启动一个拥有精美界面的 Tkinter 桌面应用程序。

```bash
python main.py
# 或 
python scr/main_app.py
```

*说明*：该桌面客户端默认设计为连接后端 REST API (`http://localhost:5000`) 来进行计算。如果后端服务器未运行，程序状态栏会显示 "Backend: Disconnected"（后端未连接），并自动切换为 **Demo Mode (演示模式)**，使用模拟数据展示 UI 交互逻辑。若需配置后端地址，可以在 `scr/main_app.py` 中修改 `CONFIG` 字典中的 `backend_url`。

### 方式二：运行本地整合检测系统 (不依赖后端，直接调用本地模型)

如果您想直接在本地使用电脑算力和模型进行真实的疲劳监测、身份验证与防遮挡报警，请运行以下脚本（运行前请确保已按前文说明**配置好 `.keras` 模型路径**）：

```bash
python scr/integrated_system.py
```

运行后将打开一个摄像头窗口：
1. **身份验证阶段**：系统首先会进行面部验证。
2. **实时监控阶段**：验证通过后，系统会在画面上叠加显示实时的 EAR、MAR、头部 Pitch 角度以及疲劳分数等信息，当检测到疲劳或视线偏离时，会触发疲劳警告。
3. 如果中途遮挡摄像头，系统会进入警告状态。
4. 按下 `q` 键可退出监控，按下 `r` 键可重置疲劳分数。

### 方式三：运行纯疲劳检测 Demo

如果您只需要测试眼睛、嘴巴与头部的疲劳特征提取，可以使用 Live Fatigue Demo：

```bash
python scr/live_fatigue_demo.py
```

## 项目结构简介

```
├── main.py                   # 前端客户端入口脚本
├── README.md                 # 项目说明文档
├── requirements.txt          # Python 依赖清单
├── scr/
│   ├── integrated_system.py  # 包含状态机、身份验证和疲劳检测的整合版
│   ├── live_fatigue_demo.py  # 实时疲劳监测 Demo
│   ├── main_app.py           # Tkinter UI 客户端实现
│   ├── train_eye_cnn.py      # 模型训练脚本
│   └── ...                   # 其他算法或测试脚本
└── utils/                    # 供 main_app 客户端使用的工具模块
    ├── api_client.py         # REST API 客户端
    ├── camera_handler.py     # 摄像头获取线程
    ├── alert_manager.py      # 警告/声音管理器
    └── ui_components.py      # 自定义 UI 组件
```
