# 驾驶员疲劳与防盗监测系统

系统采用 Tkinter 前端和本地检测后端。直接运行 `main.py` 时两者在同一程序内工作，不依赖 5000 端口；`server.py` 仍可用于需要 REST 分离部署的场景。

1. 驾驶员输入姓名并打开摄像头。
2. 点击 `Capture Face Samples`，连续采集 8 帧；至少 3 张检测到清晰人脸后才允许验证。
3. 点击 `Begin Verification`，后端将当前人脸与刚采集的授权样本比对。
4. 只有验证成功后，后端才签发临时会话并启用疲劳监测。
5. 监测 EAR（闭眼）、MAR（哈欠）和头部俯仰，连续异常时报警。
6. 画面过暗或细节消失时进入 `OCCLUDED` 防遮挡状态；遮挡清除后必须重新验证身份。

## 安装

建议使用 **64 位 Python 3.10 或 3.11**。TensorFlow、MediaPipe 和 DeepFace 对过新的 Python 版本可能没有可用安装包。

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

每位授权驾驶员至少放一张正面、光照清楚的照片到 `data/authorized_users/`。支持 `.jpg`、`.jpeg`、`.png` 和 `.bmp`。首次运行 DeepFace 时，模型初始化可能需要一些时间。

## 一键运行

直接执行（推荐）：

```powershell
python main.py
```

在包含中文字符的 Windows 路径中请优先双击 `run.bat`；它会临时映射一个英文盘符，以避免 MediaPipe 无法读取模型资源。

即使系统当前的 `python` 命令指向 Python 3.14，`main.py` 也会自动切换到已安装的 Python 3.11，并加载项目内的本地后端。

也可以在两个终端分别运行：

```powershell
python server.py
python main.py
```

## REST 接口

- `GET /api/health`：前后端连通性检查。
- `POST /api/enroll`：上传多张 `frames` 和 `driver_name`，建立授权人脸样本。
- `POST /api/verify`：上传 `frame` 和 `driver_name`，成功后返回 `session_id`。
- `POST /api/analyze`：上传 `frame` 和 `session_id`，返回疲劳指标及状态。
- `POST /api/session/end`：结束会话。

状态包括 `NORMAL`、`YAWNING`、`DROWSY`、`ALERT`、`NO_FACE`、`OCCLUDED` 和 `LOCKED`。

## 评估

将场景视频放入 `data/test_videos`，然后执行：

```powershell
python scr/evaluate_system.py
```

真实准确率评估需要为正常、未授权、遮挡和疲劳四类准备足够且互不重复的测试视频。
