# Driver Fatigue Detection System

Real-time driver fatigue monitoring using Python, OpenCV, MediaPipe, and Keras/TensorFlow. Detects drowsiness via eye closure, yawning, and head pose analysis — with occlusion detection for anti-tampering.

---

## Features

| Signal | Detection Method | Indicator |
|--------|-----------------|-----------|
| 👁 **Eye Closure** | CNN classifier (64×64 grayscale) on cropped eye regions | Eye Aspect Ratio (EAR) estimation |
| 👄 **Yawning** | Mouth Aspect Ratio (MAR) from MediaPipe lip landmarks | MAR > threshold |
| 🧠 **Head Pose** | solvePnP with 3D head model → pitch angle | Pitch exceeding calibrated threshold |
| 🚨 **Occlusion** | Brightness + Laplacian variance check | Camera blocked / tampered |

Fatigue assessment uses a **weighted fusion scorer** (eye 0.4, yawn 0.3, nod 0.3) with a **progressive frame counter** — brief recoveries don't mask sustained fatigue patterns.

---

## Project Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      main.py (entry)                     │
│                    scr/main_app.py                       │
│              Tkinter GUI — Login → Monitor              │
│                          │                               │
│            utils/api_client.py  (lazy init)              │
│            utils/camera_handler.py                       │
│            utils/alert_manager.py                        │
│            utils/ui_components.py                         │
└──────────────┬───────────────────────────────────────────┘
               │  direct function calls (same process)
┌──────────────▼───────────────────────────────────────────┐
│                scr/  (backend modules)                    │
│  ┌──────────────────┐  ┌───────────────┐                 │
│  │ integrated_system│  │fatigue_scorer │                 │
│  │  (state machine) │  │ (fusion logic)│                 │
│  └────────┬─────────┘  └───────┬───────┘                 │
│           │                     │                         │
│  ┌────────▼─────────────────────▼───────┐                │
│  │ compute_mar   nod_detector           │                │
│  │ occlusion_detector                   │                │
│  └──────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────┘
```

The GUI and backend run **in-process** — no HTTP server needed. `api_client.py` directly imports and calls backend functions. MediaPipe, the fatigue scorer, and the eye CNN are initialised **lazily** on first use so that missing optional dependencies (e.g. Keras) don't prevent the module from loading.

---

## Environment Configuration

The system consists of two layers that share the same Python environment:

| Layer | Role | Key Dependencies |
|-------|------|-----------------|
| **Frontend** | Tkinter GUI, camera preview, user interaction | `opencv-python`, `Pillow`, `requests` |
| **Backend** | MediaPipe face mesh, eye CNN inference, fatigue scoring | `mediapipe`, `tensorflow`, `keras`, `numpy` |

Both layers run in a single Python process — one virtual environment serves both.

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Dual-core x86_64 | Quad-core or better |
| **RAM** | 4 GB | 8 GB+ |
| **Camera** | Built-in or USB webcam (≥ 480p) | 720p+ with stable frame rate |
| **GPU** | Not required | NVIDIA GPU with CUDA (speeds up CNN inference) |
| **OS** | Windows 10+ / macOS 11+ / Ubuntu 20.04+ | — |

### Python Version

- **Python ≥ 3.9** (tested on 3.9–3.11)
- TensorFlow 2.x requires Python ≤ 3.11 on Windows; if using Python 3.12+, switch to `tensorflow-cpu` or install from `conda-forge`.

### Frontend Environment Setup

The frontend is a **Tkinter desktop application**. Tkinter ships with Python on Windows and macOS; on Linux you may need to install it separately:

```bash
# Ubuntu / Debian
sudo apt install python3-tk

# Fedora
sudo dnf install python3-tkinter

# Arch
sudo pacman -S tk
```

#### Frontend-only (lightweight mode)

If you only need the GUI shell (no real-time fatigue detection), install just the frontend dependencies:

```bash
pip install opencv-python Pillow requests numpy
```

The GUI will start in **Demo Mode** when backend dependencies are missing — simulated data is displayed, and the status bar shows "Backend: Disconnected".

### Backend Environment Setup

The backend requires MediaPipe and TensorFlow/Keras — these are the heaviest dependencies.

#### Option A: CPU-only (simplest)

```bash
pip install mediapipe tensorflow keras numpy opencv-python
```

TensorFlow will run on CPU. Inference is slower (~3–5 FPS with the eye CNN) but requires no extra setup.

#### Option B: GPU-accelerated (NVIDIA CUDA)

For real-time performance (15+ FPS with CNN inference):

1. Install **CUDA Toolkit 11.8** and **cuDNN 8.6** from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads).
2. Then install GPU-enabled TensorFlow:

```bash
pip install mediapipe tensorflow[and-cuda] keras numpy opencv-python
```

> **macOS note:** TensorFlow GPU support on macOS is limited to `tensorflow-metal` on Apple Silicon (M1/M2/M3). Install with:
> ```bash
> pip install tensorflow-metal
> ```

#### Verify Backend Installation

Run the MediaPipe smoke test to confirm the backend works:

```bash
python scr/test_mediapipe.py
```

A successful run opens a camera window with face mesh landmarks drawn on your face. Press `Q` to quit.

### One-Command Full Setup

```bash
# 1. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS / Linux

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Verify
python scr/test_mediapipe.py
```

---

## Backend Model Configuration

The backend uses a pre-trained eye CNN classifier (`eye_classifier.keras`) to detect whether the driver's eyes are open or closed. This section covers how to obtain, place, and configure the model.

### Step 1 — Download the Pre-trained Model

A ready-to-use model is available on Google Drive:

> 📥 **[Download eye_classifier.keras](https://drive.google.com/file/d/1xAeaX8UIEQ4cxcCDfI2mhe12cftUGDiR/view?usp=drive_link)**

### Step 2 — Place the Model File

Copy the downloaded `eye_classifier.keras` to the **project root** directory:

```
D:\1\Intro\Intro\eye_classifier.keras
```

### Step 3 — Verify the Model Is Found

The system looks for the model at `PROJECT_ROOT / "eye_classifier.keras"`. Run a quick check:

```bash
python -c "from pathlib import Path; m = Path('eye_classifier.keras'); print('OK' if m.exists() else 'MISSING — place it in the project root')"
```

### How Each Script Loads the Model

| Script | Model Path | How to Change |
|--------|-----------|---------------|
| `utils/api_client.py` | `PROJECT_ROOT / "eye_classifier.keras"` | Edit the `model_path` variable (~line 86) |
| `scr/integrated_system.py` | `PROJECT_ROOT / "eye_classifier.keras"` | Edit the `model_path` variable (~line 119) |
| `scr/live_fatigue_demo.py` | `PROJECT_ROOT / "outputs" / "eye_classifier.keras"` | Edit the path (~line 166) — or place a copy in `outputs/` |
| `scr/evaluate_system.py` | Legacy hardcoded path | Update before use |
| `scr/evaluate_eye_cnn.py` | Legacy hardcoded path | Update before use |

> If you want to store the model elsewhere, change the `model_path` variable in the scripts you plan to run. The GUI (via `api_client.py`) and the headless integrated system (`integrated_system.py`) are the two primary entry points.

### Backend Detection Thresholds

These thresholds control when each fatigue signal triggers. They live in `scr/integrated_system.py`:

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| `MAR_THRESHOLD` | `0.240` | Mouth Aspect Ratio above this → yawning detected |
| `PITCH_THRESHOLD` | `13.03` | Head pitch angle (degrees) above this → nodding detected |

And in `scr/fatigue_scorer.py`:

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| `EYE_WEIGHT` | `0.4` | Eye-closure contribution to fatigue score |
| `YAWN_WEIGHT` | `0.3` | Yawning contribution to fatigue score |
| `NOD_WEIGHT` | `0.3` | Head-nod contribution to fatigue score |
| `ALERT_THRESHOLD` | `0.5` | Score above this value → "bad frame" |
| `ALERT_FRAME_LIMIT` | `60` | Consecutive bad frames before alert fires |

To tune these for your environment, edit the constants at the top of the respective `.py` files, then re-run.

### Fallback Behaviour

If the `.keras` model is **not found** at the expected path, the system runs with **reduced accuracy**: eye-closure detection via CNN is disabled, but MAR (yawning) and head-pose (nodding) signals still work. The GUI status bar will show "Backend: Disconnected (Demo Mode)" or indicate the backend is partially available.

### Train Your Own Model

To train the CNN from scratch with your own dataset, see [Training Scripts](#training-scripts) below.

---

## Frontend Startup Guide

### Option 1 — GUI Application (Recommended)

This is the main way to use the system — a Tkinter desktop window with camera preview, real-time metrics, and alert controls.

#### Launch

```bash
# Activate the virtual environment first
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS / Linux

# Then start the GUI
python main.py
```

#### Windows One-Click Launcher

A convenience batch file `run.bat` is included in the project root. It activates the conda environment and launches the GUI:

```
双击 run.bat 即可启动
```

> If your virtual environment path differs, edit `run.bat` and change the `call` line to your own environment's `activate` path.

#### What You'll See

After running `python main.py`, the following pages appear in sequence:

| Step | Page | What Happens |
|------|------|-------------|
| 1 | **Login Page** | Enter a driver name and click "Start Monitoring" |
| 2 | **Monitoring Dashboard** | Live camera feed appears with real-time overlays |

The **Monitoring Dashboard** shows:

- 🎥 **Live camera feed** — what the webcam sees
- 📊 **Metric cards** — EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio), alert count
- 📋 **Event log** — timestamped fatigue events (DROWSY, YAWNING, HEAD_POSE)
- 🎛️ **Control buttons** — Start/Pause, Snapshot, End Session
- 🟢🟡🔴 **StatusBadge** — backend connection status

#### Controls

| Button | Action |
|--------|--------|
| **Start / Pause** | Toggle real-time monitoring on/off |
| **Snapshot** | Save the current camera frame to disk |
| **End Session** | Close the session and return to login |

#### Demo Mode vs Live Mode

| Mode | Status Bar Text | What's Happening |
|------|----------------|-----------------|
| **Live Mode** | `Backend: Connected` | MediaPipe + CNN running; real fatigue detection active |
| **Demo Mode** | `Backend: Disconnected (Demo Mode)` | Backend unavailable; simulated EAR/MAR values shown |

If you see Demo Mode, check that:
1. MediaPipe and Keras are installed (`pip list | findstr mediapipe keras`)
2. `eye_classifier.keras` exists in the project root
3. Your webcam is connected and not in use by another app

#### Troubleshooting the Frontend

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ImportError: No module named 'tkinter'` | Tkinter not installed (Linux) | `sudo apt install python3-tk` |
| GUI opens but camera is black | Webcam in use or wrong index | Change `camera_index` in `CONFIG` dict (scr/main_app.py ~line 36) |
| `ModuleNotFoundError: mediapipe` | Dependencies missing | `pip install -r requirements.txt` |
| Status stuck on "Disconnected" | Keras/TF import failed | Reinstall: `pip install tensorflow keras` |
| Window too large for screen | Default is 1280×800 | Resize manually, or edit `root.geometry("1280x800")` in main_app.py |

### Option 2 — Integrated System (headless, real-time)

```bash
python scr/integrated_system.py
```

Opens an OpenCV window with a HUD overlay showing:

- Eye closed / Yawning / Nodding status
- Head pitch angle
- Real-time fatigue score + bad-frame counter
- Alert banner when fatigue threshold is exceeded

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Reset fatigue alert |

The state machine monitors for camera occlusion and transitions between `MONITOR` and `OCCLUDED` states.

### Option 3 — Live Fatigue Demo

```bash
python scr/live_fatigue_demo.py
```

Similar to Option 2 but with a richer HUD (includes WARNING level display). Same controls: `Q` to quit, `R` to reset.

### Option 4 — System Evaluation

```bash
python scr/evaluate_system.py
```

Runs the state machine against test videos and produces:
- Confusion matrix (`confusion_matrix.png`)
- Classification report (Normal / Occluded / Fatigue)

Test videos go in `data/test_videos/`. Record them with:

```bash
python scr/record_test_data.py
```

---

## Project Structure

```
├── main.py                     # GUI entry point
├── run.bat                     # Windows launcher (activate venv → run)
├── requirements.txt            # Python dependencies
├── README.md
├── eye_classifier.keras        # Pre-trained eye CNN (not tracked)
│
├── scr/
│   ├── main_app.py             # Tkinter GUI application (~770 lines)
│   ├── integrated_system.py    # State-machine pipeline (monitor ↔ occluded)
│   ├── live_fatigue_demo.py    # Real-time fatigue HUD demo
│   ├── fatigue_scorer.py       # Weighted fusion scorer with frame counter
│   ├── compute_mar.py          # Mouth Aspect Ratio from MediaPipe landmarks
│   ├── nod_detector.py         # Head pitch via solvePnP → nodding detection
│   ├── occlusion_detector.py   # Brightness + variance check for tampering
│   │
│   ├── train_eye_cnn.py        # Train eye open/closed CNN classifier
│   ├── evaluate_eye_cnn.py     # Evaluate trained CNN on test set
│   ├── evaluate_system.py      # End-to-end system evaluation
│   ├── record_test_data.py     # Record test videos for evaluation
│   ├── test_fusion.py          # Unit tests for FatigueScorer logic
│   │
│   ├── analyza_mar.py          # MAR threshold calibration (batch)
│   ├── analyza_pitch.py        # Pitch threshold calibration (batch)
│   ├── overview_dataset.py     # Print dataset folder structure
│   ├── show_samples.py         # Display sample images from dataset
│   └── test_mediapipe.py       # MediaPipe FaceMesh smoke test
│
├── utils/
│   ├── api_client.py           # Backend bridge — lazy-init MediaPipe + CNN
│   ├── camera_handler.py       # Thread-safe OpenCV capture wrapper
│   ├── alert_manager.py        # Audible alerts with cooldown (cross-platform)
│   └── ui_components.py        # Reusable Tkinter widgets (StatusBadge, etc.)
│
├── data/
│   ├── authorized_users/       # Reference face images (face verification, unused)
│   └── test_videos/            # Test clips for evaluate_system.py
│
└── outputs/                    # Generated plots and reports
```

---

## Configuration

Key constants are scattered across modules. Here are the main tunables:

### Thresholds (in `scr/integrated_system.py` and `scr/nod_detector.py`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `MAR_THRESHOLD` | 0.240 | MAR above this → yawning |
| `PITCH_THRESHOLD` | 13.03° | Pitch above this → nodding |
| `EAR_THRESHOLD` | 0.25 | EAR below this → eye closure (GUI only) |

### Fatigue Scorer (in `scr/fatigue_scorer.py`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `EYE_WEIGHT` | 0.4 | Eye-closure contribution |
| `YAWN_WEIGHT` | 0.3 | Yawning contribution |
| `NOD_WEIGHT` | 0.3 | Head-pose contribution |
| `ALERT_THRESHOLD` | 0.5 | Score above this → "bad frame" |
| `ALERT_FRAME_LIMIT` | 60 | Consecutive bad frames to fire alert |

### GUI Config (in `scr/main_app.py` — `CONFIG` dict)

```python
CONFIG = {
    "backend_url": "http://localhost:5000",  # unused — local backend
    "camera_index": 0,
    "frame_rate": 15,
    "alert_cooldown": 3,         # seconds
    "ear_threshold": 0.25,
    "mar_threshold": 0.6,
    "consec_frames": 20,
}
```

### Alert Cooldowns (in `utils/alert_manager.py`)

| Alert Type | Cooldown |
|-----------|----------|
| DROWSY | 2.5 s |
| YAWNING | 4.0 s |
| HEAD | 3.0 s |
| UNAUTHORIZED | 5.0 s |
| OBSTRUCTION | 5.0 s |

---

## Training Scripts

### Train the Eye CNN

```bash
python scr/train_eye_cnn.py
```

Expects data at `D:/Fatigue_project/data/dataset_new/train/` with subfolders:
- `Closed/` — images of closed eyes
- `Open/`   — images of open eyes

Outputs `eye_classifier.keras` and training curves.

### Calibrate MAR Threshold

```bash
python scr/analyza_mar.py
```

Batch-processes yawn / no-yawn images to find the optimal MAR threshold.

### Calibrate Pitch Threshold

```bash
python scr/analyza_pitch.py
```

Batch-processes nod / no-nod images to find the optimal pitch threshold.

---

## Known Limitations

- **EAR** in the GUI is currently a mock value (0.35 default, 0.15 when eye-closed detected). A proper EAR computation from eye landmarks is not yet wired in.
- **Hardcoded dataset paths** exist in training/analysis scripts (e.g. `D:/Fatigue_project/…`). These need updating before use on other machines.
- **Face verification** has been removed from the current pipeline. The `scr/face_verification.py` module and related UI (verify page) were deleted. If identity verification is needed, re-implement or restore from git history.
- **Occlusion detection** uses simple brightness + variance heuristics. It may false-positive in very dark environments or false-negative against sophisticated tampering.

---

## License

Internal project — no license specified.
