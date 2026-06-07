# Driver Fatigue Detection System - Client (Desktop App)

Real-time driver fatigue monitoring system client built using **Python, OpenCV, and Tkinter**. This desktop client connects to a REST API backend to run face verification and real-time fatigue analysis. 

## Features

- 👤 **Identity Verification**: Captures the driver's face, checks lighting/positioning, and posts to the backend verification server.
- 👁 **Eye Aspect Ratio (EAR) Analysis**: Real-time monitoring of eye closure to detect drowsiness.
- 👄 **Mouth Aspect Ratio (MAR) Analysis**: Detects yawning to warn against incoming fatigue.
- 🧠 **Head Pose Anomaly Detection**: Warns if the driver turns away or drops their head.
- 🚨 **防盗防篡改警报 (Anti-Theft System)**: Triggers immediate local and audio alarms if the camera is obstructed or an unauthorized face is detected.
- 🔊 **Platform-Agnostic Audio Alarms**: Low-latency buzzer beeps with individual cooldown periods to avoid sound fatigue.
- 📋 **Live Event Logger**: Scrolling timeline displaying system status changes and warnings.
- 🌐 **Demo Mode Fallback**: Automatically switches to generating mock metrics if the backend server is offline, keeping the UI fully interactive.

---

## Architecture

The project is structured as a frontend shell:

```
├── main_app.py               # Main Tkinter desktop client
├── requirements.txt          # Python dependencies
├── utils/
│   ├── api_client.py         # REST Client (handles GET/POST calls to server)
│   ├── camera_handler.py     # Thread-safe OpenCV camera thread
│   ├── alert_manager.py      # Cross-platform sound manager & cooldown handler
│   └── ui_components.py      # Custom reusable Tkinter widgets
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.8 or higher.
- A functional USB webcam or laptop integrated camera.

### 2. Install Dependencies
Initialize a Python virtual environment and install the required modules:

```bash
# Create and activate virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Running the App
To start the desktop GUI:

```bash
python main_app.py
```

*Note: If your backend server is not running on `http://localhost:5000`, the app will show "Backend: Disconnected" in the status bar and run in **Demo Mode** with mock data. You can configure the backend URL in `main_app.py` under `CONFIG`.*

---

## API Contract (Backend Integration)

The backend server is expected to expose the following REST endpoints:

### 1. Verification
- **Endpoint**: `POST /api/verify`
- **Request (JSON)**:
  ```json
  {
    "driver_name": "Driver Name",
    "frame": "<base64-encoded JPEG image>"
  }
  ```
- **Response (JSON)**:
  ```json
  {
    "status": "authorized" | "unauthorized" | "obstructed" | "error",
    "name": "Driver Name",
    "confidence": 0.95
  }
  ```

### 2. Analysis
- **Endpoint**: `POST /api/analyze`
- **Request (JSON)**:
  ```json
  {
    "frame": "<base64-encoded JPEG image>"
  }
  ```
- **Response (JSON)**:
  ```json
  {
    "status": "NORMAL" | "DROWSY" | "YAWNING" | "ALERT",
    "ear": 0.312,
    "mar": 0.421,
    "head_pose": "normal" | "anomaly"
  }
  ```

### 3. Health Check
- **Endpoint**: `GET /api/health`
- **Response (JSON)**:
  ```json
  {
    "ok": true
  }
  ```

---

## Technologies Used

- **UI Framework**: Python Tkinter
- **Image Processing**: OpenCV (python-opencv), Pillow
- **HTTP Client**: Requests
- **Data Processing**: NumPy
