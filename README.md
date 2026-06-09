# Driver Fatigue Detection System - Backend Demo

> ⚠️ **NOTICE**: This branch (`backend-dev`) contains the backend demonstration for the Driver Fatigue Detection System. It is an experimental/demo branch and is NOT intended for production use. It showcases the architecture, LSTM model training pipeline, and MediaPipe-based feature extraction.

## Features (Demo)

- **Feature Extraction**: Uses MediaPipe to extract Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and head pose metrics from video frames.
- **LSTM Model Training**: Experimental pipeline to train an LSTM-based fatigue detection model using the extracted features.
- **REST API**: Provides a basic demonstration of API endpoints that a client application might interact with for driver fatigue and anomaly detection.
- **Interpretability**: Includes SHAP-based model interpretability examples.

---

## Architecture (Backend Demo)

The project on this branch is structured to demonstrate the backend logic:

```
├── fatigue_detection/        # Backend specific modules (Feature extraction, LSTM training)
├── main_app.py               # Main Tkinter desktop client (provided for testing the backend)
├── requirements.txt          # Python dependencies
├── utils/                    # Utility scripts (API client, camera handling, etc.)
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.8 or higher.

### 2. Install Dependencies
Initialize a Python virtual environment and install the required modules:

```bash
# Create and activate virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Running the Demo Backend
*Instructions for running the specific backend scripts or the test client can be found in the respective directories. As this is a demo, please refer to the `fatigue_detection` module for backend logic.*
