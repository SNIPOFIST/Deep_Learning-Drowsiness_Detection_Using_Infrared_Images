"""Paths and hyperparameters shared across notebooks and scripts."""

from pathlib import Path

# Project root = parent of src/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT = _PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_VIDEOS = OUTPUT_DIR / "videos"
OUTPUT_CROPS = OUTPUT_DIR / "crops"

# dlib 68-point landmark file (download separately; see README)
SHAPE_PREDICTOR_PATH = ASSETS_DIR / "shape_predictor_68_face_landmarks.dat"

# Default trained weights used by inference scripts
DEFAULT_MODEL_PATH = MODELS_DIR / "cnn_mrl_driver_aug.keras"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_cnn_mrl.keras"

IMG_SIZE = 64
SLEEPY_THRESHOLD = 0.5
SLEEPY_TIME_SEC = 5.0
FRAME_SKIP = 3

# Sample video for offline demo (place file here after download)
SAMPLE_VIDEO_PATH = DATA_DIR / "test_input" / "test_video.mp4"
