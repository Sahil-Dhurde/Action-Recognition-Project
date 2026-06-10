"""
config.py — Central configuration for the Violence Detection System
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
VIOLENCE_DIR    = os.path.join(DATA_DIR, "violence")
NON_VIOLENCE_DIR= os.path.join(DATA_DIR, "non_violence")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
MODEL_PATH      = os.path.join(MODEL_DIR, "violence_classifier.pkl")
SCALER_PATH     = os.path.join(MODEL_DIR, "feature_scaler.pkl")

# Create dirs if needed
for d in [DATA_DIR, VIOLENCE_DIR, NON_VIOLENCE_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Video Processing ─────────────────────────────────────────────────────────
FRAME_WIDTH     = 224          # Resize width
FRAME_HEIGHT    = 224          # Resize height
FRAME_SKIP      = 2            # Process every N-th frame (speed vs accuracy)
CLIP_FRAMES     = 30           # Frames sampled per clip for feature extraction

# ─── Optical Flow ─────────────────────────────────────────────────────────────
FLOW_LEVELS     = 3            # Pyramid levels for Farneback
FLOW_WIN_SIZE   = 15           # Window size
FLOW_ITERATIONS = 3            # Iterations per level
FLOW_POLY_N     = 5            # Polynomial expansion neighborhood
FLOW_POLY_SIGMA = 1.2          # Gaussian std for polynomial expansion

# ─── HOG Features ─────────────────────────────────────────────────────────────
HOG_ORIENTATIONS    = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# ─── Classification ───────────────────────────────────────────────────────────
LABELS          = {0: "Non-Violence ✅", 1: "Violence 🚨"}
LABEL_COLORS    = {0: (0, 200, 0), 1: (0, 0, 220)}   # BGR for OpenCV
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

# ─── Inference ────────────────────────────────────────────────────────────────
WINDOW_FRAMES   = 30           # Sliding window size for real-time prediction
CONFIDENCE_THRESHOLD = 0.60    # Min confidence to show alert
