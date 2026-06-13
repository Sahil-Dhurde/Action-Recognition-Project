""" 
feature_extractor.py — Optical Flow + HOG Feature Extraction Pipeline

Features extracted per video clip:
  • Optical Flow statistics (magnitude, direction, chaos)
  • HOG (Histogram of Oriented Gradients) descriptors
  • Motion energy metrics
  • Temporal aggregation (mean, std, max across frames)
"""

import cv2
import numpy as np
from config import (
    FRAME_WIDTH, FRAME_HEIGHT, CLIP_FRAMES, FRAME_SKIP,
    FLOW_LEVELS, FLOW_WIN_SIZE, FLOW_ITERATIONS, FLOW_POLY_N, FLOW_POLY_SIGMA,
    HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK
)


# ─── HOG Descriptor ───────────────────────────────────────────────────────────

def compute_hog(frame_gray):
    """Compute HOG descriptor for a grayscale frame."""
    hog = cv2.HOGDescriptor(
        _winSize=(FRAME_WIDTH, FRAME_HEIGHT),
        _blockSize=(HOG_PIXELS_PER_CELL[0] * HOG_CELLS_PER_BLOCK[0],
                    HOG_PIXELS_PER_CELL[1] * HOG_CELLS_PER_BLOCK[1]),
        _blockStride=HOG_PIXELS_PER_CELL,
        _cellSize=HOG_PIXELS_PER_CELL,
        _nbins=HOG_ORIENTATIONS
    )
    return hog.compute(frame_gray).flatten()


# ─── Optical Flow Features ────────────────────────────────────────────────────

def compute_flow_features(prev_gray, curr_gray):
    """
    Compute dense optical flow (Farneback) between two grayscale frames.
    Returns a feature vector capturing motion statistics.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,
        levels=FLOW_LEVELS,
        winsize=FLOW_WIN_SIZE,
        iterations=FLOW_ITERATIONS,
        poly_n=FLOW_POLY_N,
        poly_sigma=FLOW_POLY_SIGMA,
        flags=0
    )

    # Convert to polar (magnitude + angle)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # ── Magnitude stats ──────────────────────────────────────────
    mag_mean   = np.mean(magnitude)
    mag_std    = np.std(magnitude)
    mag_max    = np.max(magnitude)
    mag_median = np.median(magnitude)

    # ── Angle histogram (8 bins = 45° each) ─────────────────────
    angle_hist, _ = np.histogram(angle, bins=8, range=(0, 2 * np.pi))
    angle_hist = angle_hist.astype(float) / (angle_hist.sum() + 1e-6)  # normalize

    # ── Chaos score: variance of flow vectors (fights = chaotic) ─
    flow_var_x = np.var(flow[..., 0])
    flow_var_y = np.var(flow[..., 1])
    chaos_score = (flow_var_x + flow_var_y) / 2.0

    # ── Motion energy (proportion of pixels with significant motion)
    motion_threshold = 1.5
    motion_energy = np.mean(magnitude > motion_threshold)

    # ── Spatial flow statistics (divide frame into 4 quadrants) ──
    h, w = magnitude.shape
    q_means = [
        np.mean(magnitude[:h//2, :w//2]),
        np.mean(magnitude[:h//2, w//2:]),
        np.mean(magnitude[h//2:, :w//2]),
        np.mean(magnitude[h//2:, w//2:]),
    ]

    feature = np.array([
        mag_mean, mag_std, mag_max, mag_median,
        chaos_score, motion_energy,
        *angle_hist,   # 8 values
        *q_means,      # 4 values
    ], dtype=np.float32)

    return feature   # shape: (18,)


# ─── Frame Loader ─────────────────────────────────────────────────────────────

def load_frames(video_path, max_frames=CLIP_FRAMES, frame_skip=FRAME_SKIP):
    """
    Load frames from a video file.
    Returns list of resized grayscale frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_skip == 0:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        idx += 1

    cap.release()
    return frames


def load_frames_color(video_path, max_frames=CLIP_FRAMES, frame_skip=FRAME_SKIP):
    """Load color frames (for visualization)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_skip == 0:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frames.append(frame)
        idx += 1

    cap.release()
    return frames


# ─── Full Feature Extraction ──────────────────────────────────────────────────

def extract_features_from_frames(frames):
    """
    Given a list of grayscale frames, extract the full feature vector.

    Strategy:
      1. Compute optical flow between consecutive pairs → flow features per step
      2. Compute HOG on select frames
      3. Aggregate with mean, std, max across time
    """
    if len(frames) < 2:
        return None

    # ── Optical flow across all consecutive pairs ──────────────
    flow_features = []
    for i in range(1, len(frames)):
        ff = compute_flow_features(frames[i-1], frames[i])
        flow_features.append(ff)

    flow_features = np.array(flow_features)  # (T-1, 18)

    # Temporal aggregation: mean, std, max
    flow_mean = np.mean(flow_features, axis=0)
    flow_std  = np.std(flow_features,  axis=0)
    flow_max  = np.max(flow_features,  axis=0)

    # ── HOG on uniformly sampled frames ───────────────────────
    sample_indices = np.linspace(0, len(frames)-1, min(5, len(frames)), dtype=int)
    hog_features = []
    for i in sample_indices:
        try:
            h = compute_hog(frames[i])
            hog_features.append(h)
        except Exception:
            pass

    if hog_features:
        hog_mean = np.mean(hog_features, axis=0)
        # Reduce HOG dimensionality via uniform sampling to keep feature vector manageable
        step = max(1, len(hog_mean) // 64)
        hog_reduced = hog_mean[::step][:64]
        # Pad if needed
        if len(hog_reduced) < 64:
            hog_reduced = np.pad(hog_reduced, (0, 64 - len(hog_reduced)))
    else:
        hog_reduced = np.zeros(64)

    # ── Combine ────────────────────────────────────────────────
    feature_vector = np.concatenate([
        flow_mean,   # 18
        flow_std,    # 18
        flow_max,    # 18
        hog_reduced, # 64
    ])  # Total: 118 features

    return feature_vector.astype(np.float32)


def extract_features_from_video(video_path):
    """Full pipeline: video → feature vector."""
    frames = load_frames(video_path)
    if len(frames) < 2:
        print(f"  [WARNING] Not enough frames in {video_path}")
        return None
    return extract_features_from_frames(frames)
