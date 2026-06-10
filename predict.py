"""
predict.py — Inference Engine for Violence Detection

Supports:
  • Single video file prediction
  • Real-time webcam inference (sliding window)
  • Frame-by-frame analysis with confidence scores
"""

import cv2
import numpy as np
import argparse
import os
import time

from config import (
    FRAME_WIDTH, FRAME_HEIGHT, WINDOW_FRAMES, FRAME_SKIP,
    LABELS, LABEL_COLORS, CONFIDENCE_THRESHOLD
)
from feature_extractor import extract_features_from_frames, compute_flow_features
from train import load_model


# ─── Utilities ────────────────────────────────────────────────────────────────

def _draw_overlay(frame, label, confidence, alert=False):
    """Draw prediction overlay on a BGR frame."""
    color = LABEL_COLORS[label]
    text  = f"{LABELS[label]}  {confidence*100:.1f}%"

    # Background bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), color, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # Text
    cv2.putText(frame, text, (10, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Alert border
    if alert and label == 1:
        thickness = 6
        cv2.rectangle(frame,
                      (thickness//2, thickness//2),
                      (frame.shape[1]-thickness//2, frame.shape[0]-thickness//2),
                      (0, 0, 255), thickness)

    return frame


# ─── Single Video Prediction ──────────────────────────────────────────────────

def predict_video(video_path, model=None, scaler=None, show=True, save_path=None):
    """
    Predict violence label for a single video file.

    Args:
        video_path: Path to video file
        model, scaler: Pre-loaded model (or loaded automatically)
        show: Display annotated video with cv2.imshow
        save_path: If set, save annotated video here

    Returns:
        dict with label, confidence, frame_scores
    """
    if model is None or scaler is None:
        model, scaler = load_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n  Analyzing: {os.path.basename(video_path)}")
    print(f"  Frames: {total_frames} | FPS: {fps:.1f}")

    # ── Collect all frames ─────────────────────────────────────
    gray_frames, color_frames = [], []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % FRAME_SKIP == 0:
            color = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            gray  = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            color_frames.append(color.copy())
            gray_frames.append(gray)
        idx += 1
    cap.release()

    if len(gray_frames) < 2:
        return {"label": 0, "label_name": LABELS[0], "confidence": 0.0}

    # ── Sliding window prediction ──────────────────────────────
    window = WINDOW_FRAMES
    step   = max(1, window // 2)
    frame_scores = []

    for start in range(0, max(1, len(gray_frames) - window + 1), step):
        chunk = gray_frames[start:start + window]
        if len(chunk) < 2:
            continue
        feat = extract_features_from_frames(chunk)
        if feat is not None:
            feat_scaled = scaler.transform(feat.reshape(1, -1))
            prob = model.predict_proba(feat_scaled)[0]
            frame_scores.append({
                'start': start,
                'end': start + len(chunk),
                'prob_violence': prob[1],
                'prob_non_violence': prob[0],
            })

    if not frame_scores:
        return {"label": 0, "label_name": LABELS[0], "confidence": 0.5}

    # Overall decision: max violence probability across windows
    max_violence_prob = max(s['prob_violence'] for s in frame_scores)
    avg_violence_prob = np.mean([s['prob_violence'] for s in frame_scores])

    # Conservative: use average to reduce false positives
    final_prob = 0.6 * max_violence_prob + 0.4 * avg_violence_prob
    label      = 1 if final_prob >= 0.5 else 0
    confidence = final_prob if label == 1 else (1 - final_prob)

    print(f"  Result: {LABELS[label]}  (confidence: {confidence*100:.1f}%)")
    print(f"  Avg violence prob: {avg_violence_prob:.3f} | Max: {max_violence_prob:.3f}")

    # ── Annotate & display ────────────────────────────────────
    if show or save_path:
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps,
                                     (FRAME_WIDTH, FRAME_HEIGHT))

        for i, frame in enumerate(color_frames):
            # Find which window this frame belongs to
            win_label, win_conf = label, confidence
            for s in frame_scores:
                if s['start'] <= i <= s['end']:
                    wp = s['prob_violence']
                    win_label = 1 if wp >= 0.5 else 0
                    win_conf  = wp if win_label == 1 else (1 - wp)
                    break

            annotated = _draw_overlay(frame.copy(), win_label, win_conf,
                                      alert=(win_conf >= CONFIDENCE_THRESHOLD))

            if writer:
                writer.write(annotated)
            if show:
                cv2.imshow("Violence Detection", annotated)
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    break

        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    return {
        "label": label,
        "label_name": LABELS[label],
        "confidence": confidence,
        "avg_violence_prob": avg_violence_prob,
        "max_violence_prob": max_violence_prob,
        "frame_scores": frame_scores,
    }


# ─── Real-Time Webcam Inference ───────────────────────────────────────────────

def predict_webcam(model=None, scaler=None, camera_id=0):
    """
    Real-time violence detection via webcam using a sliding window buffer.
    Press 'q' to quit.
    """
    if model is None or scaler is None:
        model, scaler = load_model()

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    print("\n  🎥 Real-time inference started. Press 'q' to quit.")
    print(f"  Buffer: {WINDOW_FRAMES} frames | Skip: every {FRAME_SKIP} frame(s)")

    frame_buffer = []    # grayscale
    current_label = 0
    current_conf  = 0.5
    frame_idx = 0
    last_predict_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        if frame_idx % FRAME_SKIP == 0:
            gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
            frame_buffer.append(gray)

            # Keep buffer at window size
            if len(frame_buffer) > WINDOW_FRAMES:
                frame_buffer.pop(0)

            # Predict every 0.5 seconds
            now = time.time()
            if len(frame_buffer) >= WINDOW_FRAMES and (now - last_predict_time) > 0.5:
                feat = extract_features_from_frames(frame_buffer)
                if feat is not None:
                    feat_scaled = scaler.transform(feat.reshape(1, -1))
                    prob = model.predict_proba(feat_scaled)[0]
                    current_label = 1 if prob[1] >= 0.5 else 0
                    current_conf  = prob[1] if current_label == 1 else prob[0]
                last_predict_time = now

        annotated = _draw_overlay(display.copy(), current_label, current_conf,
                                  alert=(current_conf >= CONFIDENCE_THRESHOLD))

        # Status
        buf_pct = min(100, int(len(frame_buffer) / WINDOW_FRAMES * 100))
        cv2.putText(annotated, f"Buffer: {buf_pct}%", (10, FRAME_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Violence Detection — Live", annotated)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("  Camera closed.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Violence & Non-Violence Action Recognition — Inference"
    )
    parser.add_argument("--video",  type=str, help="Path to video file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--save",   type=str, help="Save annotated output to this path")
    parser.add_argument("--no-show", action="store_true", help="Don't display video")
    args = parser.parse_args()

    model, scaler = load_model()

    if args.webcam:
        predict_webcam(model, scaler, camera_id=args.camera)
    elif args.video:
        result = predict_video(args.video, model, scaler,
                               show=not args.no_show,
                               save_path=args.save)
        print("\n  Final Result:")
        print(f"    Label:      {result['label_name']}")
        print(f"    Confidence: {result['confidence']*100:.1f}%")
    else:
        parser.print_help()
