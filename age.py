"""
Real-Time Age & Gender Detection — Webcam
==========================================
Displays gender (Male / Female) and age range on every detected face
directly in the live webcam window.

Models are downloaded automatically on first run (~50 MB).

Install:
    pip install opencv-python numpy

Run:
    python age_gender_detection.py          # default webcam (index 0)
    python age_gender_detection.py --cam 1  # second camera

Controls:
    Q / ESC  — quit
    S        — save screenshot
    P        — pause / resume
"""

import argparse
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
CAM_W, CAM_H = 1280, 720
FACE_CONF    = 0.88        # face detection confidence threshold
FACE_PADDING = 22          # extra pixels around face crop for context
MODEL_MEAN   = (78.4263377603, 87.7689143744, 114.895847746)

AGE_BUCKETS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+"]
GENDERS     = ["Male", "Female"]

GENDER_COLOR = {
    "Male":   (220, 160,  50),   # BGR: amber
    "Female": (200,  80, 220),   # BGR: purple-pink
}

# ──────────────────────────────────────────────────────────────────────────────
# MODEL PATHS & DOWNLOAD URLS
# ──────────────────────────────────────────────────────────────────────────────
_BASE = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/"
_RAW  = "https://github.com/spmallick/learnopencv/raw/master/AgeGender/"

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELS = {
    "face_proto":   (_BASE + "opencv_face_detector.pbtxt",    "opencv_face_detector.pbtxt"),
    "face_model":   (_BASE + "opencv_face_detector_uint8.pb", "opencv_face_detector_uint8.pb"),
    "age_proto":    (_BASE + "age_deploy.prototxt",           "age_deploy.prototxt"),
    "age_model":    (_RAW  + "age_net.caffemodel",            "age_net.caffemodel"),
    "gender_proto": (_BASE + "gender_deploy.prototxt",        "gender_deploy.prototxt"),
    "gender_model": (_RAW  + "gender_net.caffemodel",         "gender_net.caffemodel"),
}


# ──────────────────────────────────────────────────────────────────────────────
# MODEL DOWNLOAD
# ──────────────────────────────────────────────────────────────────────────────
def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct  = min(int(downloaded * 100 / total_size), 100)
    done = pct // 4
    bar  = "█" * done + "░" * (25 - done)
    print(f"\r    [{bar}] {pct:3d}%", end="", flush=True)


def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    any_new = False
    for key, (url, filename) in MODELS.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            continue
        any_new = True
        print(f"  Downloading {filename} ...")
        try:
            urllib.request.urlretrieve(url, path, _progress)
            print()
        except Exception as exc:
            print(f"\n  [ERROR] Failed: {exc}")
            print(f"  Download manually from:\n    {url}\n  Save to: {path}")
            sys.exit(1)
    if any_new:
        print("  All models ready.\n")


# ──────────────────────────────────────────────────────────────────────────────
# LOAD NETWORKS
# ──────────────────────────────────────────────────────────────────────────────
def load_nets():
    def p(key):
        return os.path.join(MODEL_DIR, MODELS[key][1])

    face_net   = cv2.dnn.readNet(p("face_model"),   p("face_proto"))
    age_net    = cv2.dnn.readNet(p("age_model"),    p("age_proto"))
    gender_net = cv2.dnn.readNet(p("gender_model"), p("gender_proto"))
    return face_net, age_net, gender_net


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────────────────────
def detect_faces(frame, net):
    """Returns list of (x1, y1, x2, y2, confidence)."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                  [104, 117, 123], swapRB=False)
    net.setInput(blob)
    out   = net.forward()
    faces = []
    for i in range(out.shape[2]):
        conf = float(out[0, 0, i, 2])
        if conf < FACE_CONF:
            continue
        x1 = max(0,     int(out[0, 0, i, 3] * w))
        y1 = max(0,     int(out[0, 0, i, 4] * h))
        x2 = min(w - 1, int(out[0, 0, i, 5] * w))
        y2 = min(h - 1, int(out[0, 0, i, 6] * h))
        if (x2 - x1) > 25 and (y2 - y1) > 25:
            faces.append((x1, y1, x2, y2, conf))
    return faces


def predict_age_gender(frame, x1, y1, x2, y2, age_net, gender_net):
    """Returns (gender_str, age_str, gender_conf, age_conf)."""
    H, W = frame.shape[:2]
    fx1  = max(0,     x1 - FACE_PADDING)
    fy1  = max(0,     y1 - FACE_PADDING)
    fx2  = min(W - 1, x2 + FACE_PADDING)
    fy2  = min(H - 1, y2 + FACE_PADDING)
    crop = frame[fy1:fy2, fx1:fx2]
    if crop.size == 0:
        return "?", "?", 0.0, 0.0

    blob = cv2.dnn.blobFromImage(crop, 1.0, (227, 227),
                                  MODEL_MEAN, swapRB=False)

    gender_net.setInput(blob)
    gp  = gender_net.forward()[0]
    gid = int(np.argmax(gp))

    age_net.setInput(blob)
    ap  = age_net.forward()[0]
    aid = int(np.argmax(ap))

    return GENDERS[gid], AGE_BUCKETS[aid], float(gp[gid]), float(ap[aid])


# ──────────────────────────────────────────────────────────────────────────────
# DRAWING — face label card
# ──────────────────────────────────────────────────────────────────────────────
def draw_face(frame, x1, y1, x2, y2, gender, age, g_conf, a_conf):
    color = GENDER_COLOR.get(gender, (200, 200, 200))

    # ── Face bounding box ─────────────────────────────────────────────────────
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # ── Corner bracket accents ────────────────────────────────────────────────
    BL, BT = 18, 3   # bracket length, thickness
    for bx, by, sx, sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (bx, by), (bx + sx*BL, by),        color, BT)
        cv2.line(frame, (bx, by), (bx, by + sy*BL),        color, BT)

    # ── Label card ────────────────────────────────────────────────────────────
    sym   = "M" if gender == "Male" else "F" if gender == "Female" else "?"
    line1 = f" {sym}  {gender}"                        # e.g. " M  Male"
    line2 = f"    Age: {age}"                           # e.g. "    Age: 25-32"
    line3 = f"    {g_conf*100:.0f}% gender  |  {a_conf*100:.0f}% age"

    FONT  = cv2.FONT_HERSHEY_DUPLEX
    FONTS = cv2.FONT_HERSHEY_SIMPLEX

    (w1, h1), _ = cv2.getTextSize(line1, FONT,  0.72, 1)
    (w2, h2), _ = cv2.getTextSize(line2, FONT,  0.65, 1)
    (w3, h3), _ = cv2.getTextSize(line3, FONTS, 0.42, 1)

    card_w = max(w1, w2, w3) + 22
    card_h = h1 + h2 + h3 + 30
    cx     = x1
    cy     = max(y1 - card_h - 6, 2)

    # Semi-transparent card background
    ovl = frame.copy()
    cv2.rectangle(ovl, (cx, cy), (cx + card_w, cy + card_h), (18, 18, 18), -1)
    cv2.addWeighted(ovl, 0.78, frame, 0.22, 0, frame)

    # Left accent bar
    cv2.rectangle(frame, (cx, cy), (cx + 5, cy + card_h), color, -1)

    # Card border
    cv2.rectangle(frame, (cx, cy), (cx + card_w, cy + card_h), color, 1)

    # Text
    ty = cy + h1 + 8
    cv2.putText(frame, line1, (cx + 10, ty), FONT,  0.72, color,          1, cv2.LINE_AA)
    ty += h2 + 6
    cv2.putText(frame, line2, (cx + 10, ty), FONT,  0.65, (230, 230, 230), 1, cv2.LINE_AA)
    ty += h3 + 6
    cv2.putText(frame, line3, (cx + 10, ty), FONTS, 0.42, (130, 130, 145), 1, cv2.LINE_AA)

    # Confidence bar (gender)
    bx  = cx + 10
    by_ = cy + card_h - 8
    bw  = card_w - 20
    cv2.rectangle(frame, (bx, by_ - 4), (bx + bw, by_),       (45, 45, 45), -1)
    cv2.rectangle(frame, (bx, by_ - 4), (bx + int(bw*g_conf), by_), color,  -1)


# ──────────────────────────────────────────────────────────────────────────────
# DRAWING — HUD overlay
# ──────────────────────────────────────────────────────────────────────────────
def draw_hud(frame, fps, n_faces, paused):
    H, W = frame.shape[:2]

    # Top bar
    ovl = frame.copy()
    cv2.rectangle(ovl, (0, 0), (W, 54), (12, 12, 18), -1)
    cv2.addWeighted(ovl, 0.82, frame, 0.18, 0, frame)

    cv2.putText(frame, "REAL-TIME  AGE & GENDER  DETECTION",
                (16, 36), cv2.FONT_HERSHEY_DUPLEX, 0.85,
                (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps:5.1f}", (W - 190, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 255, 140), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Faces: {n_faces}", (W - 190, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 200, 255), 1, cv2.LINE_AA)

    # Gender legend
    for i, (g, col) in enumerate(GENDER_COLOR.items()):
        sym = "M" if g == "Male" else "F"
        cv2.putText(frame, f"{sym} = {g}",
                    (16, 72 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)

    # Bottom hint bar
    ovl2 = frame.copy()
    cv2.rectangle(ovl2, (0, H - 30), (W, H), (12, 12, 18), -1)
    cv2.addWeighted(ovl2, 0.82, frame, 0.18, 0, frame)

    hint = "[ PAUSED ]  " if paused else ""
    hint += "Q / ESC = Quit    S = Screenshot    P = Pause"
    cv2.putText(frame, hint, (16, H - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (160, 255, 160) if paused else (140, 140, 160),
                1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN WEBCAM LOOP
# ──────────────────────────────────────────────────────────────────────────────
def run_webcam(cam_index: int, save_dir: str):
    print("\n" + "=" * 54)
    print("  Age & Gender Detection — startup")
    print("=" * 54)

    print("[1/3] Checking / downloading models ...")
    download_models()

    print("[2/3] Loading neural networks ...")
    face_net, age_net, gender_net = load_nets()
    print("      Done.\n")

    print(f"[3/3] Opening camera index {cam_index} ...")
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"      [ERROR] Cannot open camera {cam_index}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"      Camera opened at {w}x{h}")
    print("\n  Live window is open — press Q or ESC to quit.\n")

    os.makedirs(save_dir, exist_ok=True)

    fps_time    = time.time()
    fps_counter = 0
    fps         = 0.0
    paused      = False
    last_frame  = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame read failed, retrying...")
                time.sleep(0.04)
                continue

            # Detect faces
            faces = detect_faces(frame, face_net)

            # Predict and draw each face
            for (x1, y1, x2, y2, _) in faces:
                gender, age, g_conf, a_conf = predict_age_gender(
                    frame, x1, y1, x2, y2, age_net, gender_net
                )
                draw_face(frame, x1, y1, x2, y2,
                          gender, age, g_conf, a_conf)

            # FPS
            fps_counter += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps         = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time    = now

            draw_hud(frame, fps, len(faces), paused)
            last_frame = frame

        if last_frame is not None:
            cv2.imshow("Age & Gender Detection", last_frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            os.makedirs(save_dir, exist_ok=True)
            name = os.path.join(save_dir, f"capture_{int(time.time())}.jpg")
            cv2.imwrite(name, last_frame)
            print(f"  [Saved] {name}")
        elif key == ord("p"):
            paused = not paused
            print(f"  [{'Paused' if paused else 'Resumed'}]")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[Done]")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Real-time Age & Gender Detection on webcam"
    )
    parser.add_argument("--cam",       type=int,   default=0,
                        help="Camera index (default 0)")
    parser.add_argument("--save-dir",  default="screenshots",
                        help="Folder for saved screenshots")
    parser.add_argument("--face-conf", type=float, default=FACE_CONF,
                        help=f"Face confidence threshold 0-1 (default {FACE_CONF})")
    args = parser.parse_args()

    global FACE_CONF
    FACE_CONF = args.face_conf

    run_webcam(args.cam, args.save_dir)


if __name__ == "__main__":
    main()