"""
2-Person Violence / Non-Violence Detection  (distance-aware)
=============================================================
KEY FIX: If two people are far apart, all fight / mutual-aggression
labels are suppressed regardless of individual motion scores.
Individual motion is shown as "Active" (not "Aggressive") when persons
are out of striking range.

Distance tiers (normalised hip-to-hip, 0 = touching, 1 = opposite edges):
  CLOSE   < 0.35  → fight detection fully active
  MEDIUM  0.35–0.55 → mutual aggression possible, no fight banner
  FAR     > 0.55  → always NON-VIOLENT regardless of motion

Detection layers (only applied within distance tier):
  Per-person  : wrist speed, arm extension, torso lean, asymmetric arm motion
  Inter-person: proximity, mutual wrist approach, facing each other

Requirements:
    pip install mediapipe opencv-python pyttsx3

Usage:
    python violence_detection_2person.py              # webcam
    python violence_detection_2person.py --source video.mp4

Controls:
    q — quit    s — save frame    v — toggle detail panel
"""

import argparse
import collections
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import cv2
import mediapipe as mp
import pyttsx3

# ─── MediaPipe setup ─────────────────────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

L          = mp_pose.PoseLandmark
NOSE       = L.NOSE.value
L_SHOULDER = L.LEFT_SHOULDER.value
R_SHOULDER = L.RIGHT_SHOULDER.value
L_WRIST    = L.LEFT_WRIST.value
R_WRIST    = L.RIGHT_WRIST.value
L_HIP      = L.LEFT_HIP.value
R_HIP      = L.RIGHT_HIP.value
L_KNEE     = L.LEFT_KNEE.value
R_KNEE     = L.RIGHT_KNEE.value
L_ANKLE    = L.LEFT_ANKLE.value
R_ANKLE    = L.RIGHT_ANKLE.value

VIS_THRESH = 0.4

# ─── Distance tier thresholds (normalised hip-to-hip distance) ───────────────────
DIST_CLOSE  = 0.35   # below this → fight detection active
DIST_MEDIUM = 0.55   # below this → mutual aggression possible; no fight
                     # above this → always NON-VIOLENT


# ─── TTS ─────────────────────────────────────────────────────────────────────────
class Speaker:
    FIGHT_COOLDOWN  = 1.2
    NORMAL_COOLDOWN = 3.0

    def __init__(self, rate=175, volume=1.0):
        self._eng = pyttsx3.init()
        self._eng.setProperty("rate",   rate)
        self._eng.setProperty("volume", volume)
        self._last = ""
        self._t    = 0.0
        self._lock = threading.Lock()
        self._busy = False

    def speak(self, text: str, urgent=False):
        cd  = self.FIGHT_COOLDOWN if urgent else self.NORMAL_COOLDOWN
        now = time.time()
        with self._lock:
            if self._busy:
                return
            if text == self._last and (now - self._t) < cd:
                return
            self._last = text
            self._t    = now
            self._busy = True

        def _run():
            try:
                self._eng.say(text)
                self._eng.runAndWait()
            finally:
                with self._lock:
                    self._busy = False

        threading.Thread(target=_run, daemon=True).start()


# ─── Per-person snapshot ──────────────────────────────────────────────────────────
@dataclass
class PersonSnap:
    lw:   tuple
    rw:   tuple
    ls:   tuple
    rs:   tuple
    hip:  tuple
    nose: tuple
    ts:   float = field(default_factory=time.time)

    @property
    def chest(self):
        return ((self.ls[0] + self.rs[0]) / 2,
                (self.ls[1] + self.rs[1]) / 2)


# ─── Distance tier helper ─────────────────────────────────────────────────────────
def distance_tier(hip_dist: float) -> str:
    """Return 'close', 'medium', or 'far' based on normalised hip distance."""
    if hip_dist < DIST_CLOSE:
        return "close"
    if hip_dist < DIST_MEDIUM:
        return "medium"
    return "far"


# ─── Per-person violence scorer ───────────────────────────────────────────────────
class PersonViolenceScorer:
    """
    Scores a single person's motion level (0.0 – 1.0).
    The caller decides whether to label it 'aggression' based on distance tier.
    """

    WEIGHTS = dict(
        wrist_speed      = 0.40,
        asymmetric_speed = 0.25,
        arm_extension    = 0.15,
        torso_lean       = 0.10,
        hip_speed        = 0.10,
    )

    WRIST_SPEED_MAX = 0.17
    ASYM_SPEED_MAX  = 0.13
    ARM_EXT_MAX     = 0.58
    TORSO_LEAN_MAX  = 0.10
    HIP_SPEED_MAX   = 0.09
    SPIKE_THRESH    = 0.17
    EMA_ALPHA       = 0.60

    def __init__(self, window=4):
        self._hist: Deque[PersonSnap] = collections.deque(maxlen=window)
        self._prev: Optional[PersonSnap] = None
        self.ema   = 0.0
        self.spike = False
        self.comps = {}

    @staticmethod
    def _d(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

    @staticmethod
    def _spd(seq):
        if len(seq) < 2:
            return 0.0
        return sum(math.hypot(seq[i][0]-seq[i-1][0], seq[i][1]-seq[i-1][1])
                   for i in range(1, len(seq))) / (len(seq) - 1)

    @staticmethod
    def _c(v): return max(0.0, min(1.0, v))

    def update(self, snap: PersonSnap) -> Tuple[float, bool]:
        self.spike = False
        if self._prev:
            lv = self._d(snap.lw, self._prev.lw)
            rv = self._d(snap.rw, self._prev.rw)
            self.spike = max(lv, rv) >= self.SPIKE_THRESH
        self._prev = snap
        self._hist.append(snap)
        hist = list(self._hist)

        if len(hist) < 2:
            return 0.0, self.spike

        lw_spd = self._spd([f.lw  for f in hist])
        rw_spd = self._spd([f.rw  for f in hist])

        self.comps = dict(
            wrist_speed      = self._c(max(lw_spd, rw_spd) / self.WRIST_SPEED_MAX),
            asymmetric_speed = self._c(abs(lw_spd - rw_spd) / self.ASYM_SPEED_MAX),
            arm_extension    = self._c(
                max(self._d(snap.lw, snap.ls),
                    self._d(snap.rw, snap.rs)) / self.ARM_EXT_MAX),
            torso_lean       = self._c(
                abs(snap.nose[0] - snap.hip[0]) / self.TORSO_LEAN_MAX),
            hip_speed        = self._c(
                self._spd([f.hip for f in hist]) / self.HIP_SPEED_MAX),
        )
        raw      = sum(self.WEIGHTS[k] * v for k, v in self.comps.items())
        self.ema = self.EMA_ALPHA * raw + (1 - self.EMA_ALPHA) * self.ema
        return self.ema, self.spike

    def reset(self):
        self._hist.clear()
        self._prev = None
        self.ema   = 0.0
        self.spike = False
        self.comps = {}


# ─── Inter-person fight analyser ─────────────────────────────────────────────────
class FightAnalyser:
    """
    Determines fight state taking DISTANCE TIER into account.

    Returns:
        interaction_score : 0..1  (proximity + wrist approach)
        fight_label       : 'fight' | 'mutual_agg' | 'non_violent'
        info dict
    """

    PROXIMITY_THRESH   = DIST_CLOSE   # same as distance tier
    WRIST_APPROACH_MAX = 0.35
    AGG_THRESH         = 0.34

    @staticmethod
    def _d(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

    @staticmethod
    def _c(v): return max(0.0, min(1.0, v))

    def analyse(self, s1: PersonSnap, s2: PersonSnap,
                score1: float, score2: float,
                tier: str) -> Tuple[float, str, dict]:
        hip_dist   = self._d(s1.hip, s2.hip)
        prox_score = self._c(1.0 - hip_dist / max(self.PROXIMITY_THRESH, 0.001))

        p1_threat = min(self._d(s1.lw, s2.chest), self._d(s1.rw, s2.chest))
        p2_threat = min(self._d(s2.lw, s1.chest), self._d(s2.rw, s1.chest))
        approach_score = self._c(
            1.0 - min(p1_threat, p2_threat) / max(self.WRIST_APPROACH_MAX, 0.001)
        )

        both_agg     = (score1 >= self.AGG_THRESH) and (score2 >= self.AGG_THRESH)
        interaction  = 0.5 * prox_score + 0.5 * approach_score

        # ── distance-gated label ──────────────────────────────────────────────
        if tier == "far":
            # People are too far apart — no fight possible
            fight_label = "non_violent"

        elif tier == "medium":
            # In medium range: acknowledge mutual aggression but NOT a fight
            fight_label = "mutual_agg" if both_agg else "non_violent"

        else:
            # Close range: full fight detection
            fight_label = "fight" if both_agg else "non_violent"

        info = dict(
            hip_dist       = round(hip_dist,      3),
            prox_score     = round(prox_score,    2),
            approach_score = round(approach_score, 2),
            both_agg       = both_agg,
            tier           = tier,
        )
        return interaction, fight_label, info


# ─── Pose label ──────────────────────────────────────────────────────────────────
def pose_label(lm_list) -> str:
    lm = lm_list

    def vis(*idx): return all(lm[i].visibility > VIS_THRESH for i in idx)
    def y(i):      return lm[i].y

    lw_up = vis(L_WRIST, L_SHOULDER) and y(L_WRIST) < y(L_SHOULDER) - 0.05
    rw_up = vis(R_WRIST, R_SHOULDER) and y(R_WRIST) < y(R_SHOULDER) - 0.05
    if lw_up and rw_up: return "Both hands raised"
    if lw_up:           return "Left hand raised"
    if rw_up:           return "Right hand raised"

    if vis(L_HIP, R_HIP, L_KNEE, R_KNEE):
        hip_y  = (y(L_HIP)  + y(R_HIP))  / 2
        knee_y = (y(L_KNEE) + y(R_KNEE)) / 2
        if knee_y - hip_y < 0.12:
            return "Sitting"

    if vis(L_ANKLE, R_ANKLE):
        if abs(y(L_ANKLE) - y(R_ANKLE)) > 0.08:
            return "Walking"

    return "Standing"


# ─── Extract PersonSnap ───────────────────────────────────────────────────────────
def extract_snap(lm_list) -> Optional[PersonSnap]:
    lm   = lm_list
    need = [L_WRIST, R_WRIST, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, NOSE]
    if not all(lm[i].visibility > VIS_THRESH for i in need):
        return None
    xy  = lambda i: (lm[i].x, lm[i].y)
    hip = ((lm[L_HIP].x + lm[R_HIP].x) / 2,
           (lm[L_HIP].y + lm[R_HIP].y) / 2)
    return PersonSnap(xy(L_WRIST), xy(R_WRIST),
                      xy(L_SHOULDER), xy(R_SHOULDER),
                      hip, xy(NOSE))


# ─── UI helpers ──────────────────────────────────────────────────────────────────
P_COLORS   = [(0, 200, 255), (0, 255, 140)]
P_AGG_COLOR = (0, 60, 255)


def _bar(frame, score, x, y, w=120, h=12):
    cv2.rectangle(frame, (x, y), (x+w, y+h), (40, 40, 40), -1)
    fill = int(score * w)
    r    = int(score * 255)
    g    = int((1 - score) * 180)
    cv2.rectangle(frame, (x, y), (x+fill, y+h), (0, g, r), -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (130, 130, 130), 1)


def _proximity_bar(frame, hip_dist: float, x, y, w=130, h=12):
    """
    Visual distance indicator.
    Green (far) → amber (medium) → red (close).
    """
    cv2.rectangle(frame, (x, y), (x+w, y+h), (40, 40, 40), -1)
    # Clamp normalised distance to 0..1 within DIST_MEDIUM range
    t    = max(0.0, min(1.0, 1.0 - hip_dist / DIST_MEDIUM))
    fill = int(t * w)
    if hip_dist < DIST_CLOSE:
        bar_col = (0, 0, 220)     # red  — close/dangerous
    elif hip_dist < DIST_MEDIUM:
        bar_col = (0, 165, 255)   # amber — medium
    else:
        bar_col = (0, 200, 80)    # green — far/safe
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x+fill, y+h), bar_col, -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (130, 130, 130), 1)


def draw_person_box(frame, pid, score: float, spike: bool,
                    plabel: str, is_agg: bool, tier: str,
                    comps: dict, show_details: bool):
    """Per-person HUD panel. Uses 'Active' label when persons are far apart."""
    H, W  = frame.shape[:2]
    x0    = 10 if pid == 0 else W // 2 + 10
    y0    = 100
    pw, ph = 230, 130 if show_details else 75

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0+pw, y0+ph), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"Person {pid+1}", (x0+8, y0+22),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, P_COLORS[pid], 2, cv2.LINE_AA)

    # ── status label changes depending on distance tier ───────────────────────
    if tier == "far":
        # Far away: never label as aggressive even if motion is high
        status    = "Non-Violent"
        col       = (0, 220, 80)
    elif is_agg:
        status    = "AGGRESSIVE"
        col       = (0, 60, 255)
    else:
        status    = "Non-Violent"
        col       = (0, 220, 80)

    cv2.putText(frame, status, (x0+8, y0+44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    cv2.putText(frame, plabel, (x0+8, y0+62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    _bar(frame, score, x0+8, y0+68, w=pw-16, h=10)
    cv2.putText(frame, f"{score:.2f}", (x0+pw-38, y0+79),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    if spike and tier != "far":
        cv2.putText(frame, "SPIKE!", (x0+pw-70, y0+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    if show_details and comps:
        dy = y0 + 86
        for nm, val in comps.items():
            cv2.putText(frame, f"{nm[:14]}: {val:.2f}",
                        (x0+8, dy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (140, 200, 255), 1)
            dy += 14


def draw_main_banner(frame, fight_label: str,
                     agg1: bool, agg2: bool,
                     inter_score: float, inter_info: dict):
    H, W  = frame.shape[:2]
    tick  = int(time.time() * 6) % 2
    tier  = inter_info.get("tier", "far")
    hd    = inter_info.get("hip_dist", 1.0)

    # ── banner content decided by distance-gated fight_label ──────────────────
    if fight_label == "fight":
        bg  = (0, 0, 200) if tick else (0, 0, 140)
        txt = "FIGHT DETECTED"
        col = (0, 60, 255)
    elif fight_label == "mutual_agg":
        bg  = (0, 50, 160)
        txt = "MUTUAL AGGRESSION (not close)"
        col = (0, 140, 255)
    else:
        # Always non-violent when far OR when neither person is aggressive
        bg  = (20, 20, 20)
        txt = "NON-VIOLENT"
        col = (0, 220, 80)

    # Distance override label shown in banner for transparency
    if tier == "far":
        dist_note = "  [FAR - safe distance]"
        note_col  = (0, 200, 80)
    elif tier == "medium":
        dist_note = "  [MEDIUM distance]"
        note_col  = (0, 165, 255)
    else:
        dist_note = "  [CLOSE range]"
        note_col  = (0, 80, 255)

    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (W, 60), bg, -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

    cv2.putText(frame, txt, (14, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, col, 2, cv2.LINE_AA)
    cv2.putText(frame, dist_note,
                (14 + int(len(txt) * 13.5), 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, note_col, 1, cv2.LINE_AA)

    # Proximity / interaction bar
    cv2.putText(frame, "Proximity:", (W-250, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    _proximity_bar(frame, hd, W-160, 10, w=140, h=14)
    cv2.putText(frame, f"d={hd:.2f}", (W-15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1,
                cv2.LINE_AA)

    # Red border only for actual fight (close + both aggressive)
    if fight_label == "fight":
        thick = 8 if tick else 5
        cv2.rectangle(frame, (0, 0), (W-1, H-1), (0, 0, 255), thick)


# ─── Two-person pose detection ───────────────────────────────────────────────────
def detect_two_persons(frame, pose_l, pose_r):
    H, W = frame.shape[:2]
    mid  = W // 2

    left_half  = frame[:, :mid]
    right_half = frame[:, mid:]

    res_l = pose_l.process(cv2.cvtColor(left_half,  cv2.COLOR_BGR2RGB))
    res_r = pose_r.process(cv2.cvtColor(right_half, cv2.COLOR_BGR2RGB))

    persons = []
    for res, x_offset in [(res_l, 0), (res_r, mid)]:
        if res and res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            for pt in lm:
                pt.x = (pt.x * (W // 2) + x_offset) / W
            persons.append((res.pose_landmarks, lm))
        else:
            persons.append(None)

    return persons, mid


def draw_skeleton(frame, landmarks, color, conn_color):
    spec_lm   = mp_drawing.DrawingSpec(color=color,     thickness=3, circle_radius=4)
    spec_conn = mp_drawing.DrawingSpec(color=conn_color, thickness=2)
    mp_drawing.draw_landmarks(
        frame, landmarks, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=spec_lm,
        connection_drawing_spec=spec_conn,
    )


# ─── Main frame processor ────────────────────────────────────────────────────────
def process_frame(frame, pose_l, pose_r,
                  scorers: list, fight_analyser: FightAnalyser,
                  speaker: Speaker, show_details: bool,
                  person_threshold: float):
    out   = frame.copy()
    H, W  = out.shape[:2]
    mid   = W // 2

    persons, _ = detect_two_persons(frame, pose_l, pose_r)
    cv2.line(out, (mid, 0), (mid, H), (60, 60, 60), 1)

    snaps      = [None, None]
    scores     = [0.0, 0.0]
    spikes     = [False, False]
    plabels    = ["Unknown", "Unknown"]
    agg        = [False, False]
    comps_list = [{}, {}]

    for pid, person in enumerate(persons):
        if person is None:
            scorers[pid].reset()
            x0 = 10 if pid == 0 else mid + 10
            cv2.putText(out, f"Person {pid+1}: not detected",
                        (x0, 130), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (80, 80, 80), 1, cv2.LINE_AA)
            continue

        landmarks, lm = person
        snap = extract_snap(lm)
        if snap is None:
            scorers[pid].reset()
            continue

        snaps[pid]      = snap
        score, spike    = scorers[pid].update(snap)
        scores[pid]     = score
        spikes[pid]     = spike
        plabels[pid]    = pose_label(lm)
        comps_list[pid] = scorers[pid].comps
        # Raw aggression flag (may be overridden by tier below)
        agg[pid]        = spike or (score >= person_threshold)

    # ── Compute distance tier BEFORE drawing any labels ───────────────────────
    inter_score  = 0.0
    fight_label  = "non_violent"
    inter_info   = {"hip_dist": 1.0, "tier": "far",
                    "prox_score": 0.0, "approach_score": 0.0,
                    "both_agg": False}

    if snaps[0] and snaps[1]:
        hip_dist    = math.hypot(snaps[0].hip[0] - snaps[1].hip[0],
                                 snaps[0].hip[1] - snaps[1].hip[1])
        tier        = distance_tier(hip_dist)
        inter_score, fight_label, inter_info = fight_analyser.analyse(
            snaps[0], snaps[1], scores[0], scores[1], tier
        )

        # ── Override per-person agg flags when people are far apart ──────────
        # Two people exercising independently at opposite ends of a room
        # should NEVER trigger aggression labels.
        if tier == "far":
            agg[0] = False
            agg[1] = False

        # Draw chest-to-chest line with tier-appropriate colour
        c1 = (int(snaps[0].chest[0] * W), int(snaps[0].chest[1] * H))
        c2 = (int(snaps[1].chest[0] * W), int(snaps[1].chest[1] * H))
        line_col = (
            (0, 0, 220)   if fight_label == "fight" else
            (0, 165, 255) if fight_label == "mutual_agg" else
            (50, 50, 50)
        )
        cv2.line(out, c1, c2, line_col, 2 if fight_label != "non_violent" else 1)
        mx, my = (c1[0]+c2[0])//2, (c1[1]+c2[1])//2
        cv2.putText(out, f"d:{inter_info['hip_dist']:.2f}",
                    (mx-20, my-6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (130, 130, 130), 1)
    else:
        tier = "far"   # default when only one person visible

    # ── Draw skeletons ────────────────────────────────────────────────────────
    for pid, person in enumerate(persons):
        if person is None:
            continue
        landmarks, _ = person
        sk_col   = P_AGG_COLOR if agg[pid] else P_COLORS[pid]
        conn_col = (0, 30, 180) if agg[pid] else tuple(int(c*0.5) for c in P_COLORS[pid])
        draw_skeleton(out, landmarks, sk_col, conn_col)

    # ── Per-person HUD panels ─────────────────────────────────────────────────
    for pid in range(2):
        if snaps[pid] is None:
            continue
        t = inter_info.get("tier", tier)
        draw_person_box(out, pid, scores[pid], spikes[pid],
                        plabels[pid], agg[pid], t,
                        comps_list[pid], show_details)

    # ── Main banner ───────────────────────────────────────────────────────────
    draw_main_banner(out, fight_label, agg[0], agg[1], inter_score, inter_info)

    # ── TTS — distance-gated ──────────────────────────────────────────────────
    if fight_label == "fight":
        speaker.speak("Warning! Fight detected!", urgent=True)
    elif fight_label == "mutual_agg":
        speaker.speak("Aggression detected!", urgent=True)
    else:
        speaker.speak("Non violent", urgent=False)

    return out


# ─── Run ──────────────────────────────────────────────────────────────────────────
def run(source, speaker, window, person_thresh):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        sys.exit(1)

    scorers      = [PersonViolenceScorer(window=window) for _ in range(2)]
    fight_anal   = FightAnalyser()
    show_details = False

    pose_cfg = dict(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("=" * 60)
    print("2-Person Violence Detection  (distance-aware)")
    print("=" * 60)
    print(f"  CLOSE  < {DIST_CLOSE}  → full fight detection")
    print(f"  MEDIUM < {DIST_MEDIUM} → mutual aggression only")
    print(f"  FAR    ≥ {DIST_MEDIUM} → always NON-VIOLENT")
    print("  q=quit   s=save frame   v=toggle details")
    print("=" * 60)

    with mp_pose.Pose(**pose_cfg) as pose_l, \
         mp_pose.Pose(**pose_cfg) as pose_r:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = process_frame(frame, pose_l, pose_r,
                                scorers, fight_anal, speaker,
                                show_details, person_thresh)
            cv2.imshow("2-Person Violence Detection", out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite("output_2person.png", out)
                print("[INFO] Saved → output_2person.png")
            elif key == ord("v"):
                show_details = not show_details

    cap.release()
    cv2.destroyAllWindows()


# ─── Entry point ─────────────────────────────────────────────────────────────────
def main():
    # Declare globals first — before any reference to these names
    global DIST_CLOSE, DIST_MEDIUM

    p = argparse.ArgumentParser(
        description="2-person violence detection — distance-aware"
    )
    p.add_argument("--source",        default="0",
                   help="0=webcam, or path to video file")
    p.add_argument("--tts-rate",      type=int,   default=175)
    p.add_argument("--tts-volume",    type=float, default=1.0)
    p.add_argument("--window",        type=int,   default=4,
                   help="Sliding window frames per person (default 4)")
    p.add_argument("--person-thresh", type=float, default=0.36,
                   help="Per-person aggression score threshold (default 0.36)")
    p.add_argument("--spike-thresh",  type=float, default=0.17,
                   help="Instant wrist-spike threshold (default 0.17)")
    p.add_argument("--close-dist",    type=float, default=DIST_CLOSE,
                   help=f"Hip distance for CLOSE tier (default {DIST_CLOSE})")
    p.add_argument("--medium-dist",   type=float, default=DIST_MEDIUM,
                   help=f"Hip distance for MEDIUM tier (default {DIST_MEDIUM})")
    args = p.parse_args()

    # Apply CLI overrides
    DIST_CLOSE  = args.close_dist
    DIST_MEDIUM = args.medium_dist
    PersonViolenceScorer.SPIKE_THRESH = args.spike_thresh
    FightAnalyser.AGG_THRESH          = args.person_thresh
    FightAnalyser.PROXIMITY_THRESH    = DIST_CLOSE

    speaker = Speaker(rate=args.tts_rate, volume=args.tts_volume)
    src     = args.source
    try:
        src = int(src)
    except ValueError:
        pass

    run(src, speaker, args.window, args.person_thresh)


if __name__ == "__main__":
    main()