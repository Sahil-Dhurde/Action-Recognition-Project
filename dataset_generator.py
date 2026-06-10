"""
dataset_generator.py — Synthetic Video Dataset Generator

Since real datasets (RWF-2000, Hockey Fights) require downloading,
this module generates synthetic video clips that simulate the motion
characteristics of violent vs non-violent scenes.

  Non-Violence: slow, smooth, low-magnitude motion (walking, standing)
  Violence:     fast, chaotic, high-magnitude motion (fighting, struggling)
"""

import cv2
import numpy as np
import os
import random
from config import VIOLENCE_DIR, NON_VIOLENCE_DIR, FRAME_WIDTH, FRAME_HEIGHT


def _draw_person(frame, x, y, color, scale=1.0):
    """Draw a stick figure at position (x, y)."""
    h, w = frame.shape[:2]
    x, y = int(np.clip(x, 20, w-20)), int(np.clip(y, 20, h-20))
    s = int(10 * scale)

    # Head
    cv2.circle(frame, (x, y - 3*s), s, color, -1)
    # Body
    cv2.line(frame, (x, y - 2*s), (x, y + 2*s), color, max(1, s//3))
    # Arms
    cv2.line(frame, (x - 2*s, y), (x + 2*s, y), color, max(1, s//3))
    # Legs
    cv2.line(frame, (x, y + 2*s), (x - s, y + 4*s), color, max(1, s//3))
    cv2.line(frame, (x, y + 2*s), (x + s, y + 4*s), color, max(1, s//3))


def generate_non_violence_clip(filepath, n_frames=40, fps=15):
    """
    Generate a non-violent video: two figures walking slowly.
    Motion is smooth, slow, predictable.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

    # Initial positions
    x1, y1 = FRAME_WIDTH // 3, FRAME_HEIGHT // 2
    x2, y2 = 2 * FRAME_WIDTH // 3, FRAME_HEIGHT // 2

    dx1 = random.uniform(1.5, 3.0) * random.choice([-1, 1])
    dy1 = random.uniform(0.5, 1.5) * random.choice([-1, 1])
    dx2 = random.uniform(1.5, 3.0) * random.choice([-1, 1])
    dy2 = random.uniform(0.5, 1.5) * random.choice([-1, 1])

    bg_color = random.choice([
        (210, 180, 140), (170, 200, 170), (200, 200, 220)
    ])

    for f in range(n_frames):
        frame = np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        frame[:] = bg_color

        # Slight background variation
        noise = np.random.randint(-5, 5, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Smooth motion
        x1 += dx1 + np.random.randn() * 0.3
        y1 += dy1 + np.random.randn() * 0.3
        x2 += dx2 + np.random.randn() * 0.3
        y2 += dy2 + np.random.randn() * 0.3

        # Bounce off walls
        if x1 < 20 or x1 > FRAME_WIDTH - 20:   dx1 *= -1
        if y1 < 20 or y1 > FRAME_HEIGHT - 20:  dy1 *= -1
        if x2 < 20 or x2 > FRAME_WIDTH - 20:   dx2 *= -1
        if y2 < 20 or y2 > FRAME_HEIGHT - 20:  dy2 *= -1

        _draw_person(frame, x1, y1, (50, 80, 200))
        _draw_person(frame, x2, y2, (200, 80, 50))

        out.write(frame)

    out.release()


def generate_violence_clip(filepath, n_frames=40, fps=15):
    """
    Generate a violent video: two figures clashing with fast, chaotic motion.
    Motion is rapid, erratic, high magnitude.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

    cx = FRAME_WIDTH // 2
    cy = FRAME_HEIGHT // 2

    bg_color = random.choice([
        (100, 100, 120), (90, 80, 100), (110, 100, 90)
    ])

    for f in range(n_frames):
        frame = np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        frame[:] = bg_color

        # Flickering background (simulates energy/chaos)
        flicker = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        frame = np.clip(frame.astype(np.int16) + flicker - 15, 0, 255).astype(np.uint8)

        # Two figures very close together, moving chaotically
        phase = f * 0.8
        offset = int(10 * np.sin(phase))
        speed = 12

        x1 = cx - 15 + int(speed * np.sin(phase * 1.7)) + random.randint(-6, 6)
        y1 = cy + offset + random.randint(-8, 8)
        x2 = cx + 15 + int(speed * np.cos(phase * 1.3)) + random.randint(-6, 6)
        y2 = cy - offset + random.randint(-8, 8)

        _draw_person(frame, x1, y1, (0, 60, 255))
        _draw_person(frame, x2, y2, (0, 180, 255))

        # Motion blur effect (simulate speed)
        if random.random() > 0.4:
            cv2.blur(frame, (3, 3), frame)

        # Impact flash
        if random.random() > 0.75:
            px = (x1 + x2) // 2 + random.randint(-10, 10)
            py = (y1 + y2) // 2 + random.randint(-10, 10)
            cv2.circle(frame, (px, py), random.randint(4, 10), (0, 255, 255), -1)

        out.write(frame)

    out.release()


def generate_dataset(n_violence=60, n_non_violence=60, verbose=True):
    """
    Generate a complete synthetic dataset.

    Args:
        n_violence:     Number of violent clips to generate
        n_non_violence: Number of non-violent clips to generate
        verbose:        Print progress
    """
    if verbose:
        print("=" * 55)
        print("  Generating Synthetic Dataset")
        print("=" * 55)

    # ── Non-Violence ──────────────────────────────────────────
    if verbose:
        print(f"\n[1/2] Generating {n_non_violence} non-violent clips...")
    for i in range(n_non_violence):
        path = os.path.join(NON_VIOLENCE_DIR, f"nv_{i:04d}.mp4")
        generate_non_violence_clip(path, n_frames=random.randint(30, 50))
        if verbose and (i + 1) % 10 == 0:
            print(f"      {i+1}/{n_non_violence} done")

    # ── Violence ──────────────────────────────────────────────
    if verbose:
        print(f"\n[2/2] Generating {n_violence} violent clips...")
    for i in range(n_violence):
        path = os.path.join(VIOLENCE_DIR, f"v_{i:04d}.mp4")
        generate_violence_clip(path, n_frames=random.randint(30, 50))
        if verbose and (i + 1) % 10 == 0:
            print(f"      {i+1}/{n_violence} done")

    if verbose:
        print(f"\n✅ Dataset ready!")
        print(f"   Violence:     {n_violence} clips → {VIOLENCE_DIR}")
        print(f"   Non-Violence: {n_non_violence} clips → {NON_VIOLENCE_DIR}")


if __name__ == "__main__":
    generate_dataset(n_violence=80, n_non_violence=80)
