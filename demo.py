"""
demo.py — End-to-End Demo

Runs the full pipeline:
  1. Generate synthetic dataset
  2. Extract features & train model
  3. Evaluate model
  4. Run inference on sample clips
  5. Save evaluation report image
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

print("=" * 60)
print("  VIOLENCE & NON-VIOLENCE ACTION RECOGNITION SYSTEM")
print("  End-to-End Demo")
print("=" * 60)

# ── Step 1: Generate Dataset ──────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 1: Generating synthetic dataset...")
print("─" * 60)

from dataset_generator import generate_dataset
start = time.time()
generate_dataset(n_violence=60, n_non_violence=60, verbose=True)
print(f"  Done in {time.time()-start:.1f}s")

# ── Step 2: Train ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 2: Training the classifier...")
print("─" * 60)

from train import train
start = time.time()
model, scaler, metrics = train(verbose=True)
print(f"  Training done in {time.time()-start:.1f}s")

# ── Step 3: Sample Predictions ────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 3: Running sample predictions...")
print("─" * 60)

import glob
from feature_extractor import extract_features_from_video
from config import VIOLENCE_DIR, NON_VIOLENCE_DIR, LABELS

def predict_clip(video_path):
    feat = extract_features_from_video(video_path)
    if feat is None:
        return None, None
    feat_s = scaler.transform(feat.reshape(1, -1))
    prob   = model.predict_proba(feat_s)[0]
    pred   = 1 if prob[1] >= 0.5 else 0
    conf   = prob[pred]
    return pred, conf

v_samples  = sorted(glob.glob(os.path.join(VIOLENCE_DIR, "*.mp4")))[:5]
nv_samples = sorted(glob.glob(os.path.join(NON_VIOLENCE_DIR, "*.mp4")))[:5]

results = []
print("\n  Violence clips:")
for vp in v_samples:
    pred, conf = predict_clip(vp)
    correct = pred == 1
    status  = "✅" if correct else "❌"
    print(f"    {status} {os.path.basename(vp):20s} → {LABELS[pred]}  ({conf*100:.1f}%)")
    results.append({"file": vp, "true": 1, "pred": pred, "conf": conf})

print("\n  Non-violence clips:")
for vp in nv_samples:
    pred, conf = predict_clip(vp)
    correct = pred == 0
    status  = "✅" if correct else "❌"
    print(f"    {status} {os.path.basename(vp):20s} → {LABELS[pred]}  ({conf*100:.1f}%)")
    results.append({"file": vp, "true": 0, "pred": pred, "conf": conf})

sample_acc = np.mean([r['pred'] == r['true'] for r in results])
print(f"\n  Sample accuracy: {sample_acc*100:.1f}%")

# ── Step 4: Full Evaluation ───────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 4: Full evaluation...")
print("─" * 60)

from evaluate import evaluate
eval_results = evaluate(output_path="models/evaluation_report.png")

# ── Step 5: Summary Dashboard ─────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 5: Generating summary dashboard...")
print("─" * 60)

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0d1117')
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)

ACCENT  = '#f85149'
GREEN   = '#3fb950'
BLUE    = '#58a6ff'
YELLOW  = '#e3b341'
BG      = '#161b22'
TEXT    = '#c9d1d9'
MUTED   = '#8b949e'

# ── Title ─────────────────────────────────────────────────────────────────────
ax_title = fig.add_subplot(gs[0, :])
ax_title.set_facecolor('#0d1117')
ax_title.axis('off')
ax_title.text(0.5, 0.75, '🚨 Violence & Non-Violence Action Recognition',
              ha='center', va='center', fontsize=20, fontweight='bold',
              color=TEXT, transform=ax_title.transAxes, family='monospace')
ax_title.text(0.5, 0.2, 'Optical Flow + HOG Features  •  SVM + Random Forest Ensemble',
              ha='center', va='center', fontsize=11, color=MUTED,
              transform=ax_title.transAxes)

# ── Metric Cards ──────────────────────────────────────────────────────────────
card_data = [
    ("Test Accuracy",  f"{metrics['accuracy']*100:.1f}%",   GREEN),
    ("CV Accuracy",    f"{metrics['cv_mean']*100:.1f}%",    BLUE),
    ("Feature Dim",    str(metrics['feature_dim']),          YELLOW),
    ("Training Clips", str(metrics['n_train']),              ACCENT),
]

for i, (title, val, color) in enumerate(card_data):
    ax = fig.add_subplot(gs[1, i])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    rect = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                           boxstyle="round,pad=0.02",
                           facecolor=BG, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(0.5, 0.62, val,   ha='center', va='center', fontsize=22,
            color=color, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.25, title, ha='center', va='center', fontsize=9,
            color=MUTED, transform=ax.transAxes)

# ── Confusion Matrix ──────────────────────────────────────────────────────────
ax_cm = fig.add_subplot(gs[2, :2])
ax_cm.set_facecolor(BG)
cm = metrics['confusion_matrix']
im = ax_cm.imshow(cm, cmap='RdYlGn', interpolation='nearest', vmin=0)
ax_cm.set_title('Confusion Matrix', color=TEXT, fontsize=12, fontweight='bold', pad=8)
ax_cm.set_xticks([0, 1]); ax_cm.set_xticklabels(['Non-Violence', 'Violence'],
                                                   color=TEXT, fontsize=9)
ax_cm.set_yticks([0, 1]); ax_cm.set_yticklabels(['Non-Violence', 'Violence'],
                                                   color=TEXT, fontsize=9, rotation=90,
                                                   va='center')
ax_cm.set_xlabel('Predicted', color=MUTED, fontsize=9)
ax_cm.set_ylabel('True', color=MUTED, fontsize=9)
thresh = cm.max() / 2.
for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, str(cm[i, j]), ha='center', va='center',
                   color='white' if cm[i,j] > thresh else 'black',
                   fontsize=18, fontweight='bold')
plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

# ── Sample Predictions ────────────────────────────────────────────────────────
ax_sp = fig.add_subplot(gs[2, 2:])
ax_sp.set_facecolor(BG)
ax_sp.set_title('Sample Predictions', color=TEXT, fontsize=12, fontweight='bold', pad=8)
ax_sp.axis('off')

y_pos = 0.95
for r in results[:8]:
    label = "Violence" if r['true'] == 1 else "Non-Violence"
    correct = r['pred'] == r['true']
    marker = "●" if correct else "✗"
    c = GREEN if correct else ACCENT
    pred_name = "Violence" if r['pred'] == 1 else "Non-Violence"
    ax_sp.text(0.02, y_pos, f"{marker} {os.path.basename(r['file'])[:18]:20s}",
               transform=ax_sp.transAxes, color=c, fontsize=8, family='monospace')
    ax_sp.text(0.72, y_pos, f"{pred_name:13s} {r['conf']*100:5.1f}%",
               transform=ax_sp.transAxes, color=TEXT, fontsize=8, family='monospace')
    y_pos -= 0.12

plt.savefig("models/demo_dashboard.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()

print("  ✅ Dashboard saved → models/demo_dashboard.png")

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DEMO COMPLETE")
print("=" * 60)
print(f"\n  ✅ Test Accuracy:    {metrics['accuracy']*100:.1f}%")
print(f"  ✅ CV Accuracy:     {metrics['cv_mean']*100:.1f}% ± {metrics['cv_std']*100:.1f}%")
print(f"  ✅ Feature Dim:     {metrics['feature_dim']}")
print(f"  ✅ Train Samples:   {metrics['n_train']}")
print(f"  ✅ Test Samples:    {metrics['n_test']}")
print(f"\n  📊 Reports saved:")
print(f"     → models/evaluation_report.png")
print(f"     → models/demo_dashboard.png")
print(f"\n  🚀 Next steps:")
print(f"     • Add real videos to data/violence/ and data/non_violence/")
print(f"     • Retrain: python train.py")
print(f"     • Predict: python predict.py --video your_video.mp4")
print(f"     • Webcam:  python predict.py --webcam")
print()
