"""
evaluate.py — Model Evaluation & Visualization

Generates:
  • Confusion matrix heatmap
  • ROC curve
  • Feature importance plot (from Random Forest)
  • Per-sample confidence distribution
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    classification_report, accuracy_score
)

from config import (
    VIOLENCE_DIR, NON_VIOLENCE_DIR,
    LABELS, TEST_SIZE, RANDOM_STATE
)
from feature_extractor import extract_features_from_video
from train import load_model, load_dataset

VIDEO_EXTS = ["*.mp4", "*.avi", "*.mov", "*.mkv"]


def plot_confusion_matrix(cm, ax, title="Confusion Matrix"):
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Non-Violence", "Violence"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')

    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)


def plot_roc(y_true, y_prob, ax):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, color='crimson', lw=2.5,
            label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.1, color='crimson')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_confidence_dist(y_true, y_prob, ax):
    mask_nv = y_true == 0
    mask_v  = y_true == 1

    ax.hist(y_prob[mask_nv], bins=20, alpha=0.7, color='steelblue',
            label='Non-Violence', edgecolor='white', linewidth=0.5)
    ax.hist(y_prob[mask_v], bins=20, alpha=0.7, color='crimson',
            label='Violence', edgecolor='white', linewidth=0.5)
    ax.axvline(0.5, color='black', linestyle='--', lw=1.5, label='Decision boundary')
    ax.set_xlabel('Violence Probability', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Confidence Distribution', fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')


def plot_feature_importance(model, ax, top_n=20):
    """Plot feature importance from the Random Forest component."""
    try:
        rf = model.estimators_[1]  # RF is index 1 in our ensemble
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        # Feature names
        feature_names = (
            [f"Flow_mean_{i}" for i in range(18)] +
            [f"Flow_std_{i}"  for i in range(18)] +
            [f"Flow_max_{i}"  for i in range(18)] +
            [f"HOG_{i}"       for i in range(64)]
        )

        top_names = [feature_names[i] if i < len(feature_names) else f"f{i}"
                     for i in indices]
        top_vals  = importances[indices]

        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))[::-1]
        bars = ax.barh(range(top_n), top_vals[::-1], color=colors[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1], fontsize=8)
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_title(f'Top {top_n} Feature Importances (RF)', fontsize=13,
                     fontweight='bold', pad=12)
        ax.grid(True, alpha=0.3, axis='x')
    except Exception as e:
        ax.text(0.5, 0.5, f"Feature importance\nnot available\n({e})",
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Feature Importances', fontsize=13, fontweight='bold')


def evaluate(output_path="models/evaluation_report.png", verbose=True):
    """Run full evaluation and save plots."""
    print("=" * 55)
    print("  Violence Detection — Evaluation")
    print("=" * 55)

    # ── Load model ─────────────────────────────────────────────
    model, scaler = load_model()

    # ── Load dataset ───────────────────────────────────────────
    print("\n[1/3] Loading dataset...")
    X, y = load_dataset(verbose=False)

    if len(X) == 0:
        print("  No videos found. Please train the model first.")
        return

    X_scaled = scaler.transform(X)

    # ── Predictions ────────────────────────────────────────────
    print("[2/3] Running predictions...")
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    acc = accuracy_score(y, y_pred)
    cm  = confusion_matrix(y, y_pred)

    print(f"\n  Overall Accuracy: {acc*100:.1f}%")
    print("\n  Classification Report:")
    target_names = ["Non-Violence", "Violence"]
    print(classification_report(y, y_pred, target_names=target_names))

    # ── Plots ──────────────────────────────────────────────────
    print("[3/3] Generating evaluation plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Violence Detection — Model Evaluation",
                 fontsize=16, fontweight='bold', y=0.98)
    fig.patch.set_facecolor('#f8f9fa')
    for ax in axes.flat:
        ax.set_facecolor('#ffffff')

    plot_confusion_matrix(cm, axes[0, 0])
    plot_roc(y, y_prob, axes[0, 1])
    plot_confidence_dist(y, y_prob, axes[1, 0])
    plot_feature_importance(model, axes[1, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"\n✅ Evaluation report saved → {output_path}")
    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "n_samples": len(X),
    }


if __name__ == "__main__":
    evaluate()
