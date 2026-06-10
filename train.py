"""
train.py — Violence Detection Model Training Pipeline

Steps:
  1. Scan dataset directories for video clips
  2. Extract optical flow + HOG features per clip
  3. Train ensemble (SVM + Random Forest)
  4. Save model and scaler
  5. Print training report
"""

import os
import glob
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    VIOLENCE_DIR, NON_VIOLENCE_DIR,
    MODEL_PATH, SCALER_PATH,
    TEST_SIZE, RANDOM_STATE, LABELS
)
from feature_extractor import extract_features_from_video


VIDEO_EXTS = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.flv"]


def _find_videos(directory):
    videos = []
    for ext in VIDEO_EXTS:
        videos.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(videos)


def load_dataset(verbose=True):
    """
    Load all videos from dataset directories and extract features.

    Returns:
        X: np.array of shape (N, F)
        y: np.array of shape (N,) with labels 0=non-violence, 1=violence
    """
    X, y = [], []

    # ── Non-Violence (label=0) ─────────────────────────────────
    nv_videos = _find_videos(NON_VIOLENCE_DIR)
    if verbose:
        print(f"  Found {len(nv_videos)} non-violent videos")

    for i, vp in enumerate(nv_videos):
        feat = extract_features_from_video(vp)
        if feat is not None:
            X.append(feat)
            y.append(0)
        if verbose and (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(nv_videos)} processed")

    # ── Violence (label=1) ─────────────────────────────────────
    v_videos = _find_videos(VIOLENCE_DIR)
    if verbose:
        print(f"\n  Found {len(v_videos)} violent videos")

    for i, vp in enumerate(v_videos):
        feat = extract_features_from_video(vp)
        if feat is not None:
            X.append(feat)
            y.append(1)
        if verbose and (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(v_videos)} processed")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def build_model():
    """Build an ensemble classifier (SVM + Random Forest)."""
    svm = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        probability=True,
        random_state=RANDOM_STATE
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    ensemble = VotingClassifier(
        estimators=[('svm', svm), ('rf', rf)],
        voting='soft',
        weights=[1, 1]
    )

    return ensemble


def train(verbose=True):
    """Full training pipeline. Returns trained model and test metrics."""
    print("=" * 55)
    print("  Violence Detection — Training Pipeline")
    print("=" * 55)

    # ── 1. Load dataset ────────────────────────────────────────
    print("\n[1/4] Loading dataset & extracting features...")
    X, y = load_dataset(verbose=verbose)

    if len(X) == 0:
        raise RuntimeError(
            "No videos found! Please add videos to:\n"
            f"  • {VIOLENCE_DIR}\n"
            f"  • {NON_VIOLENCE_DIR}\n"
            "Or run dataset_generator.py first."
        )

    print(f"\n  Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"  Violence:     {int(y.sum())} samples")
    print(f"  Non-Violence: {int((y == 0).sum())} samples")

    # ── 2. Scale features ──────────────────────────────────────
    print("\n[2/4] Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 3. Split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 4. Train ───────────────────────────────────────────────
    print("\n[3/4] Training ensemble model (SVM + Random Forest)...")
    model = build_model()
    model.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"  5-Fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── 5. Evaluate ────────────────────────────────────────────
    print("\n[4/4] Evaluating on test set...")
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"\n  Test Accuracy: {acc:.3f} ({acc*100:.1f}%)")
    print("\n  Classification Report:")
    target_names = [LABELS[0], LABELS[1]]
    print(classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(f"    {cm}")

    # ── 6. Save ────────────────────────────────────────────────
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n✅ Model saved → {MODEL_PATH}")
    print(f"✅ Scaler saved → {SCALER_PATH}")

    return model, scaler, {
        'accuracy': acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'confusion_matrix': cm,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'feature_dim': X.shape[1],
    }


def load_model():
    """Load saved model and scaler. Raises FileNotFoundError if not trained."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}\nRun train.py first."
        )
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


if __name__ == "__main__":
    train()
