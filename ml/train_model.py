import joblib
import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import TRAINING_DATA_PATH, MODEL_PATH, DATA_DIR
from ml.features import FEATURE_NAMES
from ml.generate_training_data import generate_synthetic_data, FAILURE_MODES

def train():
    # Force fresh generation across all failure modes
    generate_synthetic_data()

    df = pd.read_csv(TRAINING_DATA_PATH)
    X = df[FEATURE_NAMES]
    y = df["label"]

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- Model Evaluation ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Per-failure-mode recall analysis
    test_df = df.loc[indices_test].copy()
    test_df["y_pred"] = y_pred
    print("\n--- Per-Failure-Mode Performance ---")
    for mode in FAILURE_MODES:
        sub = test_df[(test_df["failure_mode"] == mode) & (test_df["label"] == 1)]
        if len(sub) > 0:
            mode_rec = (sub["y_pred"] == 1).mean()
            print(f"  Mode '{mode:15s}': Samples={len(sub):3d}, Recall={mode_rec:.4f}")
        else:
            print(f"  Mode '{mode:15s}': No positive test samples")

    # Sorted feature importances
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nFeature Importances:")
    for idx in sorted_idx:
        print(f"  {FEATURE_NAMES[idx]:15s}: {importances[idx]:.4f}")

    # Save artifacts into data/
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=['Healthy (0)', 'FailRisk (1)'], yticklabels=['Healthy (0)', 'FailRisk (1)'],
           ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    fig.tight_layout()
    fig.savefig(DATA_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 2. Feature Importance plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([FEATURE_NAMES[i] for i in sorted_idx][::-1], importances[sorted_idx][::-1], color='teal')
    ax.set_xlabel('Importance')
    ax.set_title('Random Forest Feature Importances')
    fig.tight_layout()
    fig.savefig(DATA_DIR / "feature_importances.png", dpi=150)
    plt.close(fig)

    # Save model and feature names
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": clf,
        "feature_names": FEATURE_NAMES
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"[ML Train] Saved trained model to {MODEL_PATH}\n")
    return clf, f1

if __name__ == "__main__":
    train()
