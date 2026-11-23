import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

def load_and_preprocess(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Real dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Clean column names
    col_map = {
        "Air temperature [K]": "air_temp",
        "Process temperature [K]": "process_temp",
        "Rotational speed [rpm]": "rpm",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear",
        "Machine failure": "target"
    }
    df = df.rename(columns=col_map)

    # Feature Engineering
    # 1. Temperature difference (Process temp - Air temp)
    df["temp_diff"] = df["process_temp"] - df["air_temp"]
    # 2. Power = Torque * (rpm in rad/s) = torque * (rpm * 2 * pi / 60) in Watts
    df["power_watts"] = df["torque"] * (df["rpm"] * 2.0 * np.pi / 60.0)
    # 3. Tool wear rate proxy (wear * torque)
    df["tool_wear_torque"] = df["tool_wear"] * df["torque"]
    # 4. Heat dissipation index
    df["heat_dissipation"] = df["temp_diff"] * (df["rpm"] / 1000.0)

    # Ordinal encode Type (L=0, M=1, H=2)
    type_map = {"L": 0, "M": 1, "H": 2}
    df["type_code"] = df["Type"].map(type_map).fillna(0)

    features = [
        "air_temp", "process_temp", "rpm", "torque", "tool_wear",
        "temp_diff", "power_watts", "tool_wear_torque", "heat_dissipation", "type_code"
    ]
    target = "target"

    X = df[features]
    y = df[target]

    return X, y, df

def train_and_evaluate():
    print("=" * 65)
    print("      TRAINING ML MODELS ON REAL UCI AI4I 2020 DATASET        ")
    print("=" * 65)

    if not config.REAL_DATA_PATH.exists():
        print(f"[Warning] Real dataset not found at {config.REAL_DATA_PATH}. Skipping.")
        return None

    X, y, df = load_and_preprocess(config.REAL_DATA_PATH)
    print(f"Dataset shape: {X.shape}, Class 0: {(y==0).sum()}, Class 1 (Failures): {(y==1).sum()} ({(y==1).mean()*100:.2f}%)")
    print("Note: Extreme class imbalance (~3.4% failure). Recall, F1, and PR-AUC are key metrics.")

    # Stratified split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
    }

    results = {}
    best_f1 = -1.0
    best_model_name = ""
    best_model = None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\nEvaluating {name} with 5-fold Stratified CV...")
        cv_scores = cross_validate(
            model, X_train, y_train, cv=skf,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            n_jobs=-1
        )
        print(f"  CV Recall:    {cv_scores['test_recall'].mean():.4f} +/- {cv_scores['test_recall'].std():.4f}")
        print(f"  CV F1:        {cv_scores['test_f1'].mean():.4f} +/- {cv_scores['test_f1'].std():.4f}")
        print(f"  CV ROC-AUC:   {cv_scores['test_roc_auc'].mean():.4f} +/- {cv_scores['test_roc_auc'].std():.4f}")

        # Train on full train set & test on holdout
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "confusion_matrix": cm,
            "cv_f1_mean": round(float(cv_scores['test_f1'].mean()), 4),
            "cv_recall_mean": round(float(cv_scores['test_recall'].mean()), 4)
        }

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    print(f"\n--> Best Model: {best_model_name} (F1 = {best_f1:.4f})")

    # Feature importances for tree models
    feature_importances = {}
    if hasattr(best_model, "feature_importances_"):
        for fname, imp in zip(X.columns, best_model.feature_importances_):
            feature_importances[fname] = round(float(imp), 4)

    # Save best model
    joblib.dump({
        "model": best_model,
        "model_name": best_model_name,
        "feature_names": list(X.columns),
        "metrics": results[best_model_name]
    }, config.MODEL_REAL_PATH)
    print(f"[Save] Saved real data model to {config.MODEL_REAL_PATH}")

    # Save metrics JSON
    metrics_payload = {
        "dataset": "UCI AI4I 2020 Predictive Maintenance",
        "samples": len(df),
        "failure_count": int((y == 1).sum()),
        "failure_rate": round(float((y == 1).mean()), 4),
        "best_model": best_model_name,
        "models": results,
        "feature_importances": feature_importances,
        "feature_names": list(X.columns)
    }
    with open(config.REAL_METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[Save] Saved metrics to {config.REAL_METRICS_PATH}")

    # Generate Plots
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix
    cm_best = np.array(results[best_model_name]["confusion_matrix"])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm_best, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {best_model_name}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Normal", "Failure"])
    plt.yticks(tick_marks, ["Normal", "Failure"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm_best[i, j]), horizontalalignment="center",
                     color="white" if cm_best[i, j] > cm_best.max() / 2 else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "real_confusion_matrix.png", dpi=150)
    plt.close()

    # 2. ROC Curves
    plt.figure(figsize=(6, 5))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random (0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curves - Real AI4I 2020 Dataset")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "real_roc_curve.png", dpi=150)
    plt.close()

    # 3. Precision-Recall Curves
    plt.figure(figsize=(6, 5))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
        prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(rec_c, prec_c, label=f"{name} (PR-AUC = {results[name]['pr_auc']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - Real AI4I 2020 Dataset")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "real_pr_curve.png", dpi=150)
    plt.close()

    # 4. Feature Importances
    if feature_importances:
        plt.figure(figsize=(7, 4.5))
        sorted_fi = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_fi]
        vals = [x[1] for x in sorted_fi]
        plt.barh(names[::-1], vals[::-1], color="#1f77b4")
        plt.xlabel("Importance")
        plt.title(f"Feature Importances ({best_model_name})")
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR / "real_feature_importance.png", dpi=150)
        plt.close()

    print(f"[Plots] Saved evaluation plots to {config.PLOTS_DIR}")
    return results

if __name__ == "__main__":
    train_and_evaluate()
