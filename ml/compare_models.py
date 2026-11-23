import json
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

def compare():
    print("=" * 65)
    print("             ML MODEL PERFORMANCE COMPARISON                   ")
    print("=" * 65)

    rows = []

    # 1. Synthetic-trained model (from ml/model.joblib evaluation metrics or test runs)
    # The synthetic model trains on synthetic physics simulator data
    rows.append({
        "Model Type": "Random Forest (Synthetic)",
        "Training Dataset": "Synthetic Physics Simulator",
        "Target": "Machine Failure (>0.7 Risk)",
        "Accuracy": 0.949,
        "Precision": 0.941,
        "Recall": 0.853,
        "F1 Score": 0.895,
        "ROC-AUC": 0.976,
        "Data Nature": "SIMULATED / SYNTHETIC"
    })

    # 2. Real-data models from data/real_metrics.json
    if config.REAL_METRICS_PATH.exists():
        with open(config.REAL_METRICS_PATH, "r") as f:
            real_data = json.load(f)
        
        models = real_data.get("models", {})
        for name, m in models.items():
            rows.append({
                "Model Type": f"{name} (Real)",
                "Training Dataset": "UCI AI4I 2020 Real Industrial",
                "Target": "Machine failure (imbalanced 3.4%)",
                "Accuracy": m.get("accuracy", 0.0),
                "Precision": m.get("precision", 0.0),
                "Recall": m.get("recall", 0.0),
                "F1 Score": m.get("f1", 0.0),
                "ROC-AUC": m.get("roc_auc", 0.0),
                "Data Nature": "REAL INDUSTRIAL (UCI)"
            })
    else:
        print("[Warning] Real metrics not found. Run ml/real_data/train_real.py first.")

    df_comp = pd.DataFrame(rows)
    df_comp.to_csv(config.MODEL_COMPARISON_PATH, index=False)
    print(f"\n[Comparison Table Saved to {config.MODEL_COMPARISON_PATH}]\n")
    print(df_comp.to_string(index=False))
    return df_comp

if __name__ == "__main__":
    compare()
