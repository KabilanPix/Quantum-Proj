import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, Optional

class AnomalyDetector:
    """
    Lightweight unsupervised anomaly detection per machine using Isolation Forest.
    Trained on recent normal sensor observations.
    """
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.models: Dict[str, IsolationForest] = {}

    def fit_machine(self, machine_id: str, df: pd.DataFrame) -> None:
        """Trains or updates the Isolation Forest model for a machine."""
        if len(df) < 15:
            return
        features = df[["temperature", "vibration", "power"]].values
        clf = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=50
        )
        clf.fit(features)
        self.models[machine_id] = clf

    def score(self, machine_id: str, current_reading: Dict[str, float]) -> float:
        """
        Returns anomaly score between 0.0 (normal) and 1.0 (highly anomalous).
        Uses normalized decision function.
        """
        clf = self.models.get(machine_id)
        temp = float(current_reading.get("temperature", 60.0))
        vib = float(current_reading.get("vibration", 0.35))
        pow_val = float(current_reading.get("power", 3.5))
        sample = np.array([[temp, vib, pow_val]])

        if clf is None:
            # Baseline rule-based score if untrained
            t_score = max(0.0, (temp - 65.0) / 30.0)
            v_score = max(0.0, (vib - 0.5) / 1.5)
            p_score = max(0.0, (pow_val - 4.0) / 3.0)
            return round(min(1.0, max(t_score, v_score, p_score)), 3)

        # decision_function gives negative for anomalies, positive for inliers
        raw_score = clf.decision_function(sample)[0]
        # Map raw_score (~[-0.3, 0.2]) to [0, 1]
        norm_score = 1.0 - (raw_score + 0.3) / 0.5
        return round(float(np.clip(norm_score, 0.0, 1.0)), 3)

# Global singleton
detector = AnomalyDetector()
