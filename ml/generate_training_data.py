import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TRAINING_DATA_PATH, WINDOW_SIZE
from ml.features import extract_features, FEATURE_NAMES

FAILURE_MODES = ["bearing_wear", "overheating", "power_surge"]

def generate_cycle(mode: str, will_fail: bool):
    """Generates a single machine lifecycle under a specific failure mode."""
    healthy_len = np.random.randint(60, 130)
    base_temp = np.random.uniform(57.0, 63.0)
    base_vib = np.random.uniform(0.28, 0.40)
    base_pow = np.random.uniform(3.2, 3.8)

    temps = list(np.random.normal(base_temp, 0.8, healthy_len))
    vibs = list(np.random.normal(base_vib, 0.03, healthy_len))
    pows = list(np.random.normal(base_pow, 0.1, healthy_len))

    if will_fail:
        degrade_len = np.random.randint(50, 110)
        for step in range(degrade_len):
            frac = (step + 1) / degrade_len
            if mode == "bearing_wear":
                # Exponential vibration, slow temperature rise
                cur_v = base_vib + 1.8 * (frac ** 1.8) + np.random.normal(0, 0.04)
                cur_t = base_temp + 26.0 * (frac ** 1.1) + np.random.normal(0, 1.0)
                cur_p = base_pow + 0.9 * frac + np.random.normal(0, 0.1)
            elif mode == "overheating":
                # Temperature climbs quickly, vibration stays normal
                cur_t = base_temp + 38.0 * (frac ** 0.8) + np.random.normal(0, 1.0)
                cur_v = base_vib + 0.1 * frac + np.random.normal(0, 0.03)
                cur_p = base_pow + 0.6 * frac + np.random.normal(0, 0.1)
            elif mode == "power_surge":
                # Power spikes, then temperature rises
                spike = np.random.uniform(1.5, 3.0) if np.random.random() < 0.25 else 0.0
                cur_p = base_pow + 1.2 * frac + spike + np.random.normal(0, 0.15)
                cur_t = base_temp + 30.0 * (frac ** 1.2) + 0.8 * spike + np.random.normal(0, 1.0)
                cur_v = base_vib + 0.5 * frac + np.random.normal(0, 0.05)
            else:
                cur_t = base_temp + 32.0 * frac + np.random.normal(0, 1.0)
                cur_v = base_vib + 1.6 * frac + np.random.normal(0, 0.05)
                cur_p = base_pow + 0.8 * frac + np.random.normal(0, 0.1)

            temps.append(cur_t)
            vibs.append(cur_v)
            pows.append(cur_p)

            if cur_t >= 85.0 or cur_v >= 1.5:
                # Hold critical
                for _ in range(np.random.randint(5, 12)):
                    temps.append(cur_t + np.random.normal(0, 1.2))
                    vibs.append(cur_v + np.random.normal(0, 0.05))
                    pows.append(cur_p + np.random.normal(0, 0.1))
                break
    else:
        extra_len = np.random.randint(40, 70)
        temps.extend(list(np.random.normal(base_temp, 0.8, extra_len)))
        vibs.extend(list(np.random.normal(base_vib, 0.03, extra_len)))
        pows.extend(list(np.random.normal(base_pow, 0.1, extra_len)))

    return pd.DataFrame({"temperature": temps, "vibration": vibs, "power": pows})

def generate_synthetic_data(num_cycles_per_mode: int = 70) -> pd.DataFrame:
    """
    Generates multi-mode synthetic training dataset across:
    - bearing_wear
    - overheating
    - power_surge
    Reports per-failure-mode samples and stores in data/training_data.csv.
    """
    np.random.seed(42)
    print(f"[ML DataGen] Generating dataset across {FAILURE_MODES}...")
    all_rows = []

    for mode in FAILURE_MODES:
        for c in range(num_cycles_per_mode):
            will_fail = (c % 2 == 0)
            cycle_df = generate_cycle(mode, will_fail)

            is_failure = (cycle_df["temperature"] > 85.0) | (cycle_df["vibration"] > 1.5)
            failure_indices = np.where(is_failure.values)[0]
            first_failure_idx = failure_indices[0] if len(failure_indices) > 0 else None

            total_len = len(cycle_df)
            for end_idx in range(WINDOW_SIZE, total_len, 4):
                window_slice = cycle_df.iloc[end_idx - WINDOW_SIZE : end_idx]
                feats = extract_features(window_slice)
                if feats is None:
                    continue

                if first_failure_idx is not None:
                    time_to_fail = first_failure_idx - end_idx
                    label = 1 if time_to_fail <= 60 else 0
                else:
                    label = 0

                row_dict = feats.iloc[0].to_dict()
                row_dict["label"] = label
                row_dict["failure_mode"] = mode if will_fail else "healthy"
                all_rows.append(row_dict)

    dataset = pd.DataFrame(all_rows)
    pos_count = (dataset["label"] == 1).sum()
    neg_count = (dataset["label"] == 0).sum()
    print(f"[ML DataGen] Generated {len(dataset)} samples. Positives: {pos_count}, Negatives: {neg_count}")

    TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(TRAINING_DATA_PATH, index=False)
    print(f"[ML DataGen] Saved multi-mode dataset to {TRAINING_DATA_PATH}")
    return dataset

if __name__ == "__main__":
    generate_synthetic_data()
