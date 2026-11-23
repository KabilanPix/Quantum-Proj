import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from typing import Dict, Any, List

def forecast_power(df: pd.DataFrame, steps_ahead: int = 30) -> List[float]:
    """
    Lightweight energy forecaster using lag features and Gradient Boosting.
    df: DataFrame with 'power' column.
    Returns: List of predicted power values for future steps (e.g., next 30 intervals).
    """
    if len(df) < 20 or "power" not in df.columns:
        # Default baseline
        last_val = df["power"].iloc[-1] if (len(df) > 0 and "power" in df.columns) else 3.5
        return [round(float(last_val), 2)] * steps_ahead

    series = df["power"].values
    # Build lag features: lag1, lag2, lag3, roll_mean5
    X, y = [], []
    for i in range(5, len(series)):
        lag1 = series[i - 1]
        lag2 = series[i - 2]
        lag3 = series[i - 3]
        mean5 = np.mean(series[i - 5 : i])
        X.append([lag1, lag2, lag3, mean5])
        y.append(series[i])

    if len(X) < 10:
        return [round(float(series[-1]), 2)] * steps_ahead

    X = np.array(X)
    y = np.array(y)

    model = GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=42)
    model.fit(X, y)

    # Autoregressive multi-step rollout
    history = list(series[-5:])
    preds = []
    for _ in range(steps_ahead):
        lag1 = history[-1]
        lag2 = history[-2]
        lag3 = history[-3]
        mean5 = np.mean(history[-5:])
        feat = np.array([[lag1, lag2, lag3, mean5]])
        next_p = float(model.predict(feat)[0])
        next_p = max(0.05, min(8.0, next_p))
        preds.append(round(next_p, 2))
        history.append(next_p)

    return preds

def get_forecasted_energy_by_machine(history_by_machine: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Returns average predicted power (kW) for each machine over next window,
    useful for quantum cost weights.
    """
    res = {}
    for m, df in history_by_machine.items():
        preds = forecast_power(df, steps_ahead=30)
        res[m] = round(float(np.mean(preds)), 2)
    return res
