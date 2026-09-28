"""
Step 4 -- Forecaster: predict congestion a few time steps ahead per road.

Uses lag features + time features with a LightGBM regressor.
Predicts the ordinal congestion value (0/1/2) as a continuous target,
then rounds to get a directional forecast (improving / stable / worsening).

Usage:
    python -m src.forecaster
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

from src.utils import (
    CONGESTION_MAP_INV,
    MODELS_DIR,
    load_processed_data,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "Hour",
    "Day_of_Week",
    "Is_Weekend",
    "Weather_Severity",
    "Is_Roadwork",
    "Historical Incident Rate",
    "Congestion_Lag_1",
    "Congestion_Lag_2",
    "Congestion_Lag_3",
    "Rolling_Congestion_3",
    "Traffic Volume",
    "Road Capacity Utilization (%)",
]

TARGET_COL = "Congestion_Ordinal"  # predict 0/1/2 as continuous

MODEL_PATH = os.path.join(MODELS_DIR, "forecaster_model.joblib")


# ---------------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------------

def time_split(df: pd.DataFrame, split_frac: float = 0.8):
    """Time-based split."""
    df = df.sort_values("Date").reset_index(drop=True)
    split_idx = int(len(df) * split_frac)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_forecaster(train_df: pd.DataFrame) -> LGBMRegressor:
    """Train a LightGBM regressor for congestion forecasting."""
    available = [c for c in FEATURE_COLS if c in train_df.columns]

    X = train_df[available]
    y = train_df[TARGET_COL]

    model = LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def evaluate_forecaster(model: LGBMRegressor, test_df: pd.DataFrame) -> dict:
    """Evaluate the forecaster on the test set."""
    available = [c for c in FEATURE_COLS if c in test_df.columns]
    X = test_df[available]
    y = test_df[TARGET_COL]

    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    # Directional accuracy: does the rounded prediction match the true label?
    y_pred_rounded = np.clip(np.round(y_pred), 0, 2).astype(int)
    directional_acc = (y_pred_rounded == y.values).mean()

    print(f"  RMSE:               {rmse:.4f}")
    print(f"  MAE:                {mae:.4f}")
    print(f"  Directional Acc:    {directional_acc:.4f}")

    return {
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": directional_acc,
    }


# ---------------------------------------------------------------------------
# Predict helpers (used by dashboard)
# ---------------------------------------------------------------------------

def load_forecaster() -> LGBMRegressor:
    """Load the saved forecaster model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Forecaster model not found at {MODEL_PATH}. Run forecaster.py first."
        )
    return joblib.load(MODEL_PATH)


def forecast_congestion(
    model: LGBMRegressor,
    current_row: pd.DataFrame,
    steps_ahead: int = 3,
) -> list[dict]:
    """
    Given the current state of a road (1-row DataFrame), simulate forecasts
    for the next `steps_ahead` time periods.

    Returns a list of dicts:
        [{"step": 1, "predicted_level": "High", "raw_score": 1.87}, ...]
    """
    available = [c for c in FEATURE_COLS if c in current_row.columns]
    row = current_row[available].copy()

    forecasts = []
    prev_scores = [
        row.iloc[0].get("Congestion_Lag_1", 1),
        row.iloc[0].get("Congestion_Lag_2", 1),
        row.iloc[0].get("Congestion_Lag_3", 1),
    ]

    for step in range(1, steps_ahead + 1):
        pred_raw = float(model.predict(row)[0])
        pred_level = CONGESTION_MAP_INV.get(
            int(np.clip(np.round(pred_raw), 0, 2)), "Medium"
        )
        forecasts.append({
            "step": step,
            "predicted_level": pred_level,
            "raw_score": round(pred_raw, 3),
        })

        # Shift lag features for next step
        prev_scores.insert(0, pred_raw)
        if "Congestion_Lag_1" in row.columns:
            row["Congestion_Lag_1"] = prev_scores[0]
        if "Congestion_Lag_2" in row.columns:
            row["Congestion_Lag_2"] = prev_scores[1]
        if "Congestion_Lag_3" in row.columns:
            row["Congestion_Lag_3"] = prev_scores[2]
        if "Rolling_Congestion_3" in row.columns:
            row["Rolling_Congestion_3"] = np.mean(prev_scores[:3])

    return forecasts


def forecast_trend(forecasts: list[dict]) -> str:
    """
    Summarize forecast as a trend: Improving / Stable / Worsening.
    """
    if len(forecasts) < 2:
        return "Stable"
    first = forecasts[0]["raw_score"]
    last = forecasts[-1]["raw_score"]
    diff = last - first
    if diff > 0.3:
        return "Worsening"
    elif diff < -0.3:
        return "Improving"
    return "Stable"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("STEP 4 -- Forecaster Training")
    print("=" * 60)

    df = load_processed_data()

    print("\nSplitting data (time-based) ...")
    train_df, test_df = time_split(df)
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    print("\nTraining LightGBM forecaster ...")
    model = train_forecaster(train_df)

    print("\nEvaluating ...")
    metrics = evaluate_forecaster(model, test_df)

    print(f"\nSaving model to {MODEL_PATH} ...")
    joblib.dump(model, MODEL_PATH)
    print("Done OK")


if __name__ == "__main__":
    main()
