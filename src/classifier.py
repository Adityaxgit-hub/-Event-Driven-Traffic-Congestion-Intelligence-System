"""
Step 3 -- CatBoost classifier for congestion level prediction.

Trains on the processed feature set, evaluates with time-based split,
and saves the trained model.

Usage:
    python -m src.classifier
"""

import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.utils import (
    CONGESTION_LEVELS,
    MODELS_DIR,
    load_processed_data,
)

# ---------------------------------------------------------------------------
# Feature / target configuration
# ---------------------------------------------------------------------------

# Features used by the classifier
FEATURE_COLS = [
    "Hour",
    "Day_of_Week",
    "Is_Weekend",
    "Month",
    "Traffic Volume",
    "Average Speed (km/h)",
    "Travel Time Index",
    "Road Capacity Utilization (%)",
    "Pedestrian & Cyclist Count",
    "Incident Reports",
    "Is_Roadwork",
    "Historical Incident Rate",
    "Rolling_Congestion_3",
    "Weather_Severity",
    "Weather_Hour",
    "Roadwork_Volume",
]

# Categorical features for CatBoost (indices into FEATURE_COLS)
CAT_FEATURE_NAMES = []  # all features are numeric after encoding

TARGET_COL = "Congestion Level"

MODEL_PATH = os.path.join(MODELS_DIR, "catboost_model.joblib")


# ---------------------------------------------------------------------------
# Time-based train/test split
# ---------------------------------------------------------------------------

def time_split(df: pd.DataFrame, split_frac: float = 0.8):
    """Split data by date: earlier records for training, later for test."""
    df = df.sort_values("Date").reset_index(drop=True)
    split_idx = int(len(df) * split_frac)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    print(f"  Train: {len(train)} rows  |  Test: {len(test)} rows")
    print(f"  Train dates: {train['Date'].min().date()} -> {train['Date'].max().date()}")
    print(f"  Test dates:  {test['Date'].min().date()} -> {test['Date'].max().date()}")
    return train, test


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_classifier(train_df: pd.DataFrame) -> CatBoostClassifier:
    """Train a CatBoost classifier with light hyperparameter tuning."""
    # Only keep features that exist in the dataframe
    available_features = [c for c in FEATURE_COLS if c in train_df.columns]

    X_train = train_df[available_features].copy()
    y_train = train_df[TARGET_COL]

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.08,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        class_names=CONGESTION_LEVELS,
    )

    model.fit(X_train, y_train, verbose=100)
    return model


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate_classifier(
    model: CatBoostClassifier,
    test_df: pd.DataFrame,
) -> dict:
    """Evaluate the model on the test set. Returns metrics dict."""
    available_features = [c for c in FEATURE_COLS if c in test_df.columns]
    X_test = test_df[available_features]
    y_test = test_df[TARGET_COL]

    y_pred = model.predict(X_test).flatten()

    macro_f1 = f1_score(y_test, y_pred, average="macro", labels=CONGESTION_LEVELS)
    report = classification_report(y_test, y_pred, labels=CONGESTION_LEVELS)
    cm = confusion_matrix(y_test, y_pred, labels=CONGESTION_LEVELS)

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(report)
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"  Labels: {CONGESTION_LEVELS}")
    print(cm)

    # Feature importance
    importances = model.get_feature_importance()
    feat_names = model.feature_names_
    importance_df = (
        pd.DataFrame({"Feature": feat_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
    )
    print(f"\nTop 10 Features:")
    print(importance_df.head(10).to_string(index=False))

    return {
        "macro_f1": macro_f1,
        "report": report,
        "confusion_matrix": cm,
        "feature_importance": importance_df,
    }


# ---------------------------------------------------------------------------
# Predict helper (used by dashboard)
# ---------------------------------------------------------------------------

def load_classifier() -> CatBoostClassifier:
    """Load the saved classifier model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Classifier model not found at {MODEL_PATH}. Run classifier.py first."
        )
    return joblib.load(MODEL_PATH)


def predict_congestion(
    model: CatBoostClassifier,
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict congestion level and probabilities for input rows.
    Returns the input_df with added columns:
        Predicted_Congestion, Prob_Low, Prob_Medium, Prob_High
    """
    available_features = [c for c in FEATURE_COLS if c in input_df.columns]
    X = input_df[available_features]

    preds = model.predict(X).flatten()
    probs = model.predict_proba(X)

    result = input_df.copy()
    result["Predicted_Congestion"] = preds
    for i, level in enumerate(CONGESTION_LEVELS):
        result[f"Prob_{level}"] = probs[:, i].round(4)

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("STEP 3 -- CatBoost Classifier Training")
    print("=" * 60)

    df = load_processed_data()

    print("\nSplitting data (time-based) ...")
    train_df, test_df = time_split(df)

    print("\nTraining CatBoost ...")
    model = train_classifier(train_df)

    print("\nEvaluating ...")
    metrics = evaluate_classifier(model, test_df)

    print(f"\nSaving model to {MODEL_PATH} ...")
    joblib.dump(model, MODEL_PATH)
    print("Done OK")


if __name__ == "__main__":
    main()
