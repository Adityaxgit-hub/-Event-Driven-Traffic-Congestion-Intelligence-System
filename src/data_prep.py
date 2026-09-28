"""
Step 1 & 2 -- Data cleaning, merging, and feature engineering.

Reads raw CSVs from data/raw/, cleans them, merges BTP incident rates,
engineers features, and writes the result to data/processed/traffic_features.csv.

Usage:
    python -m src.data_prep
"""

import os

import numpy as np
import pandas as pd

from src.utils import (
    DATA_RAW,
    DATA_PROCESSED,
    CONGESTION_MAP,
    load_coordinates,
)

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------

def load_traffic_data() -> pd.DataFrame:
    """Load the Bangalore traffic CSV."""
    path = os.path.join(DATA_RAW, "bangalore_traffic.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Traffic data not found at {path}.\n"
            "Either place the Kaggle CSV there or run:  python -m src.generate_sample_data"
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"Loaded traffic data: {df.shape}")
    return df


def load_btp_data() -> pd.DataFrame:
    """Load the BTP accident CSV."""
    path = os.path.join(DATA_RAW, "btp_accidents.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BTP data not found at {path}.\n"
            "Either place the OpenCity CSV there or run:  python -m src.generate_sample_data"
        )
    df = pd.read_csv(path)
    print(f"Loaded BTP data: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 2. Clean traffic data
# ---------------------------------------------------------------------------

def clean_traffic(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, standardize names, parse dates."""
    df = df.copy()

    # Standardize column names (in case of minor variations)
    col_map = {}
    for col in df.columns:
        clean = col.strip()
        col_map[col] = clean
    df.rename(columns=col_map, inplace=True)

    # Drop rows where critical fields are missing
    critical = ["Date", "Area Name", "Road/Intersection Name", "Congestion Level"]
    before = len(df)
    df.dropna(subset=[c for c in critical if c in df.columns], inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with missing critical fields")

    # Fill numeric NaNs with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Filled {col} NaNs with median ({median_val:.2f})")

    # Standardize area/road names (strip whitespace, title case)
    for col in ["Area Name", "Road/Intersection Name"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()

    # Standardize weather
    if "Weather Conditions" in df.columns:
        df["Weather Conditions"] = df["Weather Conditions"].str.strip().str.title()

    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"])

    print(f"  Cleaned traffic data: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 3. Compute BTP incident rate per area
# ---------------------------------------------------------------------------

def compute_incident_rate(btp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate BTP data into a historical incident rate per area.
    Returns a DataFrame with columns: [Area Name, Historical Incident Rate]
    """
    btp = btp_df.copy()

    # Standardize the area column name
    area_col = None
    for c in btp.columns:
        if "station" in c.lower() or "area" in c.lower():
            area_col = c
            break
    if area_col is None:
        raise ValueError("Cannot find area/station column in BTP data")

    btp.rename(columns={area_col: "Area Name"}, inplace=True)
    btp["Area Name"] = btp["Area Name"].str.strip().str.title()

    # Average annual accidents per area
    acc_col = None
    for c in btp.columns:
        if "total" in c.lower() and "accident" in c.lower():
            acc_col = c
            break
    if acc_col is None:
        acc_col = "Total Accidents"

    rate = (
        btp.groupby("Area Name")[acc_col]
        .mean()
        .reset_index()
        .rename(columns={acc_col: "Historical Incident Rate"})
    )
    rate["Historical Incident Rate"] = rate["Historical Incident Rate"].round(2)
    print(f"  Computed incident rates for {len(rate)} areas")
    return rate


# ---------------------------------------------------------------------------
# 4. Merge & engineer features
# ---------------------------------------------------------------------------

def merge_and_engineer(traffic_df: pd.DataFrame, incident_rate: pd.DataFrame) -> pd.DataFrame:
    """
    Merge incident rate into traffic data, then create features.
    """
    df = traffic_df.copy()

    # --- Merge BTP incident rate (left join on area) ---
    df = df.merge(incident_rate, on="Area Name", how="left")
    df["Historical Incident Rate"].fillna(0, inplace=True)

    # --- Time features ---
    df["Hour"] = df["Date"].dt.hour if "Hour" not in df.columns else df["Hour"]
    df["Day_of_Week"] = df["Date"].dt.dayofweek  # 0=Mon, 6=Sun
    df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)
    df["Month"] = df["Date"].dt.month

    # --- Congestion level as ordinal ---
    df["Congestion_Ordinal"] = df["Congestion Level"].map(CONGESTION_MAP)
    # Drop any rows where mapping failed
    df.dropna(subset=["Congestion_Ordinal"], inplace=True)
    df["Congestion_Ordinal"] = df["Congestion_Ordinal"].astype(int)

    # --- Rolling congestion trend (3-record rolling mean per road) ---
    df = df.sort_values(["Road/Intersection Name", "Date"]).reset_index(drop=True)
    df["Rolling_Congestion_3"] = (
        df.groupby("Road/Intersection Name")["Congestion_Ordinal"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
        .round(3)
    )

    # --- Lag features for forecaster ---
    for lag in [1, 2, 3]:
        df[f"Congestion_Lag_{lag}"] = (
            df.groupby("Road/Intersection Name")["Congestion_Ordinal"]
            .shift(lag)
        )
    # Fill lag NaNs with the current value (first records)
    for lag in [1, 2, 3]:
        df[f"Congestion_Lag_{lag}"].fillna(df["Congestion_Ordinal"], inplace=True)

    # --- Interaction features ---
    # Weather severity encoding
    weather_severity = {"Clear": 0, "Fog": 1, "Rain": 2, "Heavy Rain": 3}
    df["Weather_Severity"] = df["Weather Conditions"].map(weather_severity).fillna(0).astype(int)

    # Weather x Hour interaction
    df["Weather_Hour"] = df["Weather_Severity"] * df["Hour"]

    # Roadwork x Traffic Volume interaction
    if "Is_Roadwork" in df.columns:
        df["Roadwork_Volume"] = df["Is_Roadwork"] * df.get("Traffic Volume", 0)

    # --- Add coordinates ---
    coords = load_coordinates()
    df["Latitude"] = df["Road/Intersection Name"].map(
        lambda r: coords["roads"].get(r, {}).get("lat", np.nan)
    )
    df["Longitude"] = df["Road/Intersection Name"].map(
        lambda r: coords["roads"].get(r, {}).get("lon", np.nan)
    )

    print(f"  Feature-engineered dataset: {df.shape}")
    print(f"  Congestion distribution:\n{df['Congestion Level'].value_counts().to_string()}")
    return df


# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------

def save_processed(df: pd.DataFrame) -> str:
    """Save the processed DataFrame."""
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    path = os.path.join(DATA_PROCESSED, "traffic_features.csv")
    df.to_csv(path, index=False)
    print(f"  Saved processed data to {path}")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("STEP 1 -- Data Cleaning & Preparation")
    print("=" * 60)

    traffic_df = load_traffic_data()
    btp_df = load_btp_data()

    print("\nCleaning traffic data ...")
    traffic_clean = clean_traffic(traffic_df)

    print("\nComputing BTP incident rates ...")
    incident_rate = compute_incident_rate(btp_df)

    print("\nSTEP 2 -- Feature Engineering")
    print("-" * 40)
    features_df = merge_and_engineer(traffic_clean, incident_rate)

    print("\nSaving ...")
    save_processed(features_df)
    print("\nDone OK")


if __name__ == "__main__":
    main()
