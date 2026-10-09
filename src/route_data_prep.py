"""
Route Data Preparation -- Cleans, engineers features, and derives congestion
labels from the bangalore_routes.csv dataset.

This module handles the route-based dataset (source → destination pairs)
which is structurally different from the intersection-based bangalore_traffic.csv.

It produces a processed output compatible with the existing classifier/forecaster
pipeline by:
  1. Cleaning outlier coordinates, distances, and travel times
  2. Deriving congestion labels from speed/volume/capacity signals
  3. Engineering rich features (volume-capacity ratio, speed efficiency, etc.)
  4. Mapping route locations to area names
  5. Aggregating route-level data to road/area level for pipeline compatibility

Usage:
    python -m src.route_data_prep
"""

import os
import re
import json

import numpy as np
import pandas as pd

from src.utils import (
    DATA_RAW,
    DATA_PROCESSED,
    COORDINATES_PATH,
    CONGESTION_MAP,
    load_coordinates,
)


# ---------------------------------------------------------------------------
# Bangalore bounding box (generous)
# ---------------------------------------------------------------------------
BLR_LAT_MIN, BLR_LAT_MAX = 12.70, 13.25
BLR_LON_MIN, BLR_LON_MAX = 77.30, 77.90

# Distance / travel-time sanity caps
MAX_DISTANCE_KM = 50.0
MAX_TRAVEL_TIME_MIN = 300.0

# Area keyword mapping -- maps substrings in location names to area names
# Order matters: more specific patterns first
AREA_KEYWORDS = [
    ("electronic city", "Electronic City"),
    ("e-city", "Electronic City"),
    ("silk board", "Electronic City"),
    ("hosur road", "Electronic City"),
    ("bommanahalli", "Electronic City"),
    ("hebbal", "Hebbal"),
    ("ballari", "Hebbal"),
    ("bellary", "Hebbal"),
    ("yelahanka", "Hebbal"),
    ("jakkur", "Hebbal"),
    ("indiranagar", "Indiranagar"),
    ("indira nagar", "Indiranagar"),
    ("100 feet road", "Indiranagar"),
    ("cmh road", "Indiranagar"),
    ("domlur", "Indiranagar"),
    ("jayanagar", "Jayanagar"),
    ("jaya nagar", "Jayanagar"),
    ("south end", "Jayanagar"),
    ("basavanagudi", "Jayanagar"),
    ("vijayanagar", "Jayanagar"),
    ("koramangala", "Koramangala"),
    ("kormangala", "Koramangala"),
    ("sony world", "Koramangala"),
    ("sarjapur", "Koramangala"),
    ("hsr layout", "Koramangala"),
    ("hsr ", "Koramangala"),
    ("m.g. road", "M.G. Road"),
    ("mg road", "M.G. Road"),
    ("mahatma gandhi", "M.G. Road"),
    ("trinity", "M.G. Road"),
    ("anil kumble", "M.G. Road"),
    ("brigade road", "M.G. Road"),
    ("cubbon", "M.G. Road"),
    ("whitefield", "Whitefield"),
    ("white field", "Whitefield"),
    ("itpl", "Whitefield"),
    ("marathahalli", "Whitefield"),
    ("varthur", "Whitefield"),
    ("brookefield", "Whitefield"),
    ("yeshwanthpur", "Yeshwanthpur"),
    ("yeshwantpur", "Yeshwanthpur"),
    ("tumkur", "Yeshwanthpur"),
    ("peenya", "Yeshwanthpur"),
    ("rajajinagar", "Yeshwanthpur"),
    ("malleshwaram", "Yeshwanthpur"),
    ("malleswaram", "Yeshwanthpur"),
    # Additional areas for broader coverage
    ("kr puram", "KR Puram"),
    ("k r puram", "KR Puram"),
    ("mahadevapura", "KR Puram"),
    ("tin factory", "KR Puram"),
    ("btm", "BTM Layout"),
    ("btm layout", "BTM Layout"),
    ("jp nagar", "JP Nagar"),
    ("j p nagar", "JP Nagar"),
    ("bannerghatta", "JP Nagar"),
    ("bannerghata", "JP Nagar"),
    ("rr nagar", "RR Nagar"),
    ("rajarajeshwari", "RR Nagar"),
    ("nagarbhavi", "RR Nagar"),
    ("kengeri", "RR Nagar"),
    ("mysore road", "RR Nagar"),
    ("mysuru road", "RR Nagar"),
    ("majestic", "Majestic"),
    ("kempegowda", "Majestic"),
    ("city market", "Majestic"),
    ("city railway", "Majestic"),
    ("bangalore fort", "Majestic"),
    ("shivajinagar", "Majestic"),
    ("frazer town", "Majestic"),
    ("ulsoor", "Majestic"),
    ("yelachenahalli", "JP Nagar"),
    ("banaswadi", "KR Puram"),
    ("kalyan nagar", "KR Puram"),
    ("kammanahalli", "KR Puram"),
    ("hennur", "Hebbal"),
    ("thanisandra", "Hebbal"),
    ("nagawara", "Hebbal"),
    ("bagalakunte", "Yeshwanthpur"),
    ("dasarahalli", "Yeshwanthpur"),
    ("goraguntepalya", "Yeshwanthpur"),
    ("gorguntepalya", "Yeshwanthpur"),
    ("chamarajpet", "Majestic"),
    ("chamrajpet", "Majestic"),
    ("chickpet", "Majestic"),
    ("lalbagh", "Jayanagar"),
    ("wilson garden", "Jayanagar"),
    ("old airport road", "Indiranagar"),
    ("airport road", "Indiranagar"),
    ("bellandur", "Koramangala"),
    ("devarabisanahalli", "Whitefield"),
    ("kadugodi", "Whitefield"),
    ("hoodi", "Whitefield"),
    ("bettahalli", "Hebbal"),
    ("abbigere", "Yeshwanthpur"),
    ("vidyaranyapura", "Hebbal"),
    ("sahakarnagar", "Hebbal"),
    ("rt nagar", "Hebbal"),
    ("sadashivanagar", "Hebbal"),
    ("sankey", "Hebbal"),
]


# ---------------------------------------------------------------------------
# 1. Load route data
# ---------------------------------------------------------------------------

def load_route_data() -> pd.DataFrame:
    """Load the bangalore_routes.csv file."""
    path = os.path.join(DATA_RAW, "bangalore_routes.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Route data not found at {path}.\n"
            "Place bangalore_routes.csv in data/raw/"
        )
    print(f"Loading route data from {path} ...")
    df = pd.read_csv(path)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# 2. Clean route data
# ---------------------------------------------------------------------------

def clean_routes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the route dataset:
    - Remove rows with coordinates outside Bangalore bbox
    - Remove extreme distance/travel_time outliers
    - Harmonize weather categories
    - Standardize column types
    """
    df = df.copy()
    initial_len = len(df)
    print(f"\n--- Cleaning route data ({initial_len:,} rows) ---")

    # 1. Filter coordinates to Bangalore bounding box
    coord_mask = (
        (df["lat_src"] >= BLR_LAT_MIN) & (df["lat_src"] <= BLR_LAT_MAX) &
        (df["lon_src"] >= BLR_LON_MIN) & (df["lon_src"] <= BLR_LON_MAX) &
        (df["lat_dest"] >= BLR_LAT_MIN) & (df["lat_dest"] <= BLR_LAT_MAX) &
        (df["lon_dest"] >= BLR_LON_MIN) & (df["lon_dest"] <= BLR_LON_MAX)
    )
    bad_coords = (~coord_mask).sum()
    df = df[coord_mask].copy()
    print(f"  Removed {bad_coords:,} rows with coordinates outside Bangalore bbox")

    # 2. Remove extreme distance outliers
    dist_mask = (df["distance"] > 0) & (df["distance"] <= MAX_DISTANCE_KM)
    bad_dist = (~dist_mask).sum()
    df = df[dist_mask].copy()
    print(f"  Removed {bad_dist:,} rows with distance > {MAX_DISTANCE_KM} km or <= 0")

    # 3. Remove extreme travel_time outliers
    tt_mask = (df["travel_time"] > 0) & (df["travel_time"] <= MAX_TRAVEL_TIME_MIN)
    bad_tt = (~tt_mask).sum()
    df = df[tt_mask].copy()
    print(f"  Removed {bad_tt:,} rows with travel_time > {MAX_TRAVEL_TIME_MIN} min or <= 0")

    # 4. Harmonize weather categories to match existing pipeline
    weather_map = {
        "Clear": "Clear",
        "Cloudy": "Fog",        # closest mapping -- overcast/cloudy ≈ reduced visibility
        "Rainy": "Rain",
        "Foggy": "Fog",
    }
    df["weather"] = df["weather"].str.strip().str.title().map(weather_map).fillna("Clear")
    print(f"  Harmonized weather categories: {df['weather'].value_counts().to_dict()}")

    # 5. Standardize day_of_week
    df["day_of_week"] = df["day_of_week"].str.strip().str.title()

    # 6. Clamp speed to reasonable range
    df["speed"] = df["speed"].clip(lower=3.0, upper=80.0)

    # 7. Strip location names
    df["source_location"] = df["source_location"].str.strip()
    df["destination_location"] = df["destination_location"].str.strip()

    final_len = len(df)
    print(f"  Final clean dataset: {final_len:,} rows ({initial_len - final_len:,} removed, "
          f"{(initial_len - final_len) / initial_len * 100:.1f}% drop)")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Map locations to areas
# ---------------------------------------------------------------------------

def _extract_area(location_name: str) -> str:
    """Map a location name to an area using keyword matching."""
    loc_lower = location_name.lower()
    for keyword, area in AREA_KEYWORDS:
        if keyword in loc_lower:
            return area
    return "Other"


def map_locations_to_areas(df: pd.DataFrame) -> pd.DataFrame:
    """Add Source_Area and Dest_Area columns based on location name keywords."""
    df = df.copy()

    df["Source_Area"] = df["source_location"].apply(_extract_area)
    df["Dest_Area"] = df["destination_location"].apply(_extract_area)

    # Stats
    src_areas = df["Source_Area"].value_counts()
    print(f"\n--- Area mapping stats (source) ---")
    for area, count in src_areas.items():
        print(f"  {area}: {count:,} ({count / len(df) * 100:.1f}%)")

    unmapped = (df["Source_Area"] == "Other").sum()
    print(f"  Unmapped sources: {unmapped:,} ({unmapped / len(df) * 100:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# 4. Derive congestion labels
# ---------------------------------------------------------------------------

def derive_congestion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive congestion labels using volume-capacity ratio, speed, and peak hour.

    congestion_score = 0.50 * volume_capacity_ratio
                     + 0.35 * (1 - speed/65)    # normalized inverse speed
                     + 0.15 * is_peak

    Low:    score < 0.40
    Medium: 0.40 ≤ score < 0.65
    High:   score ≥ 0.65
    """
    df = df.copy()

    df["Volume_Capacity_Ratio"] = (df["vehicles"] / df["road_capacity"]).clip(0, 1.5)
    speed_factor = (1 - df["speed"] / 65.0).clip(0, 1)

    congestion_score = (
        0.50 * df["Volume_Capacity_Ratio"] +
        0.35 * speed_factor +
        0.15 * df["is_peak"]
    )

    df["Congestion_Score"] = congestion_score.round(4)

    conditions = [
        congestion_score < 0.40,
        congestion_score < 0.65,
    ]
    choices = ["Low", "Medium"]
    df["Congestion Level"] = np.select(conditions, choices, default="High")

    dist = df["Congestion Level"].value_counts()
    print(f"\n--- Derived congestion distribution ---")
    for level in ["Low", "Medium", "High"]:
        cnt = dist.get(level, 0)
        print(f"  {level}: {cnt:,} ({cnt / len(df) * 100:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# 5. Feature engineering
# ---------------------------------------------------------------------------

def engineer_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features compatible with the existing pipeline + new route-specific ones.
    """
    df = df.copy()

    # --- Time features ---
    df["Hour"] = df["hour"]

    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6,
    }
    df["Day_of_Week"] = df["day_of_week"].map(day_map).fillna(0).astype(int)
    df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)

    # Synthetic month (not available in routes data, distribute uniformly)
    np.random.seed(42)
    df["Month"] = np.random.randint(1, 13, size=len(df))

    # --- Rename to match existing pipeline columns ---
    df["Traffic Volume"] = df["vehicles"]
    df["Average Speed (km/h)"] = df["speed"]
    df["Road Capacity Utilization (%)"] = (df["Volume_Capacity_Ratio"] * 100).clip(0, 100).round(1)

    # Travel Time Index: ratio of actual travel time to free-flow estimate
    free_flow_time = (df["distance"] / 65.0) * 60  # minutes at 65 km/h
    free_flow_time = free_flow_time.clip(lower=0.1)  # avoid division by zero
    df["Travel Time Index"] = (df["travel_time"] / free_flow_time).clip(1.0, 5.0).round(2)

    # --- Pedestrian & Cyclist Count (synthetic, correlated with area density) ---
    area_density = {
        "M.G. Road": 1.5, "Majestic": 1.6, "Koramangala": 1.3, "Indiranagar": 1.2,
        "Jayanagar": 1.1, "Whitefield": 0.8, "Electronic City": 0.7, "Hebbal": 0.9,
        "Yeshwanthpur": 1.0, "KR Puram": 0.85, "BTM Layout": 1.1, "JP Nagar": 0.95,
        "RR Nagar": 0.8, "Other": 0.7,
    }
    base_ped = np.random.randint(50, 300, size=len(df))
    density_mult = df["Source_Area"].map(area_density).fillna(0.7)
    df["Pedestrian & Cyclist Count"] = (base_ped * density_mult).astype(int)

    # --- Incident Reports (synthetic, correlated with congestion & weather) ---
    incident_base = np.where(df["Congestion Level"] == "High", 0.25,
                    np.where(df["Congestion Level"] == "Medium", 0.12, 0.04))
    weather_inc = np.where(df["weather"] == "Rain", 0.10, 0.0)
    df["Incident Reports"] = np.random.poisson((incident_base + weather_inc) * 10)

    # --- Is_Roadwork (synthetic, ~10% of routes) ---
    df["Is_Roadwork"] = (np.random.random(len(df)) < 0.10).astype(int)

    # --- Historical Incident Rate (area-based average) ---
    area_incident_rates = {
        "M.G. Road": 320, "Majestic": 380, "Koramangala": 280, "Indiranagar": 250,
        "Jayanagar": 220, "Whitefield": 190, "Electronic City": 210, "Hebbal": 260,
        "Yeshwanthpur": 240, "KR Puram": 230, "BTM Layout": 200, "JP Nagar": 195,
        "RR Nagar": 180, "Other": 170,
    }
    df["Historical Incident Rate"] = df["Source_Area"].map(area_incident_rates).fillna(170).round(2)

    # --- Weather features ---
    weather_severity = {"Clear": 0, "Fog": 1, "Rain": 2, "Heavy Rain": 3}
    df["Weather_Severity"] = df["weather"].map(weather_severity).fillna(0).astype(int)
    df["Weather_Hour"] = df["Weather_Severity"] * df["Hour"]

    # --- Roadwork x Volume interaction ---
    df["Roadwork_Volume"] = df["Is_Roadwork"] * df["Traffic Volume"]

    # --- Congestion ordinal ---
    df["Congestion_Ordinal"] = df["Congestion Level"].map(CONGESTION_MAP).astype(int)

    # --- Route-specific features (NEW) ---
    df["Speed_Per_Km"] = (df["speed"] / df["distance"].clip(lower=0.1)).round(3)
    df["Signal_Density"] = (df["signal_time"] / df["distance"].clip(lower=0.1)).round(3)
    df["Time_Efficiency"] = (
        ((df["distance"] / df["speed"].clip(lower=1)) * 60) / df["travel_time"].clip(lower=0.01)
    ).clip(0, 5).round(3)

    # --- Area Name (use source area as primary) ---
    df["Area Name"] = df["Source_Area"]

    # --- Road/Intersection Name (use source location) ---
    df["Road/Intersection Name"] = df["source_location"]

    # --- Weather Conditions (renamed for compatibility) ---
    df["Weather Conditions"] = df["weather"]

    # --- Create a synthetic Date column for pipeline compatibility ---
    # Distribute dates across 2022-2024 based on day_of_week
    base_dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    day_to_dates = {}
    for day_name in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        day_to_dates[day_name] = base_dates[base_dates.dayofweek == day_map.get(day_name, 0)]

    def assign_date(row_day, row_hour, idx):
        dates = day_to_dates.get(row_day, base_dates)
        date = dates[idx % len(dates)]
        return date.replace(hour=int(row_hour))

    print("\n  Assigning synthetic dates (for time-series compatibility) ...")
    # Vectorized date assignment for speed
    date_indices = np.arange(len(df))
    dates_list = []
    for day_name, group_idx in df.groupby("day_of_week").groups.items():
        available_dates = day_to_dates.get(day_name, base_dates)
        assigned = available_dates[np.arange(len(group_idx)) % len(available_dates)]
        for i, orig_idx in enumerate(group_idx):
            dates_list.append((orig_idx, assigned[i]))

    dates_list.sort(key=lambda x: x[0])
    df["Date"] = pd.to_datetime([d[1] for d in dates_list])
    # Add hour to Date
    df["Date"] = df["Date"] + pd.to_timedelta(df["Hour"], unit="h")

    print(f"\n  Engineered feature set: {df.shape[1]} columns")

    return df


# ---------------------------------------------------------------------------
# 6. Compute lag / rolling features (per road)
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling and lag congestion features per road, matching existing pipeline."""
    df = df.sort_values(["Road/Intersection Name", "Date"]).reset_index(drop=True)

    # Rolling congestion trend (3-record rolling mean per road)
    df["Rolling_Congestion_3"] = (
        df.groupby("Road/Intersection Name")["Congestion_Ordinal"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
        .round(3)
    )

    # Lag features
    for lag in [1, 2, 3]:
        df[f"Congestion_Lag_{lag}"] = (
            df.groupby("Road/Intersection Name")["Congestion_Ordinal"]
            .shift(lag)
        )
        df[f"Congestion_Lag_{lag}"] = df[f"Congestion_Lag_{lag}"].fillna(
            df["Congestion_Ordinal"]
        )

    print(f"  Added rolling & lag features")
    return df


# ---------------------------------------------------------------------------
# 7. Add coordinates
# ---------------------------------------------------------------------------

def add_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Add Latitude and Longitude columns from source coordinates."""
    df = df.copy()
    df["Latitude"] = df["lat_src"]
    df["Longitude"] = df["lon_src"]
    return df


# ---------------------------------------------------------------------------
# 8. Extract & save new location coordinates
# ---------------------------------------------------------------------------

def extract_and_update_coordinates(df: pd.DataFrame) -> dict:
    """
    Extract unique locations and their coordinates from the route data,
    then merge with existing coordinates.json.
    """
    # Load existing coordinates
    coords = load_coordinates()

    # Extract unique source locations with their coordinates
    src_coords = (
        df.groupby("source_location")
        .agg({"lat_src": "median", "lon_src": "median", "Source_Area": "first"})
        .reset_index()
    )

    new_roads_added = 0
    new_areas = set()

    for _, row in src_coords.iterrows():
        road_name = row["source_location"]
        area = row["Source_Area"]

        if road_name not in coords["roads"]:
            coords["roads"][road_name] = {
                "area": area,
                "lat": round(float(row["lat_src"]), 6),
                "lon": round(float(row["lon_src"]), 6),
            }
            new_roads_added += 1

        if area not in coords["areas"]:
            new_areas.add(area)

    # Add new area centroids
    area_centroids = (
        df.groupby("Source_Area")
        .agg({"lat_src": "median", "lon_src": "median"})
        .reset_index()
    )
    for _, row in area_centroids.iterrows():
        area = row["Source_Area"]
        if area not in coords["areas"]:
            coords["areas"][area] = {
                "lat": round(float(row["lat_src"]), 6),
                "lon": round(float(row["lon_src"]), 6),
            }

    print(f"\n--- Coordinates update ---")
    print(f"  New roads added: {new_roads_added}")
    print(f"  New areas added: {len(new_areas)} {list(new_areas)}")
    print(f"  Total roads in coordinates: {len(coords['roads'])}")
    print(f"  Total areas in coordinates: {len(coords['areas'])}")

    # Save updated coordinates
    with open(COORDINATES_PATH, "w") as f:
        json.dump(coords, f, indent=4)
    print(f"  Saved updated coordinates to {COORDINATES_PATH}")

    return coords


# ---------------------------------------------------------------------------
# 9. Downsample for manageable training size
# ---------------------------------------------------------------------------

def downsample_for_training(df: pd.DataFrame, max_rows: int = 200_000) -> pd.DataFrame:
    """
    Stratified downsample to keep training tractable while preserving
    congestion level distribution.
    """
    if len(df) <= max_rows:
        return df

    print(f"\n  Downsampling from {len(df):,} to ~{max_rows:,} rows (stratified) ...")

    # Store congestion distribution before sampling
    orig_dist = df["Congestion Level"].value_counts()

    sampled_parts = []
    for level, group in df.groupby("Congestion Level"):
        n_sample = min(len(group), int(max_rows * len(group) / len(df)))
        sampled_parts.append(group.sample(n=n_sample, random_state=42))

    sampled = pd.concat(sampled_parts, ignore_index=True)

    print(f"  Downsampled: {len(sampled):,} rows")
    print(f"  Distribution: {sampled['Congestion Level'].value_counts().to_dict()}")
    return sampled.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 10. Save processed route data
# ---------------------------------------------------------------------------

def save_route_processed(df: pd.DataFrame) -> str:
    """Save the processed route data alongside existing processed data."""
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    path = os.path.join(DATA_PROCESSED, "route_features.csv")
    df.to_csv(path, index=False)
    print(f"\n  Saved processed route data to {path}")
    print(f"  Shape: {df.shape}")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("ROUTE DATA PREPARATION")
    print("=" * 60)

    # Step 1: Load
    df = load_route_data()

    # Step 2: Clean
    df = clean_routes(df)

    # Step 3: Map locations to areas
    df = map_locations_to_areas(df)

    # Step 4: Derive congestion labels
    df = derive_congestion(df)

    # Step 5: Feature engineering
    df = engineer_route_features(df)

    # Step 6: Add lag/rolling features
    df = add_lag_features(df)

    # Step 7: Add coordinates
    df = add_coordinates(df)

    # Step 8: Update coordinates.json with new locations
    extract_and_update_coordinates(df)

    # Step 9: Downsample for training
    df_train = downsample_for_training(df, max_rows=200_000)

    # Step 10: Save
    save_route_processed(df_train)

    # Print final column summary
    print(f"\n{'='*60}")
    print("FINAL COLUMNS")
    print(f"{'='*60}")
    for col in sorted(df_train.columns):
        print(f"  {col}: {df_train[col].dtype}")

    print(f"\n{'='*60}")
    print("Route data preparation COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
