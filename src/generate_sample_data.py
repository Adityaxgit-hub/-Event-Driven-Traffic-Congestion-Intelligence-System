"""
Generate realistic synthetic data that mirrors the Kaggle Bangalore Traffic
Dataset and BTP accident data. Run this to bootstrap the project before you
have the real CSVs.

Usage:
    python -m src.generate_sample_data
"""

import os
import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.utils import DATA_RAW, COORDINATES_PATH, WEATHER_CONDITIONS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_RECORDS = 9000  # ~8,900 in real dataset
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2024, 12, 31)

# Load road/area info from coordinates.json
with open(COORDINATES_PATH, "r") as f:
    _coords = json.load(f)

ROADS = list(_coords["roads"].keys())          # 16 roads
AREAS = {r: _coords["roads"][r]["area"] for r in ROADS}  # road -> area

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_date() -> datetime:
    delta = END_DATE - START_DATE
    return START_DATE + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def _traffic_volume(hour: int, weather: str, is_roadwork: bool) -> int:
    """Simulate traffic volume based on time/weather patterns."""
    # Base volume by hour (morning/evening peaks)
    if 8 <= hour <= 10 or 17 <= hour <= 20:
        base = random.randint(3000, 6000)
    elif 11 <= hour <= 16:
        base = random.randint(2000, 4000)
    elif 6 <= hour <= 7 or 21 <= hour <= 23:
        base = random.randint(1000, 2500)
    else:
        base = random.randint(200, 1000)

    # Weather multiplier
    weather_mult = {"Clear": 1.0, "Fog": 1.1, "Rain": 1.25, "Heavy Rain": 1.45}
    base = int(base * weather_mult.get(weather, 1.0))

    # Roadwork reduces capacity -> similar volume but slower
    if is_roadwork:
        base = int(base * random.uniform(0.85, 1.0))

    return base


def _derive_congestion(volume: int, speed: float, capacity_usage: float) -> str:
    """Deterministic-ish congestion label from volume/speed/capacity."""
    score = (capacity_usage / 100) * 0.5 + (1 - speed / 60) * 0.3 + (volume / 6000) * 0.2
    if score > 0.65:
        return "High"
    elif score > 0.40:
        return "Medium"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_traffic_data() -> pd.DataFrame:
    """Generate synthetic traffic data mimicking the Kaggle dataset schema."""
    rows = []
    for _ in range(NUM_RECORDS):
        dt = _random_date()
        road = random.choice(ROADS)
        area = AREAS[road]
        weather = random.choices(
            WEATHER_CONDITIONS,
            weights=[0.55, 0.10, 0.25, 0.10],
            k=1,
        )[0]
        is_roadwork = random.random() < 0.12  # 12 % chance

        hour = dt.hour
        volume = _traffic_volume(hour, weather, is_roadwork)

        # Average speed inversely correlated with volume
        base_speed = max(8, 55 - (volume / 120) + random.gauss(0, 5))
        avg_speed = round(base_speed, 1)

        # Road capacity utilization
        capacity_usage = min(100, round(volume / 60 + random.gauss(0, 5), 1))

        # Travel time index (ratio vs free-flow)
        tti = round(max(1.0, (55 / max(avg_speed, 5)) + random.gauss(0, 0.1)), 2)

        # Pedestrian count
        ped_count = random.randint(20, 500)

        # Incident count -- higher during peak hours & bad weather
        incident_base = 0.05
        if weather in ("Rain", "Heavy Rain"):
            incident_base += 0.10
        if 8 <= hour <= 10 or 17 <= hour <= 20:
            incident_base += 0.08
        incident_reports = np.random.poisson(incident_base * 10)

        congestion = _derive_congestion(volume, avg_speed, capacity_usage)

        rows.append({
            "Date": dt.strftime("%Y-%m-%d"),
            "Hour": hour,
            "Area Name": area,
            "Road/Intersection Name": road,
            "Weather Conditions": weather,
            "Is_Roadwork": int(is_roadwork),
            "Traffic Volume": volume,
            "Average Speed (km/h)": avg_speed,
            "Travel Time Index": tti,
            "Road Capacity Utilization (%)": capacity_usage,
            "Pedestrian & Cyclist Count": ped_count,
            "Incident Reports": incident_reports,
            "Congestion Level": congestion,
        })

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Area Name", "Road/Intersection Name"]).reset_index(drop=True)
    return df


def generate_btp_accident_data() -> pd.DataFrame:
    """Generate synthetic BTP accident data (station-wise annual)."""
    areas = list(_coords["areas"].keys())
    rows = []
    for year in range(2018, 2024):
        for area in areas:
            total_accidents = random.randint(80, 450)
            fatal = int(total_accidents * random.uniform(0.02, 0.08))
            injury = int(total_accidents * random.uniform(0.15, 0.35))
            rows.append({
                "Year": year,
                "Police Station / Area": area,
                "Total Accidents": total_accidents,
                "Fatal Accidents": fatal,
                "Injury Accidents": injury,
                "Fatalities": random.randint(fatal, fatal + 10),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("Generating synthetic traffic data ...")
    traffic_df = generate_traffic_data()
    traffic_path = os.path.join(DATA_RAW, "bangalore_traffic.csv")
    traffic_df.to_csv(traffic_path, index=False)
    print(f"  Saved {len(traffic_df)} records to {traffic_path}")

    print("Generating synthetic BTP accident data ...")
    btp_df = generate_btp_accident_data()
    btp_path = os.path.join(DATA_RAW, "btp_accidents.csv")
    btp_df.to_csv(btp_path, index=False)
    print(f"  Saved {len(btp_df)} records to {btp_path}")

    print("Done. Raw data is ready in data/raw/")


if __name__ == "__main__":
    main()
