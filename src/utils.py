"""
Shared utilities: paths, constants, coordinate lookup, distance helpers.
"""

import json
import os
from math import radians, sin, cos, sqrt, atan2

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
COORDINATES_PATH = os.path.join(PROJECT_ROOT, "data", "coordinates.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Ensure directories exist
for _dir in [DATA_RAW, DATA_PROCESSED, MODELS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONGESTION_LEVELS = ["Low", "Medium", "High"]
CONGESTION_MAP = {"Low": 0, "Medium": 1, "High": 2}
CONGESTION_MAP_INV = {v: k for k, v in CONGESTION_MAP.items()}

WEATHER_CONDITIONS = ["Clear", "Fog", "Rain", "Heavy Rain"]

# Simulated manpower: total officers available for deployment
TOTAL_OFFICERS = 50

# Number of areas / stations
NUM_AREAS = 8

# Feature columns used by the classifier (populated during training, saved here for reference)
CLASSIFIER_FEATURES = None  # set at runtime

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def load_coordinates() -> dict:
    """Load the hardcoded coordinates JSON."""
    with open(COORDINATES_PATH, "r") as f:
        return json.load(f)


def get_road_coords(road_name: str, coords: dict | None = None) -> tuple[float, float]:
    """Return (lat, lon) for a given road name."""
    if coords is None:
        coords = load_coordinates()
    road_info = coords["roads"].get(road_name)
    if road_info is None:
        raise KeyError(f"Road '{road_name}' not found in coordinates.json")
    return road_info["lat"], road_info["lon"]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def nearest_alternate_road(
    road_name: str,
    congestion_levels: dict[str, int],
    coords: dict | None = None,
) -> str | None:
    """
    Find the nearest road with a lower predicted congestion level.
    Returns road name or None if no better alternative exists.

    Parameters
    ----------
    road_name : str
        The congested road.
    congestion_levels : dict
        Mapping of road_name -> predicted congestion (0/1/2).
    coords : dict, optional
        Pre-loaded coordinates dict.
    """
    if coords is None:
        coords = load_coordinates()

    current_level = congestion_levels.get(road_name, 0)
    if current_level == 0:
        return None  # already Low, no diversion needed

    src_lat, src_lon = get_road_coords(road_name, coords)
    best_road = None
    best_dist = float("inf")

    for other_road, other_level in congestion_levels.items():
        if other_road == road_name:
            continue
        if other_level >= current_level:
            continue  # not a better option
        try:
            o_lat, o_lon = get_road_coords(other_road, coords)
        except KeyError:
            continue
        dist = haversine_km(src_lat, src_lon, o_lat, o_lon)
        if dist < best_dist:
            best_dist = dist
            best_road = other_road

    return best_road


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_processed_data() -> pd.DataFrame:
    """Load the cleaned, feature-engineered dataset."""
    path = os.path.join(DATA_PROCESSED, "traffic_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run data_prep.py first."
        )
    return pd.read_csv(path, parse_dates=["Date"])
