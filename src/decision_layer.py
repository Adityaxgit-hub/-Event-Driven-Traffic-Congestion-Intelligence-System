"""
Step 5 -- Decision layer: priority scoring and diversion suggestions.

Combines classifier predictions, forecast trends, and historical incident
rates to produce actionable route recommendations for commuters.

Usage:
    python -m src.decision_layer   (standalone test)
"""

import numpy as np
import pandas as pd

from src.utils import (
    CONGESTION_MAP,
    load_coordinates,
    nearest_alternate_road,
)

# ---------------------------------------------------------------------------
# 1. Priority Scoring
# ---------------------------------------------------------------------------

def compute_priority_scores(
    roads: list[str],
    predicted_congestion: dict[str, int],
    incident_rates: dict[str, float],
    forecast_trends: dict[str, str],
) -> pd.DataFrame:
    """
    Composite priority score per road.

    Score = 0.45 * congestion_level + 0.35 * normalized_incident_rate
            + 0.20 * forecast_trend_score

    Returns DataFrame with columns:
        Road, Congestion_Level, Incident_Rate, Forecast_Trend,
        Priority_Score, Priority_Label
    """
    # Normalize incident rates to 0-1 (protect against NaN)
    rates = np.nan_to_num(
        np.array([float(incident_rates.get(r, 0)) for r in roads]), nan=0.0
    )
    max_rate = rates.max() if rates.max() > 0 else 1
    norm_rates = rates / max_rate

    trend_score_map = {"Worsening": 1.0, "Stable": 0.5, "Improving": 0.0}

    rows = []
    for i, road in enumerate(roads):
        cong = int(predicted_congestion.get(road, 0))  # 0/1/2
        trend = forecast_trends.get(road, "Stable")
        trend_val = trend_score_map.get(trend, 0.5)

        score = 0.45 * (cong / 2) + 0.35 * float(norm_rates[i]) + 0.20 * trend_val
        rows.append({
            "Road": road,
            "Congestion_Level": cong,
            "Incident_Rate": float(rates[i]),
            "Forecast_Trend": trend,
            "Priority_Score": round(float(score), 4),
        })

    df = pd.DataFrame(rows).sort_values("Priority_Score", ascending=False).reset_index(drop=True)

    # Label: top tercile = High, middle = Medium, bottom = Low
    n = len(df)
    tercile = n // 3
    df["Priority_Label"] = "Low"
    df.loc[:tercile, "Priority_Label"] = "High"
    df.loc[tercile + 1 : 2 * tercile, "Priority_Label"] = "Medium"

    return df


# ---------------------------------------------------------------------------
# 2. Diversion Suggestions
# ---------------------------------------------------------------------------

def suggest_diversions(
    roads: list[str],
    predicted_congestion: dict[str, int],
) -> dict[str, str | None]:
    """
    For each road, suggest the nearest alternate road with lower congestion.
    Returns dict: road_name -> suggested_alternate (or None).
    """
    coords = load_coordinates()
    diversions = {}
    for road in roads:
        alt = nearest_alternate_road(road, predicted_congestion, coords)
        diversions[road] = alt
    return diversions


# ---------------------------------------------------------------------------
# 3. Combined decision output
# ---------------------------------------------------------------------------

def generate_decisions(
    roads: list[str],
    predicted_congestion: dict[str, int],
    incident_rates: dict[str, float],
    forecast_trends: dict[str, str],
) -> pd.DataFrame:
    """
    Full decision pipeline: priority scoring + diversion suggestions.
    Returns a DataFrame with one row per road and all decision outputs.
    """
    # Priority scoring
    priority_df = compute_priority_scores(
        roads, predicted_congestion, incident_rates, forecast_trends
    )

    # Diversion suggestions
    diversions = suggest_diversions(roads, predicted_congestion)
    priority_df["Suggested_Diversion"] = priority_df["Road"].map(diversions).fillna("--")

    return priority_df


# ---------------------------------------------------------------------------
# Entry point (standalone test)
# ---------------------------------------------------------------------------

def main():
    """Quick standalone test with dummy predictions."""
    from src.utils import load_coordinates

    coords = load_coordinates()
    roads = list(coords["roads"].keys())

    # Dummy predictions
    np.random.seed(42)
    predicted_congestion = {r: np.random.choice([0, 1, 2]) for r in roads}
    incident_rates = {r: round(np.random.uniform(100, 400), 1) for r in roads}
    forecast_trends = {
        r: np.random.choice(["Improving", "Stable", "Worsening"]) for r in roads
    }

    print("=" * 60)
    print("STEP 5 -- Decision Layer (test run)")
    print("=" * 60)

    decisions = generate_decisions(
        roads, predicted_congestion, incident_rates, forecast_trends
    )
    print("\n" + decisions.to_string(index=False))
    print("\nDone OK")


if __name__ == "__main__":
    main()
