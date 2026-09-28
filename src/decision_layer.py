"""
Step 5 -- Decision layer: manpower allocation, priority scoring, diversion.

Combines classifier predictions, forecast trends, and historical incident
rates to produce actionable recommendations per road.

Usage:
    python -m src.decision_layer   (standalone test)
"""

import numpy as np
import pandas as pd
from pulp import (
    LpMaximize,
    LpProblem,
    lpSum,
    value,
    HiGHS,
)

from src.utils import (
    TOTAL_OFFICERS,
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
# 2. Manpower Allocation (PuLP integer program)
# ---------------------------------------------------------------------------

def allocate_manpower(
    priority_df: pd.DataFrame,
    total_officers: int = TOTAL_OFFICERS,
    min_per_high: int = 3,
    min_per_medium: int = 1,
) -> dict[str, int]:
    """
    Solve an integer linear program to allocate officers to roads.

    Objective: maximize total weighted coverage
        (officers_i x priority_score_i)

    Constraints:
        - sum(officers) <= total_officers
        - High-priority roads get at least `min_per_high` officers
        - Medium-priority roads get at least `min_per_medium`
        - Low-priority roads get at least 0
        - Each road gets at most 10 officers

    Returns dict: road_name -> officers_assigned
    """
    roads = priority_df["Road"].tolist()
    # Ensure scores are plain Python floats (no NaN/numpy types)
    scores = {
        r: float(s) if not (s != s) else 0.0  # NaN check: NaN != NaN
        for r, s in zip(priority_df["Road"], priority_df["Priority_Score"])
    }
    labels = dict(zip(priority_df["Road"], priority_df["Priority_Label"]))

    prob = LpProblem("ManpowerAllocation", LpMaximize)

    # Decision variables (PuLP v4: create via prob.add_variable)
    x = {}
    for i, road in enumerate(roads):
        x[road] = prob.add_variable(
            f"officers_{i}", lowBound=0, upBound=10, cat="Integer"
        )

    # Objective: maximize weighted coverage
    prob += lpSum([scores[r] * x[r] for r in roads])

    # Total constraint
    prob += lpSum([x[r] for r in roads]) <= total_officers

    # Minimum assignment per priority
    for r in roads:
        if labels[r] == "High":
            prob += x[r] >= min_per_high
        elif labels[r] == "Medium":
            prob += x[r] >= min_per_medium

    # Solve (explicitly use HiGHS solver, suppress output)
    stats = prob.solve(HiGHS(msg=0))

    if not stats.has_solution:
        print(f"  [!] PuLP status: {stats.status} -- using fallback allocation")
        # Fallback: distribute evenly
        per_road = total_officers // len(roads)
        return {r: per_road for r in roads}

    allocation = {r: int(value(x[r])) for r in roads}
    total_used = sum(allocation.values())
    print(f"  Allocated {total_used}/{total_officers} officers across {len(roads)} roads")
    return allocation


# ---------------------------------------------------------------------------
# 3. Diversion Suggestions
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
# 4. Combined decision output
# ---------------------------------------------------------------------------

def generate_decisions(
    roads: list[str],
    predicted_congestion: dict[str, int],
    incident_rates: dict[str, float],
    forecast_trends: dict[str, str],
    total_officers: int = TOTAL_OFFICERS,
) -> pd.DataFrame:
    """
    Full decision pipeline: priority -> allocation -> diversion.
    Returns a DataFrame with one row per road and all decision outputs.
    """
    # Priority scoring
    priority_df = compute_priority_scores(
        roads, predicted_congestion, incident_rates, forecast_trends
    )

    # Manpower allocation
    allocation = allocate_manpower(priority_df, total_officers=total_officers)
    priority_df["Officers_Assigned"] = priority_df["Road"].map(allocation)

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
