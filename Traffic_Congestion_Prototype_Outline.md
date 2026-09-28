# Event-Driven Traffic Congestion Intelligence System (Prototype)

## 1. Core Concept

Predict traffic congestion in Bengaluru, then turn the prediction into simple operational suggestions — where to send police, which junctions need priority attention, and a basic diversion suggestion. Built as a resume-focused prototype, not a production system.

"Event-driven" here means the system reacts to changing inputs — a user picks a different date/time/weather scenario, and predictions + recommendations update instantly. Not a real-time streaming system, but a responsive what-if simulator backed by ML.

## 2. Datasets Used

- **Bangalore City Traffic Dataset (Kaggle)** — ~8,900 records, 2022-2024, 8 neighborhoods / 16 roads. Fields: date, area, road, weather, roadwork flag, traffic volume, average speed, congestion level, travel time index, road capacity usage, pedestrian count, incident counts.
- **Bengaluru Traffic Police open data (OpenCity.in)** — station-wise accidents & fatalities (2018-2023).

### Merge Strategy

The BTP data is station-level annual; the Kaggle data is road-level with dates. These don't join row-for-row. Instead:
- Map each BTP station to the nearest Kaggle area/road (manual mapping, ~8 areas).
- Aggregate BTP data into a **historical incident rate per area** — a static feature, not a time-varying join.
- This gives each road a "danger score" from the accident data, which feeds into priority scoring and the classifier.

## 3. Simplifying Assumptions (state these openly, don't over-engineer)

- **Coordinates**: manually assign approximate lat/long to the 16 roads (small fixed list, no need for a full geocoding pipeline).
- **Manpower availability**: assume a fixed pool (e.g. 50 officers across 8 areas) — simulated, same approach as the FASTag simulation in the toll project.
- **Barricade locations**: use the highest-incident junctions from the accident data as "priority points" — no separate barricade dataset needed.
- **Weather**: use the historical weather column as-is; assume it's known at prediction time.
- **Real-time data**: none. The prototype runs on historical data; the dashboard simulates "current" conditions by letting the user pick a date/time/weather scenario.
- **Time granularity**: the Kaggle dataset has date-level records per road. If hourly granularity is needed, simulate it from time features (hour-of-day patterns).

These are reasonable, defensible shortcuts for a prototype — mention them plainly if asked, don't try to hide them.

## 4. Pipeline

### Step 1 — Data Cleaning & Preparation
- Handle missing values (drop or impute — document which and why)
- Standardize road/area names across both datasets (e.g. "Koramangala" vs "Koramangala Area")
- Parse dates, extract hour/day-of-week
- Map BTP stations → Kaggle areas (manual lookup table)
- Compute historical incident rate per area from BTP data
- Merge incident rate as a static feature into the traffic dataset
- Assign approximate lat/long coordinates to each road (hardcoded dict)

**Output**: a single clean DataFrame with all features, ready for modeling.

### Step 2 — Feature Engineering
- **Time features**: hour of day, day of week, is_weekend flag
- **Rolling congestion trend**: rolling mean of congestion level over past N records for each road (e.g. 3-day rolling average)
- **Historical incident rate**: from BTP merge (static per area)
- **Interaction features**: weather × hour, roadwork × traffic volume (test and keep only if they improve CV score)
- **Target encoding**: encode congestion_level as ordinal (Low=0, Medium=1, High=2) for the forecaster; keep as categorical for the CatBoost classifier

**Output**: feature matrix X, target y. Save feature list for reproducibility.

### Step 3 — Classifier (CatBoost)
- **Target**: congestion level — Low / Medium / High
- **Why CatBoost**: handles categorical features natively, works well on tabular data with modest tuning
- **Split**: time-based split (train on earlier dates, test on later) — not random, to avoid temporal leakage
- **Metrics**: macro F1 (primary), confusion matrix, per-class precision/recall
- **Tuning**: light grid search on depth, learning rate, iterations. Nothing heavy — this is a prototype.
- **Save**: trained model via joblib

### Step 4 — Forecaster
- **Goal**: predict congestion level a few time steps ahead per road (e.g. next 1–3 periods)
- **Approach**: lag-feature regression — use past congestion values and time features as inputs, predict future congestion as a continuous target (travel time index or ordinal congestion level)
- **Model**: Ridge or LightGBM — whichever gives better CV scores. Keep it simple.
- **Evaluation**: RMSE on held-out time window
- **Save**: trained model via joblib

This is intentionally simpler than the classifier. The forecast exists to show "things might get worse here in the next hour" — directional, not precise.

### Step 5 — Decision Layer
- **Manpower allocation**: integer linear program (PuLP). Inputs: predicted risk score per location, available officers. Output: how many officers to assign where. Constraint: total assigned ≤ available pool.
- **Priority scoring**: composite score = weighted sum of (predicted congestion level + historical incident rate + forecast trend). Rank locations, label top tercile as High, middle as Medium, bottom as Low.
- **Diversion suggestion**: for each congested road, suggest the nearest alternate road (by straight-line distance from hardcoded coordinates) that has a lower predicted congestion level. No graph routing.

**Output**: a dict per road → `{officers_assigned, priority_label, suggested_diversion}`.

## 5. Tech Stack

| Purpose | Tool |
|---|---|
| Data processing | Pandas, NumPy |
| Classifier | CatBoost |
| Forecaster | scikit-learn (Ridge) or LightGBM |
| Optimization | PuLP |
| Model persistence | joblib |
| Dashboard | Streamlit |
| Map visualization | Folium (via streamlit-folium) or PyDeck |
| Coordinates | Hardcoded dict (no geocoding API) |

## 6. Folder Structure

```
traffic-congestion-system/
│
├── data/
│   ├── raw/                  # Original CSVs (Kaggle + BTP)
│   ├── processed/            # Cleaned, merged DataFrame
│   └── coordinates.json      # Hardcoded lat/long for 16 roads
│
├── notebooks/
│   └── eda.ipynb             # Exploratory analysis (optional, for your own reference)
│
├── src/
│   ├── data_prep.py          # Cleaning, merging, feature engineering
│   ├── classifier.py         # CatBoost training & evaluation
│   ├── forecaster.py         # Forecaster training & evaluation
│   ├── decision_layer.py     # PuLP optimization, priority scoring, diversion logic
│   └── utils.py              # Shared helpers (load data, constants, coordinate lookup)
│
├── models/
│   ├── catboost_model.joblib
│   └── forecaster_model.joblib
│
├── dashboard/
│   └── app.py                # Streamlit app
│
├── requirements.txt
└── README.md
```

## 7. User Interaction Flow

The user is a traffic control operator, not a commuter — so the flow is kept minimal.

1. **Map overview** — open the Streamlit app, see a map of Bengaluru with 16 roads marked as colored pins (green / amber / red by predicted congestion). A summary table beside the map.
2. **Select a road** — pick from a dropdown or click on the map. See:
   - Current predicted congestion level + confidence
   - Short forecast (next 1–3 periods: improving / stable / worsening)
   - Historical incident rate for that area
3. **View recommended action** — below the prediction, three cards:
   - 🚔 **Personnel**: "Assign 3 officers" (from PuLP allocation)
   - ⚠️ **Priority**: High / Medium / Low (with score breakdown)
   - 🔀 **Diversion**: "Suggest rerouting via [alternate road]" (or "No diversion needed")
4. **What-if controls** — sidebar with date/time picker, weather dropdown, roadwork toggle. Changing these re-runs the prediction pipeline and updates everything live.

Single-page app. No login, no saved sessions, no multi-page navigation.

## 8. Evaluation & Metrics (know these for interviews)

| Model | Primary Metric | Target |
|---|---|---|
| CatBoost classifier | Macro F1 | > 0.75 |
| Forecaster | RMSE | Reasonable (scale-dependent) |
| PuLP allocation | Feasibility | All constraints satisfied, no over-assignment |

Also prepare: confusion matrix visualization, feature importance plot (CatBoost has this built-in), and a short paragraph on what you'd improve with more data/time.

## 9. What to Say in Interviews

- Real data used for classification and forecasting (Kaggle + BTP open data)
- Coordinates, manpower numbers, and barricade points are stated assumptions for a prototype — not hidden
- Focus the explanation on: (1) the ML pipeline (CatBoost + forecaster), (2) the decision layer logic (optimization is the differentiator), (3) the end-to-end flow from data → prediction → actionable recommendation
- If asked **"why not real-time?"**: "This is a prototype to demonstrate the pipeline. In production, you'd plug in a streaming source (Google Maps API, sensor feeds) and run the models on a schedule or trigger."
- If asked **"why CatBoost over XGBoost/Random Forest?"**: "Native categorical support, fast training, good defaults. For a prototype, the pipeline design matters more than model selection."

## 10. Demo Script (2-minute walkthrough)

1. Open dashboard → "Here's the live view of 16 roads across 8 areas in Bengaluru."
2. Point to a red pin → "This road is predicted High congestion right now."
3. Select it → "The model says High with 87% confidence. The forecast shows it worsening in the next period."
4. Show action cards → "The system recommends 3 officers here, flags it as High priority, and suggests diverting traffic to [alternate road]."
5. Change weather to Rain → "Watch the predictions shift — rain increases congestion on these 4 roads."
6. Close with → "The entire pipeline — data, ML, optimization, dashboard — runs end-to-end in under 2 seconds."
