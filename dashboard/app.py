"""
Streamlit Dashboard -- Traffic Congestion Intelligence System

Single-page app for traffic control operators.
Displays a map, predictions, forecasts, and decision-layer recommendations.

Usage:
    streamlit run dashboard/app.py
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

from src.utils import (
    CONGESTION_MAP,
    CONGESTION_MAP_INV,
    CONGESTION_LEVELS,
    WEATHER_CONDITIONS,
    TOTAL_OFFICERS,
    load_coordinates,
    load_processed_data,
)
from src.classifier import load_classifier, predict_congestion, FEATURE_COLS as CLF_FEATURES
from src.forecaster import load_forecaster, forecast_congestion, forecast_trend
from src.decision_layer import generate_decisions
from src.api_config import (
    CARTO_API_KEY,
    MAP_STYLES,
    DEFAULT_MAP_STYLE,
    get_map_config,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Traffic Congestion Intelligence",
    page_icon="🚦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Light theme overrides */
    .stApp {
        background-color: #f5f7fa;
    }

    /* Main header */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Metric cards -- light theme */
    .metric-card {
        background: #ffffff;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #667eea;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; color: #6c757d; }
    .metric-card h1 { margin: 0.3rem 0 0 0; font-size: 1.8rem; color: #1a1a2e; }

    .card-red { border-left-color: #e74c3c; }
    .card-red h1 { color: #e74c3c; }
    .card-amber { border-left-color: #f39c12; }
    .card-amber h1 { color: #f39c12; }
    .card-green { border-left-color: #27ae60; }
    .card-green h1 { color: #27ae60; }
    .card-blue { border-left-color: #3498db; }
    .card-blue h1 { color: #3498db; }

    /* Decision cards */
    .decision-card {
        background: #ffffff;
        border-left: 4px solid #667eea;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
    }
    .decision-card.high { border-left-color: #e74c3c; }
    .decision-card.medium { border-left-color: #f39c12; }
    .decision-card.low { border-left-color: #27ae60; }

    /* Hide Streamlit default elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data & model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def get_data():
    return load_processed_data()

@st.cache_resource
def get_classifier():
    return load_classifier()

@st.cache_resource
def get_forecaster():
    return load_forecaster()

@st.cache_data
def get_coordinates():
    return load_coordinates()


# ---------------------------------------------------------------------------
# Sidebar -- What-if controls
# ---------------------------------------------------------------------------

st.sidebar.markdown("## 🎛️ Scenario Controls")
st.sidebar.markdown("Adjust parameters to simulate different conditions.")

try:
    df = get_data()
    clf_model = get_classifier()
    fcast_model = get_forecaster()
    coords = get_coordinates()
    data_loaded = True
except FileNotFoundError as e:
    data_loaded = False
    st.error(f"[!]️ {e}")
    st.info(
        "Run the pipeline first:\n"
        "```\n"
        "python -m src.generate_sample_data\n"
        "python -m src.data_prep\n"
        "python -m src.classifier\n"
        "python -m src.forecaster\n"
        "```"
    )
    st.stop()

roads = list(coords["roads"].keys())
areas = list(coords["areas"].keys())

# Date/time picker
sim_date = st.sidebar.date_input(
    "📅 Date",
    value=df["Date"].max().date(),
    min_value=df["Date"].min().date(),
    max_value=df["Date"].max().date(),
)
sim_hour = st.sidebar.slider("🕐 Hour of Day", 0, 23, 9)

# Weather
sim_weather = st.sidebar.selectbox("🌤️ Weather", WEATHER_CONDITIONS, index=0)

# Roadwork toggle
sim_roadwork = st.sidebar.checkbox("🚧 Active Roadwork", value=False)

# Officer count
sim_officers = st.sidebar.slider(
    "👮 Available Officers", 10, 100, TOTAL_OFFICERS, step=5
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Map Settings")

# Map style picker
map_style = st.sidebar.selectbox(
    "Map Style",
    list(MAP_STYLES.keys()),
    index=list(MAP_STYLES.keys()).index(DEFAULT_MAP_STYLE),
)

# Show API key status
if CARTO_API_KEY:
    st.sidebar.success("CARTO API key loaded", icon="🔑")
else:
    st.sidebar.caption("No CARTO API key set. Using free tiles.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Prototype -- Bengaluru Traffic Intelligence System</small>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Build scenario DataFrame
# ---------------------------------------------------------------------------

def build_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a scenario DataFrame (one row per road) using the sidebar inputs
    and historical data for baseline values.
    """
    # Get the closest historical records for baseline numeric values
    date_mask = df["Date"].dt.date == sim_date
    if date_mask.sum() == 0:
        # Fallback: use all data averaged by road
        baseline = df.groupby("Road/Intersection Name").mean(numeric_only=True).reset_index()
    else:
        baseline = df[date_mask].groupby("Road/Intersection Name").mean(numeric_only=True).reset_index()

    scenario_rows = []
    for road in roads:
        road_baseline = baseline[baseline["Road/Intersection Name"] == road]
        if len(road_baseline) == 0:
            road_baseline = df[df["Road/Intersection Name"] == road].mean(numeric_only=True)
        else:
            road_baseline = road_baseline.iloc[0]

        weather_severity = {"Clear": 0, "Fog": 1, "Rain": 2, "Heavy Rain": 3}

        row = {
            "Road/Intersection Name": road,
            "Area Name": coords["roads"][road]["area"],
            "Date": pd.Timestamp(sim_date),
            "Hour": sim_hour,
            "Day_of_Week": pd.Timestamp(sim_date).dayofweek,
            "Is_Weekend": int(pd.Timestamp(sim_date).dayofweek >= 5),
            "Month": pd.Timestamp(sim_date).month,
            "Weather Conditions": sim_weather,
            "Weather_Severity": weather_severity.get(sim_weather, 0),
            "Is_Roadwork": int(sim_roadwork),
            "Traffic Volume": road_baseline.get("Traffic Volume", 3000),
            "Average Speed (km/h)": road_baseline.get("Average Speed (km/h)", 30),
            "Travel Time Index": road_baseline.get("Travel Time Index", 1.5),
            "Road Capacity Utilization (%)": road_baseline.get("Road Capacity Utilization (%)", 60),
            "Pedestrian & Cyclist Count": road_baseline.get("Pedestrian & Cyclist Count", 200),
            "Incident Reports": road_baseline.get("Incident Reports", 1),
            "Historical Incident Rate": road_baseline.get("Historical Incident Rate", 200),
            "Rolling_Congestion_3": road_baseline.get("Rolling_Congestion_3", 1.0),
            "Congestion_Lag_1": road_baseline.get("Congestion_Lag_1", 1.0),
            "Congestion_Lag_2": road_baseline.get("Congestion_Lag_2", 1.0),
            "Congestion_Lag_3": road_baseline.get("Congestion_Lag_3", 1.0),
            "Weather_Hour": weather_severity.get(sim_weather, 0) * sim_hour,
            "Roadwork_Volume": int(sim_roadwork) * road_baseline.get("Traffic Volume", 3000),
            "Latitude": coords["roads"][road]["lat"],
            "Longitude": coords["roads"][road]["lon"],
        }
        scenario_rows.append(row)

    return pd.DataFrame(scenario_rows)


# ---------------------------------------------------------------------------
# Run predictions
# ---------------------------------------------------------------------------

scenario_df = build_scenario(df)
pred_df = predict_congestion(clf_model, scenario_df)

# Build dicts for decision layer
# CatBoost predictions may have whitespace; strip and map safely
pred_df["Predicted_Congestion"] = pred_df["Predicted_Congestion"].astype(str).str.strip()
pred_df["Predicted_Congestion_Ordinal"] = (
    pred_df["Predicted_Congestion"]
    .map(CONGESTION_MAP)
    .fillna(1)  # default to Medium if mapping fails
    .astype(int)
)
predicted_congestion = dict(
    zip(pred_df["Road/Intersection Name"], pred_df["Predicted_Congestion_Ordinal"])
)
incident_rates = dict(
    zip(pred_df["Road/Intersection Name"], pred_df["Historical Incident Rate"].fillna(0))
)

# Forecasts per road
forecast_results = {}
forecast_trends = {}
for _, row in pred_df.iterrows():
    road = row["Road/Intersection Name"]
    fc = forecast_congestion(fcast_model, pd.DataFrame([row]), steps_ahead=3)
    forecast_results[road] = fc
    forecast_trends[road] = forecast_trend(fc)

# Decision layer
decisions_df = generate_decisions(
    roads=roads,
    predicted_congestion=predicted_congestion,
    incident_rates=incident_rates,
    forecast_trends=forecast_trends,
    total_officers=sim_officers,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<div class="main-header">🚦 Traffic Congestion Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Bengaluru · {sim_date.strftime("%d %b %Y")} · '
    f'{sim_hour:02d}:00 · {sim_weather}</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

n_high = sum(1 for v in predicted_congestion.values() if v == 2)
n_medium = sum(1 for v in predicted_congestion.values() if v == 1)
n_low = sum(1 for v in predicted_congestion.values() if v == 0)
total_assigned = decisions_df["Officers_Assigned"].sum()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f'<div class="metric-card card-red"><h3>🔴 High Congestion</h3><h1>{n_high}</h1></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="metric-card card-amber"><h3>🟡 Medium Congestion</h3><h1>{n_medium}</h1></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f'<div class="metric-card card-green"><h3>🟢 Low Congestion</h3><h1>{n_low}</h1></div>',
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f'<div class="metric-card card-blue"><h3>👮 Officers Deployed</h3><h1>{total_assigned}/{sim_officers}</h1></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Map + Detail columns
# ---------------------------------------------------------------------------

map_col, detail_col = st.columns([1.2, 1])

# --- MAP ---
with map_col:
    st.markdown("### 🗺️ Bengaluru Road Network")

    # Build map with selected tile style
    tile_cfg = get_map_config(map_style)
    map_kwargs = {
        "location": [12.94, 77.63],
        "zoom_start": 12,
        "tiles": tile_cfg["tiles"],
    }
    if tile_cfg["attr"]:
        map_kwargs["attr"] = tile_cfg["attr"]
    m = folium.Map(**map_kwargs)

    color_map = {0: "green", 1: "orange", 2: "red"}
    icon_map = {0: "ok-sign", 1: "warning-sign", 2: "exclamation-sign"}

    for _, row in pred_df.iterrows():
        road = row["Road/Intersection Name"]
        level = predicted_congestion.get(road, 0)
        label = CONGESTION_MAP_INV.get(level, "Medium")

        # Get decision info
        dec_row = decisions_df[decisions_df["Road"] == road]
        officers = dec_row["Officers_Assigned"].values[0] if len(dec_row) else 0
        priority = dec_row["Priority_Label"].values[0] if len(dec_row) else "--"

        popup_html = f"""
        <div style="font-family: sans-serif; min-width: 180px;">
            <b>{road}</b><br>
            <span style="color: {color_map[level]};">● {label} Congestion</span><br>
            Priority: {priority}<br>
            Officers: {officers}<br>
            Trend: {forecast_trends.get(road, 'Stable')}
        </div>
        """

        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{road} -- {label}",
            icon=folium.Icon(
                color=color_map[level],
                icon=icon_map[level],
                prefix="glyphicon",
            ),
        ).add_to(m)

    st_folium(m, width=None, height=480, returned_objects=[])

# --- DETAIL PANEL ---
with detail_col:
    st.markdown("### 📋 Road Details")

    selected_road = st.selectbox(
        "Select a road",
        roads,
        index=0,
        label_visibility="collapsed",
    )

    # Get prediction for selected road
    sel_pred = pred_df[pred_df["Road/Intersection Name"] == selected_road].iloc[0]
    sel_level = predicted_congestion.get(selected_road, 0)
    sel_label = CONGESTION_MAP_INV.get(sel_level, "Medium")
    sel_confidence = sel_pred.get(f"Prob_{sel_label}", 0)

    # Get decision for selected road
    sel_dec = decisions_df[decisions_df["Road"] == selected_road]
    if len(sel_dec):
        sel_dec = sel_dec.iloc[0]
    else:
        sel_dec = None

    # Congestion badge
    badge_colors = {"Low": "#38ef7d", "Medium": "#ffd200", "High": "#ee5a24"}
    st.markdown(
        f"""
        <div style="background: {badge_colors.get(sel_label, '#667eea')};
                    color: {'white' if sel_label == 'High' else '#1a1a2e'};
                    padding: 1rem; border-radius: 10px; text-align: center;
                    margin-bottom: 1rem;">
            <h2 style="margin:0;">{sel_label} Congestion</h2>
            <p style="margin:0.3rem 0 0 0; opacity: 0.85;">
                Confidence: {sel_confidence:.0%}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Forecast
    fc = forecast_results.get(selected_road, [])
    trend = forecast_trends.get(selected_road, "Stable")
    trend_emoji = {"Worsening": "📈", "Stable": "➡️", "Improving": "📉"}

    st.markdown(f"**Forecast:** {trend_emoji.get(trend, '➡️')} {trend}")
    if fc:
        fc_df = pd.DataFrame(fc)
        fc_df.columns = ["Period Ahead", "Predicted Level", "Score"]
        st.dataframe(fc_df, width="stretch", hide_index=True)

    st.markdown("---")

    # Decision cards
    if sel_dec is not None:
        priority_class = sel_dec["Priority_Label"].lower()

        st.markdown(
            f"""
            <div class="decision-card {priority_class}">
                <strong>🚔 Personnel Allocation</strong><br>
                Assign <b>{sel_dec['Officers_Assigned']}</b> officers to this road
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="decision-card {priority_class}">
                <strong>[!]️ Priority Level</strong><br>
                <b>{sel_dec['Priority_Label']}</b>
                (score: {sel_dec['Priority_Score']:.3f})
            </div>
            """,
            unsafe_allow_html=True,
        )

        diversion = sel_dec["Suggested_Diversion"]
        diversion_text = (
            f"Suggest rerouting via <b>{diversion}</b>"
            if diversion != "--"
            else "No diversion needed"
        )
        st.markdown(
            f"""
            <div class="decision-card {priority_class}">
                <strong>🔀 Diversion</strong><br>
                {diversion_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Historical incident rate
    st.markdown(
        f"**📊 Historical Incident Rate:** "
        f"{sel_pred.get('Historical Incident Rate', 'N/A'):.0f} avg. accidents/year"
    )

# ---------------------------------------------------------------------------
# Full summary table
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### 📊 All Roads -- Summary")

summary = decisions_df.copy()
summary["Congestion"] = summary["Congestion_Level"].map(CONGESTION_MAP_INV)
summary["Trend"] = summary["Road"].map(forecast_trends)

display_cols = [
    "Road", "Congestion", "Priority_Label", "Officers_Assigned",
    "Trend", "Suggested_Diversion", "Priority_Score",
]
display_df = summary[[c for c in display_cols if c in summary.columns]]
display_df.columns = [
    "Road", "Congestion", "Priority", "Officers",
    "Trend", "Diversion", "Score",
]

# Color-code the dataframe
st.dataframe(
    display_df,
    width="stretch",
    hide_index=True,
    column_config={
        "Score": st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=1, format="%.3f",
        ),
    },
)
