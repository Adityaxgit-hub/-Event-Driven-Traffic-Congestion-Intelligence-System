"""
Streamlit Dashboard -- Bengaluru Traffic Congestion Intelligence System

Real-time commuter portal for traffic prediction, congestion forecasting,
and intelligent alternate route recommendations across Bengaluru.

Usage:
    streamlit run dashboard/app.py
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import Geocoder
from streamlit_folium import st_folium

from src.utils import (
    CONGESTION_MAP,
    CONGESTION_MAP_INV,
    CONGESTION_LEVELS,
    WEATHER_CONDITIONS,
    load_coordinates,
    load_combined_data,
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
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bengaluru Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS -- Premium modern aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Clean background & typography */
    .stApp {
        background-color: #f7f9fc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Main header banner */
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 0.98rem;
        margin-bottom: 1.2rem;
        font-weight: 500;
    }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        padding: 1.1rem 1.2rem;
        border-radius: 12px;
        text-align: left;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
        border: 1px solid #e2e8f0;
        border-left: 5px solid #3b82f6;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
    }
    .metric-card h1 {
        margin: 0.35rem 0 0 0;
        font-size: 1.9rem;
        color: #0f172a;
        font-weight: 700;
    }

    .card-red { border-left-color: #ef4444; }
    .card-red h1 { color: #dc2626; }
    .card-amber { border-left-color: #f59e0b; }
    .card-amber h1 { color: #d97706; }
    .card-green { border-left-color: #10b981; }
    .card-green h1 { color: #059669; }
    .card-blue { border-left-color: #3b82f6; }
    .card-blue h1 { color: #2563eb; }

    /* Commuter Recommendation Cards */
    .recommendation-card {
        background: #ffffff;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.2rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.85rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        border: 1px solid #e2e8f0;
        border-left-width: 4px;
    }
    .recommendation-card.high {
        border-left-color: #ef4444;
        background: #fffafa;
    }
    .recommendation-card.medium {
        border-left-color: #f59e0b;
        background: #fffdfa;
    }
    .recommendation-card.low {
        border-left-color: #10b981;
        background: #fbfdfc;
    }

    /* Badge banner */
    .status-badge {
        padding: 1.1rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .status-badge.high {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        color: #ffffff;
    }
    .status-badge.medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #ffffff;
    }
    .status-badge.low {
        background: linear-gradient(135deg, #10b981 0%, #047857 100%);
        color: #ffffff;
    }

    /* Area summary pills */
    .area-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
        background: #e2e8f0;
        color: #334155;
    }

    /* Hide standard Streamlit header and footer */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data & model loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def get_data():
    return load_combined_data()

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
# Initial loading & error handling
# ---------------------------------------------------------------------------

try:
    df = get_data()
    clf_model = get_classifier()
    fcast_model = get_forecaster()
    coords = get_coordinates()
    data_loaded = True
except FileNotFoundError as e:
    data_loaded = False
    st.error(f"⚠️ {e}")
    st.info(
        "Please ensure the data and models are initialized:\n"
        "```\n"
        "python -m src.data_prep\n"
        "python -m src.classifier\n"
        "python -m src.forecaster\n"
        "```"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Major corridors & area index setup
# ---------------------------------------------------------------------------

all_valid_roads = sorted(set(df["Road/Intersection Name"].unique()) & set(coords["roads"].keys()))
all_areas = sorted(list(coords["areas"].keys()))

MAJOR_ARTERIAL_ROADS = [
    "100 Feet Road", "CMH Road", "Sony World Junction", "Sarjapur Road",
    "Silk Board Junction", "Hosur Road", "Hebbal Flyover", "Ballari Road",
    "Marathahalli Bridge", "ITPL Main Road", "Jayanagar 4th Block",
    "South End Circle", "Trinity Circle", "Anil Kumble Circle",
    "Tumkur Road", "Yeshwanthpur Circle",
]


# ---------------------------------------------------------------------------
# Sidebar -- Exploration & What-if controls
# ---------------------------------------------------------------------------

st.sidebar.markdown("## 🗺️ Navigation & Area")

area_options = ["🌐 All Bengaluru (City Overview)"] + all_areas
selected_area_choice = st.sidebar.selectbox(
    "📍 Select Region / Area",
    area_options,
    index=0,
    help="Filter by Bengaluru region or view the citywide network",
)

is_overview = (selected_area_choice == "🌐 All Bengaluru (City Overview)")
selected_area = None if is_overview else selected_area_choice

# Search bar for ANY road in Bengaluru
search_road = st.sidebar.selectbox(
    "🔍 Search Any Road / Junction",
    ["-- None (Use Area View) --"] + all_valid_roads,
    index=0,
    help="Search and jump directly to any monitored road in Bengaluru",
)
has_searched_road = (search_road != "-- None (Use Area View) --")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎛️ Scenario Simulation")
st.sidebar.caption("Simulate traffic conditions dynamically:")

today_date = datetime.date.today()
sim_date = st.sidebar.date_input(
    "📅 Date",
    value=today_date,
    min_value=df["Date"].min().date(),
    max_value=today_date + datetime.timedelta(days=365),
)
sim_hour = st.sidebar.slider("🕐 Time of Day (Hour)", 0, 23, 9, format="%02d:00")
sim_weather = st.sidebar.selectbox("🌤️ Weather Condition", WEATHER_CONDITIONS, index=0)
sim_roadwork = st.sidebar.checkbox("🚧 Roadwork / Construction Active", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🗺️ Map Settings")
map_style = st.sidebar.selectbox(
    "Map Style",
    list(MAP_STYLES.keys()),
    index=list(MAP_STYLES.keys()).index(DEFAULT_MAP_STYLE),
)

if CARTO_API_KEY:
    st.sidebar.success("CARTO vector map loaded", icon="🔑")
else:
    st.sidebar.caption("Using free OpenStreetMap / CARTO tiles.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Bengaluru Commuter Intelligence Portal · Powered by Machine Learning</small>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Determine active roads to process
# ---------------------------------------------------------------------------

if is_overview:
    # Use top arterial corridors present in valid dataset
    curated = [r for r in MAJOR_ARTERIAL_ROADS if r in all_valid_roads]
    if len(curated) < 16:
        # Fill with top roads by frequency
        top_freq = [r for r in df["Road/Intersection Name"].value_counts().index if r in all_valid_roads][:25]
        active_roads = list(dict.fromkeys(curated + top_freq))[:25]
    else:
        active_roads = curated
else:
    # Filter roads for the selected area
    area_roads = [r for r in all_valid_roads if coords["roads"].get(r, {}).get("area") == selected_area]
    if len(area_roads) > 40:
        # Prioritize top 40 roads by volume/frequency
        area_subset = df[df["Road/Intersection Name"].isin(area_roads)]
        top_in_area = [r for r in area_subset["Road/Intersection Name"].value_counts().index if r in area_roads][:40]
        active_roads = top_in_area
    elif len(area_roads) > 0:
        active_roads = area_roads
    else:
        active_roads = [r for r in MAJOR_ARTERIAL_ROADS if r in all_valid_roads][:10]

# If the user specifically searched for a road, ensure it's in active_roads
if has_searched_road and search_road not in active_roads:
    active_roads = [search_road] + active_roads


# ---------------------------------------------------------------------------
# High-speed Scenario Builder (Optimized O(1) dictionary lookups)
# ---------------------------------------------------------------------------

def get_diurnal_multipliers(hour: int) -> tuple[float, float, float, float]:
    """Returns (volume_mult, speed_mult, capacity_mult, tti_mult) for a given hour."""
    if 8 <= hour <= 10:       # Morning Peak
        return 1.45, 0.65, 1.40, 1.40
    elif 17 <= hour <= 20:    # Evening Peak
        return 1.65, 0.55, 1.55, 1.55
    elif 11 <= hour <= 16:    # Midday Off-Peak
        return 1.00, 0.90, 1.00, 1.00
    elif 6 <= hour <= 7 or 21 <= hour <= 22:  # Shoulder Hours
        return 0.75, 1.15, 0.75, 0.85
    else:                     # Night
        return 0.30, 1.50, 0.30, 0.70


def build_scenario_fast(
    df: pd.DataFrame,
    roads_to_build: list[str],
    sim_date: datetime.date,
    sim_hour: int,
    sim_weather: str,
    sim_roadwork: bool,
    coords: dict,
) -> pd.DataFrame:
    """Build scenario rows for active roads in milliseconds using indexed lookups."""
    selected_dow = pd.Timestamp(sim_date).dayofweek
    date_mask = df["Date"].dt.date == sim_date

    # Fast filter only the needed roads
    subset = df[df["Road/Intersection Name"].isin(roads_to_build)]
    if len(subset) == 0:
        subset = df

    if date_mask.sum() > 0:
        match_df = subset[subset["Date"].dt.date == sim_date]
        if len(match_df) > 0:
            base_dict = match_df.groupby("Road/Intersection Name").mean(numeric_only=True).to_dict("index")
        else:
            base_dict = subset.groupby("Road/Intersection Name").mean(numeric_only=True).to_dict("index")
    else:
        dow_mask = (subset["Day_of_Week"] == selected_dow) & (subset["Hour"] == sim_hour)
        match_df = subset[dow_mask]
        if len(match_df) > 0:
            base_dict = match_df.groupby("Road/Intersection Name").mean(numeric_only=True).to_dict("index")
        else:
            dow_only_mask = (subset["Day_of_Week"] == selected_dow)
            match_df = subset[dow_only_mask]
            if len(match_df) > 0:
                base_dict = match_df.groupby("Road/Intersection Name").mean(numeric_only=True).to_dict("index")
            else:
                base_dict = subset.groupby("Road/Intersection Name").mean(numeric_only=True).to_dict("index")

    global_avg = {
        "Traffic Volume": 3200,
        "Average Speed (km/h)": 28.0,
        "Travel Time Index": 1.55,
        "Road Capacity Utilization (%)": 65.0,
        "Pedestrian & Cyclist Count": 180,
        "Incident Reports": 1,
        "Historical Incident Rate": 200.0,
        "Rolling_Congestion_3": 1.2,
        "Congestion_Lag_1": 1.0,
        "Congestion_Lag_2": 1.0,
        "Congestion_Lag_3": 1.0,
    }

    vol_m, speed_m, cap_m, tti_m = get_diurnal_multipliers(sim_hour)
    weather_severity = {"Clear": 0, "Fog": 1, "Rain": 2, "Heavy Rain": 3}

    rows = []
    for road in roads_to_build:
        r_meta = coords["roads"].get(road, {"area": "Other", "lat": 12.9716, "lon": 77.5946})
        road_base = base_dict.get(road, global_avg)

        base_vol = road_base.get("Traffic Volume", 3200)
        base_speed = road_base.get("Average Speed (km/h)", 28.0)
        base_cap = road_base.get("Road Capacity Utilization (%)", 65.0)
        base_tti = road_base.get("Travel Time Index", 1.55)

        sim_vol = int(base_vol * vol_m)
        sim_speed = round(max(5.0, base_speed * speed_m), 1)
        sim_cap = round(min(100.0, max(5.0, base_cap * cap_m)), 1)
        sim_tti = round(max(1.0, base_tti * tti_m), 2)

        row = {
            "Road/Intersection Name": road,
            "Area Name": r_meta.get("area", "Other"),
            "Date": pd.Timestamp(sim_date),
            "Hour": sim_hour,
            "Day_of_Week": selected_dow,
            "Is_Weekend": int(selected_dow >= 5),
            "Month": pd.Timestamp(sim_date).month,
            "Weather Conditions": sim_weather,
            "Weather_Severity": weather_severity.get(sim_weather, 0),
            "Is_Roadwork": int(sim_roadwork),
            "Traffic Volume": sim_vol,
            "Average Speed (km/h)": sim_speed,
            "Travel Time Index": sim_tti,
            "Road Capacity Utilization (%)": sim_cap,
            "Pedestrian & Cyclist Count": int(road_base.get("Pedestrian & Cyclist Count", 180)),
            "Incident Reports": road_base.get("Incident Reports", 1),
            "Historical Incident Rate": road_base.get("Historical Incident Rate", 200.0),
            "Rolling_Congestion_3": road_base.get("Rolling_Congestion_3", 1.2),
            "Congestion_Lag_1": road_base.get("Congestion_Lag_1", 1.0),
            "Congestion_Lag_2": road_base.get("Congestion_Lag_2", 1.0),
            "Congestion_Lag_3": road_base.get("Congestion_Lag_3", 1.0),
            "Weather_Hour": weather_severity.get(sim_weather, 0) * sim_hour,
            "Roadwork_Volume": int(sim_roadwork) * sim_vol,
            "Latitude": r_meta.get("lat", 12.9716),
            "Longitude": r_meta.get("lon", 77.5946),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run ML Pipeline for active roads
# ---------------------------------------------------------------------------

scenario_df = build_scenario_fast(
    df=df,
    roads_to_build=active_roads,
    sim_date=sim_date,
    sim_hour=sim_hour,
    sim_weather=sim_weather,
    sim_roadwork=sim_roadwork,
    coords=coords,
)

pred_df = predict_congestion(clf_model, scenario_df)
pred_df["Predicted_Congestion"] = pred_df["Predicted_Congestion"].astype(str).str.strip()
pred_df["Predicted_Congestion_Ordinal"] = (
    pred_df["Predicted_Congestion"].map(CONGESTION_MAP).fillna(1).astype(int)
)

predicted_congestion = dict(zip(pred_df["Road/Intersection Name"], pred_df["Predicted_Congestion_Ordinal"]))
incident_rates = dict(zip(pred_df["Road/Intersection Name"], pred_df["Historical Incident Rate"].fillna(0)))

# Autoregressive 3-step forecast per active road
forecast_results = {}
forecast_trends = {}
for _, row in pred_df.iterrows():
    r_name = row["Road/Intersection Name"]
    fc = forecast_congestion(fcast_model, pd.DataFrame([row]), steps_ahead=3)
    forecast_results[r_name] = fc
    forecast_trends[r_name] = forecast_trend(fc)

# Commuter route decision & alternate suggestions
decisions_df = generate_decisions(
    roads=active_roads,
    predicted_congestion=predicted_congestion,
    incident_rates=incident_rates,
    forecast_trends=forecast_trends,
)


# ---------------------------------------------------------------------------
# Dashboard Header
# ---------------------------------------------------------------------------

view_title = f"{selected_area_choice}" if is_overview else f"Region: {selected_area}"
st.markdown(
    f'<div class="main-header">🚦 Bengaluru Traffic Intelligence <span style="font-size:1.1rem; color:#64748b; font-weight:500;">· {view_title}</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="sub-header">Live Simulation for {sim_date.strftime("%d %B %Y")} · '
    f'{sim_hour:02d}:00 Hrs · Weather: {sim_weather}'
    f'{" · 🚧 Active Roadwork Active" if sim_roadwork else ""}</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Commuter Metric Cards
# ---------------------------------------------------------------------------

n_high = sum(1 for v in predicted_congestion.values() if v == 2)
n_medium = sum(1 for v in predicted_congestion.values() if v == 1)
n_low = sum(1 for v in predicted_congestion.values() if v == 0)
n_diversions = sum(1 for d in decisions_df["Suggested_Diversion"] if d != "--")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f'<div class="metric-card card-red"><h3>🔴 Heavy Bottlenecks</h3><h1>{n_high}</h1></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="metric-card card-amber"><h3>🟡 Moderate Traffic</h3><h1>{n_medium}</h1></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f'<div class="metric-card card-green"><h3>🟢 Clear Routes</h3><h1>{n_low}</h1></div>',
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f'<div class="metric-card card-blue"><h3>🔀 Alternate Routes</h3><h1>{n_diversions}</h1></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Map & Road Detail Columns
# ---------------------------------------------------------------------------

map_col, detail_col = st.columns([1.25, 0.95])

# --- MAP VIEW ---
with map_col:
    map_header_col1, map_header_col2 = st.columns([2, 1])
    with map_header_col1:
        st.markdown("### 🗺️ Live Traffic Map")
    with map_header_col2:
        # Quick road picker from active roads
        current_focus = search_road if has_searched_road else active_roads[0]
        focus_road = st.selectbox(
            "Highlight on Map",
            active_roads,
            index=active_roads.index(current_focus) if current_focus in active_roads else 0,
            key="map_focus_selector",
            label_visibility="collapsed",
        )

    # Determine map center and zoom level
    if has_searched_road and search_road in coords["roads"]:
        map_center = [coords["roads"][search_road]["lat"], coords["roads"][search_road]["lon"]]
        map_zoom = 15
    elif not is_overview and selected_area in coords["areas"]:
        map_center = [coords["areas"][selected_area]["lat"], coords["areas"][selected_area]["lon"]]
        map_zoom = 13
    else:
        map_center = [12.9716, 77.5946]
        map_zoom = 12

    # Initialize Folium Map
    tile_cfg = get_map_config(map_style)
    map_kwargs = {
        "location": map_center,
        "zoom_start": map_zoom,
        "tiles": tile_cfg["tiles"],
    }
    if tile_cfg["attr"]:
        map_kwargs["attr"] = tile_cfg["attr"]
    m = folium.Map(**map_kwargs)

    # Embedded search geocoder
    Geocoder(position="topleft", add_marker=True).add_to(m)

    color_map = {0: "green", 1: "orange", 2: "red"}
    hex_color_map = {0: "#10b981", 1: "#f59e0b", 2: "#ef4444"}
    icon_map = {0: "ok-sign", 1: "warning-sign", 2: "exclamation-sign"}

    # If in Overview mode, draw the 14 Area Hub Badges
    if is_overview:
        for area_name, a_pos in coords["areas"].items():
            # Check how many roads in this area are high congestion
            hub_popup = f"""
            <div style="font-family: sans-serif; font-size: 13px; line-height: 1.4;">
                <b style="font-size: 14px;">📍 {area_name} Hub</b><br>
                <span>Select <b>{area_name}</b> in sidebar for street-by-street view.</span>
            </div>
            """
            folium.CircleMarker(
                location=[a_pos["lat"], a_pos["lon"]],
                radius=14,
                color="#6366f1",
                fill=True,
                fill_color="#818cf8",
                fill_opacity=0.35,
                weight=2,
                tooltip=f"📍 {area_name} Area Hub",
                popup=folium.Popup(hub_popup, max_width=220),
            ).add_to(m)

    # Plot Active Road Markers
    for _, row in pred_df.iterrows():
        road = row["Road/Intersection Name"]
        level = predicted_congestion.get(road, 1)
        label = CONGESTION_MAP_INV.get(level, "Medium")

        dec_row = decisions_df[decisions_df["Road"] == road]
        severity = dec_row["Priority_Label"].values[0] if len(dec_row) else "--"
        diversion = dec_row["Suggested_Diversion"].values[0] if len(dec_row) else "--"

        alt_note = f"<b>Alternate:</b> {diversion}" if diversion != "--" else "<i>Clear flow — No alternate needed</i>"

        popup_html = f"""
        <div style="font-family: sans-serif; min-width: 190px; font-size: 13px; line-height: 1.45;">
            <b style="font-size: 14px; color: #0f172a;">{road}</b><br>
            <span style="color: {hex_color_map[level]}; font-weight: bold;">● {label} Congestion</span><br>
            <b>Avg Speed:</b> {row['Average Speed (km/h)']:.1f} km/h<br>
            <b>Delay Index:</b> {row['Travel Time Index']:.2f}x<br>
            {alt_note}<br>
            <b>Trend:</b> {forecast_trends.get(road, 'Stable')}
        </div>
        """

        is_focused = (road == focus_road)

        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=folium.Popup(popup_html, max_width=260, show=is_focused),
            tooltip=f"{'🔍 ' if is_focused else ''}{road} — {label}",
            icon=folium.Icon(
                color="blue" if is_focused else color_map[level],
                icon="search" if is_focused else icon_map[level],
                prefix="glyphicon",
            ),
        ).add_to(m)

        # Highlight ring for currently focused road
        if is_focused:
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=20,
                color="#2563eb",
                fill=True,
                fill_color="#3b82f6",
                fill_opacity=0.3,
                weight=3,
            ).add_to(m)

    st_folium(m, width=None, height=500, returned_objects=[])


# --- DETAIL & RECOMMENDATION PANEL ---
with detail_col:
    st.markdown("### 📋 Road & Route Recommendation")

    # Road selector
    selected_road = st.selectbox(
        "Select Road for Detailed Forecast",
        active_roads,
        index=active_roads.index(focus_road) if focus_road in active_roads else 0,
        label_visibility="collapsed",
    )

    sel_pred = pred_df[pred_df["Road/Intersection Name"] == selected_road].iloc[0]
    sel_level = predicted_congestion.get(selected_road, 1)
    sel_label = CONGESTION_MAP_INV.get(sel_level, "Medium")
    sel_conf = sel_pred.get(f"Prob_{sel_label}", 0.0)

    sel_dec = decisions_df[decisions_df["Road"] == selected_road]
    sel_dec = sel_dec.iloc[0] if len(sel_dec) else None

    # Status Badge
    badge_style = sel_label.lower()
    st.markdown(
        f"""
        <div class="status-badge {badge_style}">
            <h2 style="margin:0; font-size:1.75rem; font-weight:700;">{sel_label} Congestion</h2>
            <p style="margin:0.35rem 0 0 0; opacity: 0.92; font-size:0.95rem;">
                Confidence: <b>{sel_conf:.0%}</b> · Speed: <b>{sel_pred['Average Speed (km/h)']:.1f} km/h</b> · Delay: <b>{sel_pred['Travel Time Index']:.2f}x</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3-Step Forecast Table
    fc = forecast_results.get(selected_road, [])
    trend = forecast_trends.get(selected_road, "Stable")
    trend_emoji = {"Worsening": "📈", "Stable": "➡️", "Improving": "📉"}

    st.markdown(f"**Expected Trend:** {trend_emoji.get(trend, '➡️')} **{trend}**")
    if fc:
        fc_display = []
        time_labels = ["Next 30 mins", "Next 60 mins", "Next 90 mins"]
        for i, step_data in enumerate(fc):
            label_text = time_labels[i] if i < len(time_labels) else f"Step +{i+1}"
            fc_display.append({
                "Horizon": label_text,
                "Predicted Condition": step_data.get("predicted_level", "Medium"),
                "Congestion Score": f"{step_data.get('raw_score', 1.0):.2f}",
            })
        st.dataframe(pd.DataFrame(fc_display), width="stretch", hide_index=True)

    st.markdown("---")

    # Smart Commuter Guidance Cards
    if sel_dec is not None:
        severity_class = sel_dec["Priority_Label"].lower()
        diversion = sel_dec["Suggested_Diversion"]

        if diversion != "--":
            diversion_html = f"Consider taking <b>{diversion}</b> instead — offers lower congestion and reduced delay."
        else:
            diversion_html = "Current route has normal traffic flow. No alternate route needed."

        st.markdown(
            f"""
            <div class="recommendation-card {severity_class}">
                <strong>⚠️ Congestion Severity</strong><br>
                <b>{sel_dec['Priority_Label']}</b> (Priority index: {sel_dec['Priority_Score']:.3f})
            </div>
            <div class="recommendation-card {severity_class}">
                <strong>🔀 Commuter Alternate Route</strong><br>
                {diversion_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Historical Accident & Safety context
    hist_incidents = sel_pred.get("Historical Incident Rate", 0)
    safety_note = "High Caution" if hist_incidents > 250 else "Moderate Caution" if hist_incidents > 100 else "Standard Caution"
    st.markdown(
        f"**🛡️ Road Safety Context:** {hist_incidents:.0f} historical recorded incidents/yr ({safety_note})"
    )


# ---------------------------------------------------------------------------
# Commuter Route Summary Table
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### 📊 Active Monitored Corridors — Summary")

summary = decisions_df.copy()
summary["Congestion"] = summary["Congestion_Level"].map(CONGESTION_MAP_INV)
summary["Trend"] = summary["Road"].map(forecast_trends)

display_cols = [
    "Road", "Congestion", "Priority_Label",
    "Trend", "Suggested_Diversion", "Priority_Score",
]
display_df = summary[[c for c in display_cols if c in summary.columns]].copy()
display_df.columns = [
    "Road Name", "Congestion Status", "Severity",
    "Trend Outlook", "Alternate Route", "Congestion Score",
]

st.dataframe(
    display_df,
    width="stretch",
    hide_index=True,
    column_config={
        "Congestion Score": st.column_config.ProgressColumn(
            "Congestion Score", min_value=0, max_value=1, format="%.3f"
        ),
    },
)
