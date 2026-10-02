"""
API configuration -- loads keys from .env file in the project root.

Paste your keys in the .env file:
    CARTO_API_KEY=your_key_here
"""

import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# CARTO Basemaps
# ---------------------------------------------------------------------------
CARTO_API_KEY = os.environ.get("CARTO_API_KEY", "")

# Available map styles (OpenStreetMap always works, CARTO needs key)
MAP_STYLES = {
    "OpenStreetMap": {
        "tiles": "OpenStreetMap",
        "attr": None,  # Folium handles attribution for built-in tiles
        "needs_key": False,
    },
    "CARTO Voyager": {
        "tiles": "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        "attr": '&copy; <a href="https://carto.com/">CARTO</a> &copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
        "needs_key": True,
    },
    "CARTO Positron (Light)": {
        "tiles": "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        "attr": '&copy; <a href="https://carto.com/">CARTO</a> &copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
        "needs_key": True,
    },
    "CARTO Dark Matter": {
        "tiles": "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        "attr": '&copy; <a href="https://carto.com/">CARTO</a> &copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
        "needs_key": True,
    },
}

DEFAULT_MAP_STYLE = "OpenStreetMap"


def get_map_config(style_name: str = DEFAULT_MAP_STYLE) -> dict:
    """
    Get Folium-compatible map tile config for the selected style.
    Returns dict with 'tiles' and 'attr' keys.

    For CARTO styles, appends the API key if available.
    """
    style = MAP_STYLES.get(style_name, MAP_STYLES[DEFAULT_MAP_STYLE])
    tiles = style["tiles"]
    attr = style["attr"]

    # Append API key for CARTO tiles if available
    if style["needs_key"] and CARTO_API_KEY:
        tiles = tiles + f"?api_key={CARTO_API_KEY}"

    return {"tiles": tiles, "attr": attr}
