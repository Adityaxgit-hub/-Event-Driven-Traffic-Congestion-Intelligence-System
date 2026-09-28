# 🚦 Event-Driven Traffic Congestion Intelligence System

A prototype system that predicts traffic congestion in Bengaluru and generates operational recommendations — officer deployment, priority scoring, and diversion suggestions.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample data (or place real CSVs in data/raw/)
python -m src.generate_sample_data

# 3. Run data preparation & feature engineering
python -m src.data_prep

# 4. Train the classifier
python -m src.classifier

# 5. Train the forecaster
python -m src.forecaster

# 6. (Optional) Test the decision layer standalone
python -m src.decision_layer

# 7. Launch the dashboard
streamlit run dashboard/app.py
```

## Pipeline

| Step | Module | What it does |
|------|--------|--------------|
| 1–2 | `src/data_prep.py` | Clean data, merge BTP incident rates, engineer features |
| 3 | `src/classifier.py` | CatBoost classifier → Low / Medium / High congestion |
| 4 | `src/forecaster.py` | LightGBM lag-feature regressor → directional forecast |
| 5 | `src/decision_layer.py` | PuLP optimization + priority scoring + diversion logic |
| — | `dashboard/app.py` | Streamlit app tying it all together |

## Data

- **Bangalore City Traffic Dataset (Kaggle)** — place as `data/raw/bangalore_traffic.csv`
- **BTP Accident Data (OpenCity.in)** — place as `data/raw/btp_accidents.csv`
- Or run `python -m src.generate_sample_data` for synthetic data

## Simplifying Assumptions

- Coordinates are hardcoded approximations (`data/coordinates.json`)
- Manpower pool is simulated (default: 50 officers)
- No real-time data; the dashboard simulates conditions via what-if controls
- Diversion uses straight-line distance, not road-network routing

These are stated assumptions for a prototype, not hidden limitations.
