# Bolinas Surf Forecast

A fully local, configurable surf-forecasting pipeline for Bolinas, CA. Fetches real environmental data (WW3, CDIP, NOAA tides, NWS winds, sunrise/sunset), calibrates offshore swell to local break behavior, computes surf scores, and displays everything in an interactive Streamlit UI.

## Features

- Deep-water WW3 swell forecast ingestion
- CDIP buoy calibration (Point Bonita 029)
- NWS wind forecasts
- Sunrise, sunset, civil twilight
- NOAA tide predictions (MLLW)
- Surf-score model for Bolinas
- Interactive UI via Streamlit
- Configurable through YAML
- All data stored locally (`.pkl`) — no large files committed

## Installation

```bash
git clone https://github.com/<your-username>/bolinas-surf-forecast.git
cd bolinas-surf-forecast
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

All settings live in:

```yaml
config/surf_config.yaml
```

Includes:
- Latitude / longitude
- NOAA tide station
- CDIP buoy ID
- Forecast horizon
- Processing options

Modify this file to customize the pipeline or adapt to another break.

## Usage

### 1. Generate/refresh forecast data

```bash
python src/run_surf_app.py
```

This creates:

```
src/forecast_df.pkl
```

(ignored by `.gitignore`)

### 2. Launch the Streamlit UI

```bash
streamlit run src/surf_app_streamlit.py
```

Then open:

```
http://localhost:8501
```

## Repository Structure

```
bolinas-surf-forecast/
├── config/
│   └── surf_config.yaml
│
├── src/
│   ├── fetch_data.py
│   ├── process_data.py
│   ├── surf_app_streamlit.py
│   ├── run_surf_app.py
│   └── __init__.py
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## License

MIT License — See the `LICENSE` file for details.