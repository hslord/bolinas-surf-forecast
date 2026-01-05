# Bolinas Surf Forecast

A fully local, configurable surf-forecasting pipeline for Bolinas, CA. Fetches real environmental data (WW3, NOAA tides, NWS winds, sunrise/sunset), calibrates offshore swell to local break behavior, computes surf scores, and displays everything in an interactive Streamlit UI.

## Features

- Deep-water WW3 swell forecast ingestion
- NWS wind forecasts
- Sunrise, sunset, civil twilight
- NOAA tide predictions (MLLW)
- Surf-score model for Bolinas
- Interactive UI via Streamlit
- Configurable through YAML

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
- Location latitude / longitude
- WW3 latitude / longitude
- NOAA tide station
- Forecast horizon
- Processing options

Modify this file to customize the pipeline or adapt to another break.

## Usage

### Run the Forecast Locally

```bash
python src/run_surf_app_local.py
```

### Launch the Streamlit UI to Visualize Outputs

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
│   ├── reference_functions.py
│   ├── surf_app_streamlit.py
│   ├── run_surf_app_local.py
│   └── __init__.py
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## License

MIT License — See the `LICENSE` file for details.