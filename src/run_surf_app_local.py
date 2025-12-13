import pandas as pd
from fetch_data import fetch_data_wrapper
from process_data import process_data_wrapper
import yaml
from pathlib import Path


# Use this file to run pickle the surf forecast locally

with open("../config/surf_config.yaml", "r") as f:
    config = yaml.safe_load(f)


def load_forecast():
    raw = fetch_data_wrapper(config["data_sources"])
    df = process_data_wrapper(raw, config)
    return df

forecast_df = load_forecast()

# Resolve repo root and store df in .pkl file
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

forecast_df.to_pickle(DATA_DIR / "forecast_df.pkl")