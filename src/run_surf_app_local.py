import pandas as pd
from fetch_data import fetch_data_wrapper
from process_data import process_data_wrapper
import yaml

# Use this file to run pickle the surf forecast locally

with open("../config/surf_config.yaml", "r") as f:
    config = yaml.safe_load(f)


def load_forecast():
    raw = fetch_data_wrapper(config["data_sources"])
    df = process_data_wrapper(raw, config)
    return df

forecast_df = load_forecast()
forecast_df.to_pickle("forecast_df.pkl")
