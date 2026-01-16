import pandas as pd
from fetch_data import fetch_data_wrapper
from process_data import process_data_wrapper
import yaml
from pathlib import Path
from reference_functions import status

# with open("../config/surf_config.yaml", "r") as f:
#     config = yaml.safe_load(f)


# def load_forecast():
#     # fetch data
#     raw = fetch_data_wrapper(config["data_sources"])

#     if raw["ww3"] is None:
#         status("CRITICAL: Swell data missing. Cannot generate forecast.")
#         return # Stop here if the core data is gone

#     df = process_data_wrapper(raw, config)
#     return df

# forecast_df = load_forecast()

# # Resolve repo root and store df in .parquet file
# BASE_DIR = Path(__file__).resolve().parents[1]
# DATA_DIR = BASE_DIR / "data"
# DATA_DIR.mkdir(exist_ok=True)

# forecast_df.to_parquet(DATA_DIR / "forecast_df.parquet")

# Use absolute pathing so it works dynamically
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config" / "surf_config.yaml"
DATA_DIR = BASE_DIR / "data"


def load_forecast():
    # Check if config exists before opening
    if not CONFIG_PATH.exists():
        status(f"CRITICAL: Config file not found at {CONFIG_PATH}")
        return None

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Fetch data using the correct config keys
    status("Starting Data Fetching...")
    raw = fetch_data_wrapper(config["data_sources"])

    # Handle the "None" case and abort
    if raw.get("ww3") is None:
        status("CRITICAL: Swell data (WW3) missing. Aborting run.")
        return None

    status("Running Surf Model Logic...")
    df = process_data_wrapper(raw, config)
    return df


# --- Execution Block ---
forecast_df = load_forecast()

if forecast_df is not None:
    DATA_DIR.mkdir(exist_ok=True)
    # Use the standardized Parquet format for the cloud
    output_path = DATA_DIR / "forecast_df.parquet"
    forecast_df.to_parquet(output_path)
    status(f"SUCCESS: Forecast saved to {output_path}")
else:
    status("FAILED: Forecast was not generated.")
