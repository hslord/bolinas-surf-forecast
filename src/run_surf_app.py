from fetch_data import fetch_data_wrapper
from process_data import process_data_wrapper
import yaml
from pathlib import Path
from reference_functions import status

# Use absolute pathing so it works dynamically
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config" / "surf_config.yaml"
DATA_DIR = BASE_DIR / "data"


def load_forecast():
    """
    Orchestrate the end-to-end data pipeline: from configuration loading
    and remote data fetching to final model processing.

    This function serves as the primary entry point for the Streamlit UI.
    It manages the high-level workflow, ensures configuration integrity,
    validates that critical wave data is present before proceeding,
    and coordinates the hand-off between the data acquisition and
    processing layers.

    Returns
    -------
    pandas.DataFrame or None
        A fully processed forecast DataFrame ready for UI rendering.
        Returns None if a critical failure occurs (e.g., missing config
        file or empty WW3 swell data), allowing the UI to fail gracefully.
    """
    # Check if config exists before opening
    if not CONFIG_PATH.exists():
        status(f"CRITICAL: Config file not found at {CONFIG_PATH}")
        return None

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Fetch data
    status("Starting Data Fetching...")
    raw = fetch_data_wrapper(config["data_sources"])

    # Handle the "None" case and abort
    if raw.get("ww3") is None:
        status("CRITICAL: Swell data (WW3) missing. Aborting run.")
        return None

    status("Running Surf Model Logic...")
    df = process_data_wrapper(raw, config)
    return df


forecast_df = load_forecast()

if forecast_df is not None:
    DATA_DIR.mkdir(exist_ok=True)
    # Use the standardized Parquet format for the cloud
    output_path = DATA_DIR / "forecast_df.parquet"
    forecast_df.to_parquet(output_path)
    status(f"SUCCESS: Forecast saved to {output_path}")
else:
    status("FAILED: Forecast was not generated.")
