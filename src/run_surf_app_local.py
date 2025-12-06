# IMPORTS
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
from fetch_data import *
from process_data import *
import yaml

pacific = ZoneInfo("America/Los_Angeles")

with open("../config/surf_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# FETCH ALL DATA

# 1. Fetch WW3 Swell Forecast
print('fetching ww3 swell forecast')
swell_forecast_df = fetch_ww3_timeseries(config['location_lat'], config['location_lon'], start_deg=0.3, step=0.05, max_deg=2.0)

# 2. Fetch CDIP actuals
print('fetching clip 029 actuals')
cdip_3h = fetch_cdip_029()

# 3. Fetch tide predictions
print('fetching tides')
tide_df = fetch_tide_predictions(config['tide_station'])

# 4. Fetch wind forecast
print('fetching wind')
wind_df = fetch_wind_forecast(config['location_lat'], config['location_lon'])

# 5. Fetch sunrise/sunset times
print('fetching sunrise/sunset')
sun_df = fetch_sunrise_sunset(config['location_lat'], config['location_lon'])


# PROCESS AND MERGE DATA 

# 1. Calibrate WW3 Forecasts with CDIP 029 Actuals
print('calibrating swell')
calibrated_swell_forecast_df, a, b, lag_steps, combo = calibrate_ww3_to_cdip029(swell_forecast_df, cdip_3h)

# 2. Create forecast timeline
print('creating forecast dataframe')
start_time = datetime.now(pacific).replace(minute=0, second=0, microsecond=0)
forecast_index = pd.date_range(start=start_time, periods=config['forecast_hours'], freq="h", tz=pacific).tz_localize(None)
forecast_df = pd.DataFrame(index=forecast_index)

# 3. Join swell, tide, and wind 
forecast_df = forecast_df.join(calibrated_swell_forecast_df, how="left")
forecast_df = forecast_df.join(tide_df[["tide_height"]], how="left")
forecast_df = forecast_df.join(wind_df[["wind_speed", "wind_direction"]], how="left")

# 4. Join sunrise/sunset 
forecast_df["date"] = forecast_df.index.date
forecast_df = forecast_df.join(
    sun_df[["first_light", "last_light"]],
    on="date",
    how="left")

# 5. Daylight flag
forecast_df["is_daylight"] = (
    (forecast_df.index >= forecast_df["first_light"]) &
    (forecast_df.index <= forecast_df["last_light"]))

# 6. Clean - forward-fill small gaps, keep non-null rows
forecast_df = forecast_df.ffill()
forecast_df = forecast_df.dropna()


# CALCULATE SURF SCORES AND HEIGHTS
print('calculating surf scores and heights')
forecast_df["surf_score"] = forecast_df.apply(
    lambda row: calculate_surf_score(
        row["Hs_029_pred_m"],
        row["Tp_s"],
        row["Dir_deg"],
        row["wind_speed"],
        row["wind_direction"],
        row["tide_height"]),
    axis=1)

surf_heights = forecast_df.apply(
    lambda row: predict_bolinas_surf_height(
        row["Hs_029_pred_m"],
        row["Tp_s"],
        row["Dir_deg"]),
    axis=1)

# Concatenate surf heights with df
surf_heights_df = pd.DataFrame.from_records(surf_heights)
surf_heights_df.index = forecast_df.index
forecast_df = pd.concat([forecast_df, surf_heights_df], axis=1)

# Clean final dataframe columns
forecast_df = forecast_df[['surf_score', 'swell_quality', 'bolinas_surf_min_ft', 
			   'bolinas_surf_max_ft', 'wind_speed', 'wind_direction',
			   'tide_height', 'Hs_029_pred_m', 'Tp_s', 'Dir_deg', 'is_daylight']]
forecast_df.rename(columns={'surf_score': 'Surf Score (1-10)', 
			    'swell_quality': 'Swell Quality (0-1)',
			    'bolinas_surf_min_ft': 'Surf Height Min (ft)',
		            'bolinas_surf_max_ft': 'Surf Height Max (ft)',
			    'wind_speed': 'Wind Speed (MPH)',
			    'wind_direction': 'Wind Direction (Degrees)',
			    'tide_height': 'Tide Height (ft)',
		            'Hs_029_pred_m': 'Swell Size (m)',
  			    'Tp_s': 'Swell Period (Seconds)',
			    'Dir_deg': 'Swell Direction (Degrees)'}, inplace=True)

# Pickle dataframe for streamlit
print('saving df as .pkl file')
forecast_df.to_pickle("forecast_df.pkl")