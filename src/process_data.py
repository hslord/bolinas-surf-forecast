# IMPORTS
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import xarray as xr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import yaml

pacific = ZoneInfo("America/Los_Angeles")

with open("../config/surf_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# DATA PROCESSING

def calibrate_ww3_to_cdip029(df_ww3, cdip_3h):
    """
    Calibrate WW3 Hs to CDIP 029.
    Inputs:
        df_ww3 : DataFrame from fetch_ww3_timeseries(), DatetimeIndex
        cdip_3h: DataFrame from fetch_cdip_029(), DatetimeIndex

    Returns:
        df_cal : df_ww3 with added column 'Hs_029_pred_m'
        a, b   : regression coefficients (slope, intercept)
        lag_steps : best 3-hour lag applied
        combo : merged WW3/CDIP dataset after alignment
    """

    # Align CDIP onto WW3 timeline
    cdip_on_ww3 = cdip_3h.reindex(
        df_ww3.index,
        method="nearest",
        tolerance=pd.Timedelta("2h")
    )

    # Combine + keep rows containing both
    combo = pd.concat([df_ww3, cdip_on_ww3], axis=1).dropna()

    # 1. Find best lag (in 3-hour increments)

    def best_lag_steps(series_off, series_obs, max_steps=3):
        best_corr = -np.inf
        best_lag = 0
        for h in range(-max_steps, max_steps + 1):
            shifted = series_off.shift(h)
            valid = pd.concat([shifted, series_obs], axis=1).dropna()
            if len(valid) < 20:
                continue
            corr = valid.iloc[:,0].corr(valid.iloc[:,1])
            if corr > best_corr:
                best_corr = corr
                best_lag = h
        return best_lag, best_corr

    lag_steps, _ = best_lag_steps(combo["Hs_m"], combo["Hs_029_m"], max_steps=3)

    # Apply lag
    combo["Hs_lagged_m"] = combo["Hs_m"].shift(lag_steps)
    combo = combo.dropna(subset=["Hs_lagged_m", "Hs_029_m"])

    # 2. Linear Regression: CDIP Hs = a * WW3 + b

    reg = LinearRegression()
    reg.fit(combo[["Hs_lagged_m"]].values, combo["Hs_029_m"].values)

    a = float(reg.coef_[0])
    b = float(reg.intercept_)

    # 3. Apply calibration to original WW3 timeseries

    df_cal = df_ww3.copy()
    df_cal["Hs_029_pred_m"] = a * df_ww3["Hs_m"] + b

    return df_cal, a, b, lag_steps, combo


def bolinas_wave_quality(direction_deg, period_sec, height_ft):
    """
    Calculate wave quality factor for Bolinas based on direction, period, and size.
    This encodes Bolinas-specific knowledge about which waves actually work.

    Parameters:
    -----------
    direction_deg : float
        Wave direction in degrees
    period_sec : float
        Wave period in seconds
    height_ft : float
        Wave height in feet (offshore)

    Returns:
    --------
    'quality_factor': float (0.0-1.0) - How well this wave will work
    """

    # SSW (150-240°) - Ideal, unblocked direction towards beach
    if 150 <= direction_deg < 240:
        quality_factor = 1.0

    # W (240-280°) - Good with right conditions
    elif 240 <= direction_deg < 280:
        # Needs long period and size to wrap around reef
        if period_sec > 13 and height_ft > 3.5:
            quality_factor = 0.7
        elif period_sec >= 11 and height_ft > 3:
            quality_factor = 0.5
        else:
            quality_factor = 0.2

    # NW (280-315°) - Marginal, needs big size and period
    elif 280 <= direction_deg < 315:
        if period_sec > 13 and height_ft >= 6:
            quality_factor = 0.6
        else:
            quality_factor = 0.1

    # Blocked directions (>315° or <150°)
    else:
        quality_factor = 0.05

    return quality_factor

def wind_direction_to_degrees(direction):
    """Convert compass direction to degrees"""
    directions = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    return directions.get(direction, 0)

def calculate_surf_score(height_m, period_s, direction_deg,
                         wind_speed, wind_direction,
                         tide_height,
                         optimal_tide_range=(-1.0, 3.0),
                         coast_orientation=165):
    """
    Score surf quality at Bolinas using *single-swell* WW3 forecast output.

    Parameters:
    ----------
    height_m        : significant swell height (meters)
    period_s        : swell period (seconds)
    direction_deg   : swell direction (degrees)
    wind_speed      : wind speed in mph
    wind_direction  : string ("NW", "S", etc.) or degrees
    tide_height     : tide height in ft MLLW
    """

    # 1. SWELL SCORE (40% weight)

    if pd.isna(height_m) or pd.isna(period_s) or pd.isna(direction_deg):
        swell_score = 0
    else:
        # Convert height to feet
        height_ft = height_m * 3.28084

        # Smooth logistic period-quality multiplier
        L = 0.3   # minimum multiplier (short-period wind swell)
        H = 1.2   # maximum multiplier (long-period groundswell)
        T0 = 13   # midpoint where curve crosses ~0.75
        k = 0.8   # steepness
        period_mult = L + (H - L) / (1 + np.exp(-k * (period_s - T0)))

        # Directional multiplier
        dir_mult = bolinas_wave_quality(direction_deg, period_s, height_ft)

        # Component score
        swell_score = height_ft * period_mult * dir_mult * 1.5
        swell_score = min(10, swell_score)

	# Directional multiplier using custom coastal wrap logic
        dir_mult = bolinas_wave_quality(direction_deg, period_s, height_ft)

        # omponent score
        swell_score = height_ft * period_mult * dir_mult * 1.5
        swell_score = min(10, swell_score)


    # 2. WIND SCORE (30% weight)
    if pd.isna(wind_speed):
        wind_score = 5
    else:
        # Convert wind to degrees
        if isinstance(wind_direction, str):
            wind_deg = wind_direction_to_degrees(wind_direction)
        else:
            wind_deg = wind_direction

        # Relative angle between wind and coastline orientation
        wind_angle = abs(wind_deg - coast_orientation)
        if wind_angle > 180:
            wind_angle = 360 - wind_angle

        is_offshore = 90 < wind_angle < 270

        if is_offshore:
            if wind_speed < 5:
                wind_score = 10
            elif wind_speed < 10:
                wind_score = 8
            elif wind_speed < 15:
                wind_score = 6
            else:
                wind_score = 4
        else:
            if wind_speed < 5:
                wind_score = 7
            elif wind_speed < 10:
                wind_score = 4
            elif wind_speed < 20:
                wind_score = 2
            else:
                wind_score = 0


    # 3. TIDE SCORE (20%)
    if pd.isna(tide_height):
        tide_score = 5
    else:
        min_opt, max_opt = optimal_tide_range

        if min_opt <= tide_height <= max_opt:
            tide_score = 10
        else:
            distance = min(abs(tide_height - min_opt),
                           abs(tide_height - max_opt))
            tide_score = max(0, 10 - distance * 2)


    # 4. FINAL SCORE (0–10)
    final_score = (
        swell_score * 0.4 +
        wind_score * 0.3 +
        tide_score * 0.2 +
        5 * 0.1   # baseline consistency
    )

    return round(final_score, 1)

def predict_bolinas_surf_height(height_m, period_s, direction_deg,
                                nearshore_factor=0.85):
    """
    Predict Bolinas surf height from a *single* offshore WW3 swell.

    Inputs:
        height_m      -> Hs (meters) from WW3 calibrated to CDIP
        period_s      -> Tp_s (seconds)
        direction_deg -> Dir_deg (degrees)

    Returns:
        Clean dictionary with surf height range + swell characteristics
    """

    # If no swell data, return zeros
    if pd.isna(height_m) or pd.isna(period_s) or pd.isna(direction_deg):
        return {
            'bolinas_surf_min_ft': 0.0,
            'bolinas_surf_max_ft': 0.0,
            'swell_quality': None,
        }

    # Convert units
    height_ft = float(height_m) * 3.28084

    # Bolinas-specific wrap/transformation
    quality_factor = bolinas_wave_quality(direction_deg, period_s, height_ft)

    # Apply nearshore shoaling factor
    bolinas_height = height_ft * quality_factor * nearshore_factor

    # Variability range for single swell (tight range)
    range_factor = 0.15
    low = bolinas_height * (1 - range_factor)
    high = bolinas_height * (1 + range_factor)

    # Round to nearest 0.5 ft
    round_half = lambda x: round(x * 2) / 2

    return {
        'bolinas_surf_min_ft': round_half(max(0.5, low)),
        'bolinas_surf_max_ft': round_half(high),
        'swell_quality': round(quality_factor, 3),
    }

# PROCESS DATA WRAPPER
def process_data_wrapper(fetch_data_output, config):
    """
    Process and merge all forecast inputs (swell, tide, wind, daylight)
    into a unified surf-forecast dataframe and compute surf scores +
    predicted surf heights.

    Parameters (all dfs included in fetch_data_output)
    ----------
    ww3_df : pd.DataFrame
        Raw WW3 forecast dataset (hourly), before calibration.

    cdip_df : pd.DataFrame
        3-hourly CDIP buoy observations (for calibrating WW3 to CDIP reality).

    tide_df : pd.DataFrame
        Tide predictions indexed by datetime with column "tide_height".

    wind_df : pd.DataFrame
        Wind predictions indexed by datetime with columns
        ["wind_speed", "wind_direction"].

    sun_df : pd.DataFrame
        Sunrise/sunset table (indexed by date) with columns:
        ["first_light", "last_light"].
        Must be timezone-naive for direct comparison.

    config : dict
        Loaded YAML config containing forecast_hours and other parameters.

    Returns
    -------
    pd.DataFrame
        Final cleaned forecast dataframe containing:
        - Surf Score (1–10)
        - Swell Quality (0–1)
        - Surf Height Min (ft)
        - Surf Height Max (ft)
        - Wind Speed (MPH)
        - Wind Direction (Degrees)
        - Tide Height (ft)
        - Swell Size (m)
        - Swell Period (Seconds)
        - Swell Direction (Degrees)
        - is_daylight flag
    """
    # 1. Calibrate WW3 Forecasts with CDIP 029 Actuals
    print('calibrating swell')
    calibrated_df, a, b, lag_steps, combo = calibrate_ww3_to_cdip029(fetch_data_output['ww3'], fetch_data_output['cdip'])

    # 2. Create forecast timeline
    print('creating forecast dataframe')
    start_time = datetime.now(pacific).replace(minute=0, second=0, microsecond=0)
    forecast_index = pd.date_range(start=start_time, periods=config['forecast_hours'], freq="h", tz=pacific).tz_localize(None)
    forecast_df = pd.DataFrame(index=forecast_index)

    # 3. Join swell, tide, and wind 
    forecast_df = forecast_df.join(calibrated_df, how="left")
    forecast_df = forecast_df.join(fetch_data_output['tide'][["tide_height"]], how="left")
    forecast_df = forecast_df.join(fetch_data_output['wind'][["wind_speed", "wind_direction"]], how="left")

    # 4. Join sunrise/sunset 
    forecast_df["date"] = forecast_df.index.date
    forecast_df = forecast_df.join(
        fetch_data_output['sun'][["first_light", "last_light"]],
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
    return forecast_df