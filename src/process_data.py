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

def wind_direction_to_degrees(direction: str):
    """
    Convert a compass direction string to degrees.

    Parameters
    ----------
    direction : str
        Compass direction abbreviation (e.g., 'NW')

    Returns
    -------
    float
        Direction in degrees. Returns 0.0 if not recognized.
    """
    directions = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    return directions.get(direction, 0)


def calibrate_ww3_to_cdip029(df_ww3: pd.DataFrame,
                             cdip_3h: pd.DataFrame):
    """
    Calibrate WW3 significant wave height (Hs) to CDIP buoy 029 observations.

    This function aligns hourly WW3 forecasts with 3-hourly CDIP measurements,
    determines the best time lag between the two records, and fits a linear
    regression to convert WW3 Hs into a CDIP-consistent Hs prediction.

    Parameters
    ----------
    df_ww3 : pd.DataFrame
        WW3 forecast with DatetimeIndex.
    cdip_3h : pd.DataFrame
        CDIP 029 observations with DatetimeIndex.

    Returns
    -------
    df_cal : pd.DataFrame
        WW3 data with calibrated Hs added.
    a : float
        Regression slope.
    b : float
        Regression intercept.
    lag_steps : int
        Applied 3-hour lag.
    combo : pd.DataFrame
        Merged WW3/CDIP dataset after alignment.
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
    df_cal["Hs_lagged_m"] = df_cal["Hs_m"].shift(lag_steps)
    df_cal["Hs_029_pred_ft"] = (a * df_cal["Hs_lagged_m"] + b) * 3.28084

    return df_cal, a, b, lag_steps, combo


def bolinas_wave_propagation(
        direction_deg: float,
        period_sec: float,
        height_ft: float,
        propagation_cfg: dict):
    """
    Purpose
    -------
    Compute the Bolinas directional propagation factor for a single offshore
    swell, based strictly on coastline geometry and empirically tuned wrap
    parameters supplied in the configuration.

    Parameters
    ----------
    direction_deg : float
        Offshore swell direction (degrees).
    period_sec : float
        Peak period (seconds).
    height_ft : float
        Offshore significant wave height (feet).
    propagation_cfg : dict
        Configuration dictionary under `surf_model.propagation`
        containing all directional and wrap parameters.

    Returns
    -------
    float
        Propagation factor in the range [0, 1].
    """

    # SOUTH SWEET SPOT
    south_sweet_min, south_sweet_max = propagation_cfg["south_sweet_spot"]
    if south_sweet_min <= direction_deg <= south_sweet_max:
        return 1.0

    # SOUTH EDGES
    south_edge_min, south_edge_max = propagation_cfg["south_edges"]
    if (south_edge_min <= direction_deg < south_sweet_min) or \
       (south_sweet_max < direction_deg <= south_edge_max):
        # Simple reduced contribution for near-south angles
        return 0.7

    # WEST WRAP (logistic)
    west_min, west_max = propagation_cfg["west_range"]
    if west_min <= direction_deg <= west_max:
        wrap = 1.0 / (
            1.0 + np.exp(-(period_sec - propagation_cfg["west_period_mid"])
                         * propagation_cfg["west_period_k"])
        )
        size = 1.0 / (
            1.0 + np.exp(-(height_ft - propagation_cfg["west_height_mid"])
                         * propagation_cfg["west_height_k"])
        )
        # baseline + conditional bonus
        return (
            propagation_cfg["west_base"]
            + propagation_cfg["west_bonus"] * wrap * size
        )

    # NW RARE WRAP (tanh) 
    nw_min, nw_max = propagation_cfg["nw_range"]
    if nw_min <= direction_deg <= nw_max:
        rare = (
            propagation_cfg["nw_base"]
            + propagation_cfg["nw_bonus"]
              * np.tanh((period_sec - propagation_cfg["nw_period_mid"])
                        / propagation_cfg["nw_period_scale"])
              * np.tanh((height_ft - propagation_cfg["nw_height_mid"])
                        / propagation_cfg["nw_height_scale"])
        )
        return max(propagation_cfg["nw_floor"], rare)

    # BLOCKED
    return propagation_cfg["blocked_value"]


def predict_bolinas_surf_height(
        height_ft: float,
        period_s: float,
        direction_deg: float,
        nearshore_cfg: dict,
        propagation_cfg: dict):
    """
    Purpose
    -------
    Estimate nearshore surf height range at Bolinas for a single offshore swell,
    combining directional propagation (wrap/blocking) with nearshore shoaling
    and a configurable variability range.

    Parameters
    ----------
    height_ft : float
        Offshore significant wave height (ft).
    period_s : float
        Peak swell period (s).
    direction_deg : float
        Swell direction (degrees, oceanographic).
    nearshore_cfg : dict
        Configuration block: surf_model.nearshore
        Keys:
            base_factor
            min_direction_factor
            range_factor
    propagation_cfg : dict
        Configuration block: surf_model.propagation

    Returns
    -------
    dict
        {
            "bolinas_surf_min_ft": float,
            "bolinas_surf_max_ft": float,
            "swell_propagation": float in [0,1] or None
        }
    """

    # missing inputs → zero
    if (
        pd.isna(height_ft)
        or pd.isna(period_s)
        or pd.isna(direction_deg)
    ):
        return {
            "bolinas_surf_min_ft": 0.0,
            "bolinas_surf_max_ft": 0.0,
            "swell_propagation": None,
        }

    # 1) directional attenuation
    propagation_factor = bolinas_wave_propagation(
        direction_deg=direction_deg,
        period_sec=period_s,
        height_ft=height_ft,
        propagation_cfg=propagation_cfg
    )

    # 2) nearshore shoaling (dampen extreme blocked dirs)
    transfer = (
        nearshore_cfg["base_factor"]
        * np.clip(propagation_factor,
                  nearshore_cfg["min_direction_factor"],
                  1.0)
    )

    # raw nearshore height (ft)
    bolinas_height = max(0.0, height_ft * transfer)

    # 3) variability range
    rf = nearshore_cfg["range_factor"]
    h_min = bolinas_height * (1 - rf)
    h_max = bolinas_height * (1 + rf)

    # round to nearest 0.5'
    to_half = lambda x: round(x * 2) / 2
    h_min = to_half(max(0.5, h_min))
    h_max = to_half(max(h_min, h_max))   # enforce ordering

    return {
        "bolinas_surf_min_ft": h_min,
        "bolinas_surf_max_ft": h_max,
        "swell_propagation": round(propagation_factor, 3),
    }


def calculate_surf_score(
   # height_ft: float,
    period_s: float,
   # direction_deg: float,
    wind_speed: float,
    wind_direction,
    tide_height: float,
    bolinas_surf_min_ft: float,
    bolinas_surf_max_ft: float,
    swell_propagation: float,
    coast_orientation: float,
    surf_model: dict
):
    """
    Purpose
        Compute a 0-10 surf quality score for Bolinas for a single WW3 swell
        component, incorporating swell energy, coastal exposure, wind,
        and tide alignment using model parameters defined in the config.

    Parameters
        height_ft (float): Significant swell height (ft)
        period_s (float): Swell period (seconds)
        direction_deg (float): Swell direction (degrees)
        wind_speed (float): Wind speed in mph
        wind_direction: Wind direction as degrees or a compass string
        tide_height (float): Tide height in ft MLLW
        bolinas_surf_min_ft (float): Predicted min surf height from predict_bolinas_surf_height()
        bolinas_surf_max_ft (float): Predicted max surf height from predict_bolinas_surf_height()
        swell_propagation (float): Predicted propagation fafctor from bolinas_wave_propagation()
        coast_orientation (float): Local coastline orientation (degrees)
        surf_model (dict): Full surf_model config block

    Returns
        float: Final surf score 0-10 (rounded to 0.1)
    """

    # 1. SWELL SCORE
    if pd.isna(bolinas_surf_min_ft) or pd.isna(period_s):
        swell_score = 0
        energy = 0
    else:
        # logistic period multiplier
        pm = surf_model["period_multiplier"]
        period_mult = (
            pm["min"] +
            (pm["max"] - pm["min"]) /
            (1 + np.exp(-pm["steepness"] * (period_s - pm["midpoint"])))
        )

        nearshore_h = 0.5 * (bolinas_surf_min_ft + bolinas_surf_max_ft)

        # energy law
        eg = surf_model["energy"]
        energy = nearshore_h**eg["height_exp"] * period_s**eg["period_exp"]

        # shape multiplier from propagation
        shape_mult = 0.8 + 0.3 * swell_propagation

        swell_raw = energy * period_mult * shape_mult
        swell_score = 10 * (1 - np.exp(-eg["swell_saturation"] * swell_raw))

    # 2. WIND SCORE
    if pd.isna(wind_speed):
        wind_score = 5
    else:
        wind_deg = (
            wind_direction_to_degrees(wind_direction)
            if isinstance(wind_direction, str)
            else float(wind_direction)
        )

        wind_angle = abs(wind_deg - coast_orientation)
        if wind_angle > 180:
            wind_angle = 360 - wind_angle

        offshore_quality = np.cos(np.deg2rad(wind_angle))
        speed_quality = 1 / (1 + np.exp((wind_speed - 8) / 2))
        raw_wind = offshore_quality * speed_quality

        wind_score = (raw_wind + 1) * 5  # map [-1,1] → [0,10]

    # 3. TIDE SCORE
    if pd.isna(tide_height):
        tide_score = 5
    else:
        td = surf_model["tide"]
        min_opt = td["optimal_low"]
        max_opt = td["optimal_high"]

        tide_buffer = min(td["max_buffer"], energy / td["energy_scale"])

        min_adj = min_opt - tide_buffer
        max_adj = max_opt + tide_buffer

        if min_adj <= tide_height <= max_adj:
            tide_score = 10
        else:
            distance = min(abs(tide_height - min_adj), abs(tide_height - max_adj))
            tide_score = max(0, 10 - distance * 2)

        # damp tide importance when waves are tiny
        energy_scale = np.clip(energy / td["energy_scale"], 0.0, 1.0)
        tide_score *= (0.4 + 0.6 * energy_scale)

    # 4. FINAL SCORE
    w = surf_model["weights"]
    final_score = (
        swell_score * w["swell"] +
        wind_score * w["wind"] +
        tide_score * w["tide"] +
        w["baseline"] * 0.1
    )

    return round(final_score, 1), round(swell_score, 1), round(wind_score, 1), round(tide_score, 1)


def process_data_wrapper(
    fetch_data_output: dict,
    config: dict
):
    """
    Purpose
        Process and merge all forecast inputs, then compute surf score
        and nearshore surf heights for each hourly forecast step.

    Parameters
        fetch_data_output (dict)
            Dictionary containing raw forecast input sources with the
            following required keys:
            - 'ww3'
            - 'cdip'
            - 'tide'
            - 'wind'
            - 'sun'

        config (dict)
            Full configuration dictionary including:
            - data_sources
            - surf_model

    Returns
        pandas.DataFrame
            Forecast dataframe containing hourly surf metrics and
            final surf scores.
    """
    print('processing data')

    # Config references 
    surf_cfg = config["surf_model"]
    data_src = config["data_sources"]

    coast_orientation = data_src["coast_orientation"]
    forecast_hours = data_src["forecast_hours"]

    # 1. Calibrate swell
    calibrated_df, a, b, lag_steps, combo = calibrate_ww3_to_cdip029(
        fetch_data_output["ww3"],
        fetch_data_output["cdip"]
    )

    # 2. Build hourly timeline
    start_time = datetime.now(pacific).replace(minute=0, second=0, microsecond=0)

    forecast_index = (
        pd.date_range(start=start_time,
                      periods=forecast_hours,
                      freq="h",
                      tz=pacific)
        .tz_localize(None)
    )

    forecast_df = pd.DataFrame(index=forecast_index)

    # 3. Join inputs
    forecast_df = forecast_df.join(calibrated_df, how="left")

    forecast_df = forecast_df.join(
        fetch_data_output["tide"][["tide_height"]],
        how="left"
    )

    forecast_df = forecast_df.join(
        fetch_data_output["wind"][["wind_speed", "wind_direction"]],
        how="left"
    )

    # 4. Sunrise / sunset
    forecast_df["date"] = forecast_df.index.date

    forecast_df = forecast_df.join(
        fetch_data_output["sun"][["first_light", "last_light"]],
        on="date",
        how="left"
    )

    # 5. Daylight flag
    forecast_df["is_daylight"] = (
        (forecast_df.index >= forecast_df["first_light"]) &
        (forecast_df.index <= forecast_df["last_light"])
    )

    # 6. Clean up missing data
    forecast_df = forecast_df.ffill()

    # 7. Predict surf heights 
    surf_heights = [
        predict_bolinas_surf_height(
            row["Hs_029_pred_ft"],
            row["Tp_s"],
            row["Dir_deg"],
            surf_cfg["nearshore"],   
            surf_cfg["propagation"]   
        )
        for _, row in forecast_df.iterrows()
    ]

    surf_heights_df = pd.DataFrame(
        surf_heights,
        index=forecast_df.index
    )

    forecast_df = pd.concat([forecast_df, surf_heights_df], axis=1)

    # 7. Predict surf score
    score_outputs = forecast_df.apply(
        lambda row: calculate_surf_score(
        #    row["Hs_029_pred_ft"],
            row["Tp_s"],
        #    row["Dir_deg"],
            row["wind_speed"],
            row["wind_direction"],
            row["tide_height"],
            row["bolinas_surf_min_ft"],
            row["bolinas_surf_max_ft"],
            row["swell_propagation"],
            coast_orientation,
            surf_cfg 
        ),
        axis=1
    )

    # # 7. Predict surf scores (all components returned separately)
    # score_outputs = forecast_df.apply(
    #     lambda row: calculate_surf_score(
    #         row["Hs_029_pred_ft"],
    #         row["Tp_s"],
    #         row["Dir_deg"],
    #         row["wind_speed"],
    #         row["wind_direction"],
    #         row["tide_height"],
    #         row["bolinas_surf_min_ft"],
    #         row["bolinas_surf_max_ft"],
    #         row["swell_propagation"],
    #         coast_orientation,
    #         surf_cfg
    #     ),
    #     axis=1
    # )

    # score_outputs is a Series of tuples → expand into columns
    forecast_df[[
        "surf_score",
        "swell_score_component",
        "wind_score_component",
        "tide_score_component"
    ]] = pd.DataFrame(score_outputs.tolist(), index=forecast_df.index)

    # 8. Final fields (display order)
    forecast_df = forecast_df[[
        "surf_score",
        "bolinas_surf_min_ft",
        "bolinas_surf_max_ft",
        "wind_speed",
        "wind_direction",
        "tide_height",
        "Hs_029_pred_ft",
        "Tp_s",
        "Dir_deg",
        "swell_score_component",
        "wind_score_component",
        "tide_score_component",
        "swell_propagation",
        "is_daylight"
    ]]

    # 9. Rename for end-user display
    forecast_df = forecast_df.rename(columns={
        "surf_score": "Surf Score (1-10)",
        "bolinas_surf_min_ft": "Surf Height Min (ft)",
        "bolinas_surf_max_ft": "Surf Height Max (ft)",
        "wind_speed": "Wind Speed (MPH)",
        "wind_direction": "Wind Direction (Degrees)",
        "tide_height": "Tide Height (ft)",
        "Hs_029_pred_ft": "Swell Size (ft)",
        "Tp_s": "Swell Period (Seconds)",
        "Dir_deg": "Swell Direction (Degrees)",
        "swell_score_component": "Swell Score (1-10)",
        "wind_score_component": "Wind Score (1-10)",
        "tide_score_component": "Tide Score (1-10)",
        "swell_propagation": "Propagation Score (0-1)",
    })

    return forecast_df

