# IMPORTS
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import xarray as xr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import yaml
from reference_functions import status

pacific = ZoneInfo("America/Los_Angeles")


def classify_wind_relative_to_coast(
    wind_direction: int,
    coast_orientation_deg: float,
    offshore_threshold: float,
    onshore_threshold: float,
):
    """
    Classify wind as offshore / crosshore / onshore relative to coastline.

    wind_direction is meteorological (direction FROM which wind blows).
    coast_orientation_deg is the direction the coast faces (toward ocean).

    Parameters
    ----------
    wind_direction: int
    coast_orientation_deg: float
    offshore_threshold: float
    onshore_threshold: float

    Returns
    -------
    string value of 'offshore', 'onshore', 'crosshore'
    """
    if pd.isna(wind_direction):
        return np.nan

    # Convert FROM → TOWARD
    wind_toward_deg = (wind_direction + 180) % 360

    # Angular difference (wrapped to [0, 180])
    angle = abs(wind_toward_deg - coast_orientation_deg)
    if angle > 180:
        angle = 360 - angle

    if angle <= offshore_threshold:
        return "offshore"
    elif angle >= onshore_threshold:
        return "onshore"
    else:
        return "crosshore"


def expand_ww3_for_surf_model(df_ww3):
    """
    Process WW3 for surf modeling in the following ways:
    - Expand 3-hourly WW3 partitioned swell data to hourly resolution
    using physical interpolation
    - Add a height in ft column

    Parameters
    ----------
    df_ww3 : pandas.DataFrame
        Partitioned WW3 output indexed by time (3-hourly),
        with columns including:
        - swell_idx
        - Hs_m
        - Tp_s
        - Dir_deg

    Returns
    -------
    pandas.DataFrame
        Hourly WW3 dataframe with the same columns and partition structure.
    """
    # return hourly
    if not isinstance(df_ww3.index, pd.DatetimeIndex):
        raise ValueError("df_ww3 must be indexed by time")

    full_index = pd.date_range(
        start=df_ww3.index.min(), end=df_ww3.index.max(), freq="h"
    )

    def _interp_partition(g):
        g = g.reindex(full_index)

        # Linear interpolation for scalar quantities
        for col in ["Hs_m", "Tp_s"]:
            if col in g:
                g[col] = g[col].interpolate(limit_direction="both")

        # Circular interpolation for direction
        if "Dir_deg" in g:
            rad = np.deg2rad(g["Dir_deg"])
            g["_sin"] = np.sin(rad)
            g["_cos"] = np.cos(rad)

            g["_sin"] = g["_sin"].interpolate(limit_direction="both")
            g["_cos"] = g["_cos"].interpolate(limit_direction="both")

            g["Dir_deg"] = (np.rad2deg(np.arctan2(g["_sin"], g["_cos"])) + 360) % 360
            g = g.drop(columns=["_sin", "_cos"])

        return g

    hourly = df_ww3.groupby("swell_idx", group_keys=False).apply(_interp_partition)

    hourly.index.name = "time"
    hourly["Hs_ft"] = hourly["Hs_m"] * 3.28084

    return hourly


def expand_wind_to_hourly(df_wind: pd.DataFrame):
    """
    Expand NWS gridpoint wind forecast to hourly resolution
    using stepwise hold (forward fill).

    Parameters
    ----------
    df_wind : pandas.DataFrame
        Wind forecast indexed by datetime, containing:
        - wind_speed
        - wind_gust
        - wind_direction

    Returns
    -------
    pandas.DataFrame
        Hourly wind forecast.
    """
    if not isinstance(df_wind.index, pd.DatetimeIndex):
        raise ValueError("df_wind must be indexed by datetime")

    full_index = pd.date_range(
        start=df_wind.index.min(), end=df_wind.index.max(), freq="h"
    )

    # 1. Reindex
    hourly = df_wind.reindex(full_index)

    # 2. Interpolate Speeds (Linear)
    hourly[["wind_speed", "wind_gust"]] = hourly[
        ["wind_speed", "wind_gust"]
    ].interpolate(method="linear")

    # 3. Forward Fill Direction (Avoids the 350 -> 10 deg South flip)
    hourly["wind_direction"] = hourly["wind_direction"].ffill()

    hourly.index.name = "time"
    return hourly


def bolinas_wave_propagation(
    direction_deg: float, period_sec: float, height_ft: float, propagation_cfg: dict
):
    """
    Purpose
    -------
    Compute the Bolinas blended directional propagation factor for a single offshore
    swell, based on coastline geometry weighting three directional windows
    (South, West, NW) using logistic smoothing.

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
    def calculate_window_weight(angle, center, width, steepness):
        # The 'Weight' of this window based on swell direction
        # Result is 1.0 inside the core and tapers to 0.0 outside
        return 1.0 / (1.0 + np.exp(steepness * (abs(angle - center) - width)))

    def calculate_performance_factor(mid_val, k_val, current_val):
        # Reusable logistic curve 
        return 1.0 / (1.0 + np.exp(-(current_val - mid_val) * k_val))

    # 1. CALCULATE WEIGHTS 
    weights = {}
    for window in ["south_sweet_spot", "west_wrap", "nw_reef_gate"]:
        cfg = propagation_cfg[window]
        weights[window] = calculate_window_weight(
            direction_deg, cfg["center"], cfg["width"], cfg["steepness"]
        )

    # 2. CALCULATE WINDOW-SPECIFIC SCORES
    # South Logic: full swell
    score_south = 1.0 

    # West Logic: Highly dependent on period/height to wrap onto the reef
    west_wrap_factor = calculate_performance_factor(
        propagation_cfg["west_period_mid"], propagation_cfg["west_period_k"], period_sec
    )
    west_size_factor = calculate_performance_factor(
        propagation_cfg["west_height_mid"], propagation_cfg["west_height_k"], height_ft
    )
    score_west = propagation_cfg["west_base"] + (
        propagation_cfg["west_bonus"] * west_wrap_factor * west_size_factor
    )

    # NW Logic: The Gate requires even more energy to wrap around Duxbury
    nw_wrap_factor = calculate_performance_factor(
        propagation_cfg["nw_period_mid"], propagation_cfg["nw_period_k"], period_sec
    )
    nw_size_factor = calculate_performance_factor(
        propagation_cfg["nw_height_mid"], propagation_cfg["nw_height_k"], height_ft
    )
    score_nw = propagation_cfg["nw_base"] + (
        propagation_cfg["nw_bonus"] * nw_wrap_factor * nw_size_factor
    )

    # 3. COMPUTE WEIGHTED AVERAGE
    # This prevents the score jump by blending scores based on directional overlap
    total_weight = sum(weights.values())
    
    if total_weight < 0.01:
        return propagation_cfg["blocked_value"]

    final_score = (
        (score_south * weights["south_sweet_spot"]) +
        (score_west * weights["west_wrap"]) +
        (score_nw * weights["nw_reef_gate"])
    ) / total_weight

    return max(propagation_cfg["blocked_value"], final_score)


def predict_bolinas_surf_height(
    height_ft: float,
    period_s: float,
    direction_deg: float,
    nearshore_cfg: dict,
    propagation_cfg: dict,
):
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
    if pd.isna(height_ft) or pd.isna(period_s) or pd.isna(direction_deg):
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
        propagation_cfg=propagation_cfg,
    )

    # 2) nearshore shoaling (dampen extreme blocked dirs)
    transfer = nearshore_cfg["base_factor"] * np.clip(
        propagation_factor, propagation_cfg["blocked_value"], 1.0
    )

    # raw nearshore height (ft)
    bolinas_height = max(0.0, height_ft * transfer)

    # 3) variability range
    rf = nearshore_cfg["range_factor"]

    # Increase the range by X% for every second of period over Ys
    dynamic_rf = rf + (
        max(0, period_s - nearshore_cfg["range_period_min"])
        * nearshore_cfg["range_step"]
    )

    # Apply to height
    h_min = bolinas_height * (1 - dynamic_rf)
    h_max = bolinas_height * (1 + dynamic_rf)

    # round to nearest 0.5'
    to_half = lambda x: round(x * 2) / 2
    h_min = to_half(max(0.5, h_min))
    h_max = to_half(max(h_min, h_max))  # enforce ordering

    return {
        "bolinas_surf_min_ft": h_min,
        "bolinas_surf_max_ft": h_max,
        "swell_propagation": round(propagation_factor, 3),
    }


def calculate_surf_score(
    period_s: float,
    wind_speed: float,
    wind_gust: float,
    wind_category: str,
    tide_height: float,
    bolinas_surf_min_ft: float,
    bolinas_surf_max_ft: float,
    swell_propagation: float,
    coast_orientation: float,
    surf_model: dict,
):
    """
    Purpose
        Compute a 0-10 surf quality score for Bolinas for a single WW3 swell
        component, incorporating swell energy, coastal exposure, wind,
        and tide alignment using model parameters defined in the config.

    Parameters
        period_s (float): Swell period (seconds)
        wind_speed (float): Wind speed in mph
        wind_gust (float): Wind gusts in mph
        wind_category: Wind direction as categorical string - offshore, onshore, crosshore
        tide_height (float): Tide height in ft MLLW
        bolinas_surf_min_ft (float): Predicted min surf height from predict_bolinas_surf_height()
        bolinas_surf_max_ft (float): Predicted max surf height from predict_bolinas_surf_height()
        swell_propagation (float): Predicted propagation fafctor from bolinas_wave_propagation()
        coast_orientation (float): Local coastline orientation (degrees)
        surf_model (dict): Full surf_model config block

    Returns
        float: Final surf score 0-10 (rounded to 0.1)
    """
    # 1. SWELL SCORE — Bolinas-first
    if pd.isna(bolinas_surf_min_ft) or pd.isna(period_s):
        swell_score = 0.0
        energy = 0.0
    elif swell_propagation <= 0.05:
        # Hard block: swell does not meaningfully reach Bolinas
        swell_score = 0.05
        energy = 0.05
    else:
        # Period multiplier (quality only)
        pm = surf_model["period_multiplier"]
        period_mult = pm["min"] + (pm["max"] - pm["min"]) / (
            1 + np.exp(-pm["steepness"] * (period_s - pm["midpoint"]))
        )

        # Nearshore height proxy
        nearshore_h = 0.5 * (bolinas_surf_min_ft + bolinas_surf_max_ft)

        # Energy = height only
        eg = surf_model["energy"]
        energy = nearshore_h ** eg["height_exp"]

        # Propagation as gate
        swell_raw = energy * period_mult

        swell_score = 10 * (
            1 - np.exp(-eg["energy_k"] * eg["swell_saturation"] * swell_raw)
        )

    # 2. WIND SCORE
    wind_cfg = surf_model["wind"]

    if pd.isna(wind_speed) or pd.isna(wind_category):
        wind_score = 5.0
    else:
        # Global speed penalty using logistic decay
        s_mid = wind_cfg["speed_midpoint"]
        s_k = wind_cfg["speed_k"]
        speed_factor = 1 / (1 + np.exp(s_k * (wind_speed - s_mid)))

        # Directional Weighting
        raw_weight = wind_cfg[f"{wind_category}_weight"]

        # Light Wind Blend
        blend = np.clip(wind_speed / wind_cfg["glassy_cutoff"], 0, 1)
        final_weight = 1.0 - (blend * (1.0 - raw_weight))

        # Final Wind Score
        wind_score = 10.0 * speed_factor * final_weight

    # 3. TIDE SCORE (shape: 10 at center, ~edge_score at bounds, fast drop outside)
    if pd.isna(tide_height):
        tide_score = 5.0
    else:
        td = surf_model["tide"]

        center = 0.5 * (td["optimal_low"] + td["optimal_high"])
        half_width = 0.5 * (td["optimal_high"] - td["optimal_low"])

        # normalized distance from center: 0 at center, 1 at edges
        x = abs(tide_height - center) / half_width

        if x <= 1.0:
            # Inside window: smooth dome from 10 (center) down to edge_score at edges
            # Using a quadratic drop: 10 - (10-edge_score)*x^2
            tide_score = 10.0 - (10.0 - td["edge_score"]) * (x**2)
        else:
            # Outside window: decay quickly from edge_score
            # (x-1) is distance beyond edge in "half-width" units
            tide_score = td["edge_score"] * np.exp(-td["outside_decay"] * (x - 1.0))

        tide_score = float(np.clip(tide_score, 0.0, 10.0))

    # 4. FINAL SCORE
    w = surf_model["weights"]

    # Dampen the tide's influence on small/poor swells while ensuring
    # it maintains a X% minimum impact for tide-sensitive breaks.
    tide_weight_factor = np.clip(
        (swell_score + w["tide_swell_offset"]) / 10.0, w["tide_min_influence"], 1.0
    )

    # Wind only contributes to the score if there is actual swell energy.
    wind_weight_factor = np.clip(swell_score / w["wind_swell_threshold"], 0.0, 1.0)

    # Put all components together
    final_score = (
        swell_score * w["swell"]
        + wind_score * w["wind"] * wind_weight_factor
        + tide_score * w["tide"] * tide_weight_factor
        + w["baseline"] * 0.1
    )

    return (
        round(final_score, 1),
        round(swell_score, 1),
        round(wind_score, 1),
        round(tide_score, 1),
    )


def compute_partition_energy(
    bolinas_surf_min_ft: float,
    bolinas_surf_max_ft: float,
    Tp_s: float,
    energy_cfg: dict,
):
    """
    Nearshore energy proxy used for partition weighting.
    """
    h = 0.5 * (bolinas_surf_min_ft + bolinas_surf_max_ft)

    if h <= 0 or Tp_s <= 0:
        return 0.0

    return (h ** energy_cfg["height_exp"]) * (Tp_s ** energy_cfg["period_exp"])


def aggregate_hourly_partitions(group: pd.DataFrame):
    """
    Aggregate partition-level surf metrics into a single hourly row,
    retaining dominant and secondary swell diagnostics.

    GUARANTEE:
        All returned values are scalars (float / int / np.nan).
    """

    def _scalar(x):
        """
        Force scalar extraction.
        Fails loudly if something unexpected slips through.
        """
        if isinstance(x, pd.Series):
            if len(x) == 0:
                return np.nan
            return x.iloc[0]
        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                return x.item()
            if x.size == 1:
                return x.flat[0]
            raise ValueError(f"Non-scalar ndarray: shape={x.shape}")
        return x

    energy = group["partition_energy"]

    if len(group) == 0 or energy.sum() <= 0:
        return pd.Series(
            {
                "surf_score": 0.0,
                "bolinas_surf_min_ft": 0.0,
                "bolinas_surf_max_ft": 0.0,
                "wind_score": 0.0,
                "tide_score": 0.0,
                "wind_speed": 0.0,
                "wind_direction": np.nan,
                "wind_category": np.nan,
                "tide_height": 0.0,
                "dominant_Hs_ft": np.nan,
                "dominant_Tp_s": np.nan,
                "dominant_Dir_deg": np.nan,
                "dominant_energy": 0.0,
                "dominant_surf_score": 0.0,
                "dominant_swell_score": 0.0,
                "dominant_swell_propagation": 0.0,
                "secondary_Hs_ft": np.nan,
                "secondary_Tp_s": np.nan,
                "secondary_Dir_deg": np.nan,
                "secondary_energy": 0.0,
                "secondary_surf_score": 0.0,
                "secondary_energy_frac": 0.0,
                "secondary_swell_score": 0.0,
                "secondary_swell_propagation": 0.0,
            }
        )

    # Energy-weighted surf score
    surf_score = float(
        np.round(np.average(group["partition_surf_score"], weights=energy), 1)
    )

    # Rank partitions by Bolinas surf relevance using height and swell score
    group["partition_swell_score"] = group["partition_swell_score"].fillna(0.0)
    ranked = group.sort_values(
        by=["bolinas_surf_max_ft", "partition_swell_score"],
        ascending=[False, False],
    )

    dom = ranked.iloc[0]

    if len(ranked) > 1:
        sec = ranked.iloc[1]
        secondary_energy_frac = _scalar(sec["partition_energy"]) / energy.sum()
    else:
        sec = None
        secondary_energy_frac = 0.0

    return pd.Series(
        {
            # Aggregated outcomes
            "surf_score": surf_score,
            "bolinas_surf_min_ft": float(_scalar(group["bolinas_surf_min_ft"].max())),
            "bolinas_surf_max_ft": float(_scalar(group["bolinas_surf_max_ft"].max())),
            "wind_score": float(_scalar(group["partition_wind_score"].max())),
            "tide_score": float(_scalar(group["partition_tide_score"].max())),
            "wind_speed": float(_scalar(group["wind_speed"].max())),
            "wind_gust": float(_scalar(group["wind_gust"].max())),
            "wind_direction": _scalar(group["wind_direction"].mean()),
            "wind_category": _scalar(group["wind_category"].iloc[0]),
            "tide_height": float(_scalar(group["tide_height"].max())),
            # Dominant swell
            "dominant_Hs_ft": float(_scalar(dom["Hs_ft"])),
            "dominant_Tp_s": float(_scalar(dom["Tp_s"])),
            "dominant_Dir_deg": float(_scalar(dom["Dir_deg"])),
            "dominant_energy": float(_scalar(dom["partition_energy"])),
            "dominant_surf_score": float(_scalar(dom["partition_surf_score"])),
            "dominant_swell_score": float(_scalar(dom["partition_swell_score"])),
            "dominant_swell_propagation": float(_scalar(dom["swell_propagation"])),
            # Secondary swell
            "secondary_Hs_ft": float(_scalar(sec["Hs_ft"]))
            if sec is not None
            else np.nan,
            "secondary_Tp_s": float(_scalar(sec["Tp_s"]))
            if sec is not None
            else np.nan,
            "secondary_Dir_deg": float(_scalar(sec["Dir_deg"]))
            if sec is not None
            else np.nan,
            "secondary_energy": float(_scalar(sec["partition_energy"]))
            if sec is not None
            else 0.0,
            "secondary_surf_score": float(_scalar(sec["partition_surf_score"]))
            if sec is not None
            else 0.0,
            "secondary_energy_frac": float(secondary_energy_frac),
            "secondary_swell_score": float(_scalar(sec["partition_swell_score"]))
            if sec is not None
            else 0.0,
            "secondary_swell_propagation": float(_scalar(sec["swell_propagation"]))
            if sec is not None
            else 0.0,
        }
    )


def process_data_wrapper(fetch_data_output: dict, config: dict):
    """
    Purpose
        Process and merge all forecast inputs, then compute an
        energy-weighted surf score per hour using partitioned WW3 swell.

    Returns
        pandas.DataFrame
            One row per hour with aggregated surf metrics and
            dominant partition diagnostics.
    """
    status("Starting Surf Model processing pipeline...")

    surf_cfg = config["surf_model"]
    data_src = config["data_sources"]

    coast_orientation = data_src["coast_orientation"]
    forecast_hours = data_src["forecast_hours"]

    # Prepare WW3: hourly + partitioned + units
    df_ww3 = expand_ww3_for_surf_model(fetch_data_output["ww3"])
    status(f"WW3 data expanded to hourly. Current shape: {df_ww3.shape}")
    # expected columns:
    # time, swell_idx, Hs_ft, Tp_s, Dir_deg

    # Join environmental inputs onto partition rows
    df_ww3 = df_ww3.join(
        fetch_data_output["tide"][["tide_height"]], on="time", how="left"
    )

    # Get wind to hourly then join
    wind_df_hourly = expand_wind_to_hourly(
        fetch_data_output["wind"][["wind_speed", "wind_gust", "wind_direction"]]
    )
    df_ww3 = df_ww3.join(wind_df_hourly, on="time", how="left")

    # Wind Direction Classification
    df_ww3["wind_category"] = df_ww3["wind_direction"].apply(
        lambda wd: classify_wind_relative_to_coast(
            wind_direction=wd,
            coast_orientation_deg=coast_orientation,
            offshore_threshold=surf_cfg["wind"]["offshore_threshold_deg"],
            onshore_threshold=surf_cfg["wind"]["onshore_threshold_deg"],
        )
    )

    # Partition-level surf physics
    status("Computing Bolinas-specific wave propagation and surf heights...")
    surf_heights = df_ww3.apply(
        lambda row: predict_bolinas_surf_height(
            row["Hs_ft"],
            row["Tp_s"],
            row["Dir_deg"],
            surf_cfg["nearshore"],
            surf_cfg["propagation"],
        ),
        axis=1,
    )

    surf_heights_df = pd.DataFrame(surf_heights.tolist(), index=df_ww3.index)

    df_ww3 = pd.concat([df_ww3, surf_heights_df], axis=1)

    # Partition-level surf score
    status("Calculating surf quality scores based on wind and tide alignment...")
    score_outputs = df_ww3.apply(
        lambda row: calculate_surf_score(
            row["Tp_s"],
            row["wind_speed"],
            row["wind_gust"],
            row["wind_category"],
            row["tide_height"],
            row["bolinas_surf_min_ft"],
            row["bolinas_surf_max_ft"],
            row["swell_propagation"],
            coast_orientation,
            surf_cfg,
        ),
        axis=1,
    )

    df_ww3[
        [
            "partition_surf_score",
            "partition_swell_score",
            "partition_wind_score",
            "partition_tide_score",
        ]
    ] = pd.DataFrame(score_outputs.tolist(), index=df_ww3.index)

    # Partition energy (for weighting)
    df_ww3["partition_energy"] = df_ww3.apply(
        lambda r: compute_partition_energy(
            r["bolinas_surf_min_ft"],
            r["bolinas_surf_max_ft"],
            r["Tp_s"],
            surf_cfg["energy"],
        ),
        axis=1,
    )

    # Aggregate to one row per hour
    status("Aggregating swell partitions into final hourly forecast...")
    forecast_df = df_ww3.groupby("time").apply(aggregate_hourly_partitions).sort_index()

    # Daylight logic (hourly, post-aggregation)
    forecast_df["date"] = forecast_df.index.date

    forecast_df = forecast_df.join(
        fetch_data_output["sun"][["first_light", "last_light"]], on="date", how="left"
    )

    forecast_df["is_daylight"] = (forecast_df.index >= forecast_df["first_light"]) & (
        forecast_df.index <= forecast_df["last_light"]
    )

    forecast_df = forecast_df.drop(columns=["date"])

    # Final display
    forecast_df = forecast_df[
        [
            "surf_score",
            "bolinas_surf_min_ft",
            "bolinas_surf_max_ft",
            "wind_speed",
            "wind_gust",
            "wind_direction",
            "wind_category",
            "tide_height",
            "dominant_Hs_ft",
            "dominant_Tp_s",
            "dominant_Dir_deg",
            "dominant_surf_score",
            "dominant_swell_score",
            "dominant_swell_propagation",
            "secondary_Hs_ft",
            "secondary_Tp_s",
            "secondary_Dir_deg",
            "secondary_surf_score",
            "secondary_swell_score",
            "secondary_swell_propagation",
            "wind_score",
            "tide_score",
            "is_daylight",
        ]
    ]

    # Rename for end-user display
    forecast_df = forecast_df.rename(
        columns={
            "surf_score": "Surf Score (1-10)",
            "bolinas_surf_min_ft": "Surf Height Min (ft)",
            "bolinas_surf_max_ft": "Surf Height Max (ft)",
            "wind_speed": "Wind Speed (MPH)",
            "wind_gust": "Wind Gust (MPH)",
            "wind_direction": "Wind Direction (Deg)",
            "wind_category": "Wind Direction",
            "tide_height": "Tide Height (ft)",
            "dominant_Hs_ft": "Dominant Swell Size (ft)",
            "dominant_Tp_s": "Dominant Swell Period",
            "dominant_Dir_deg": "Dominant Swell Direction",
            "secondary_Hs_ft": "Secondary Swell Size (ft)",
            "secondary_Tp_s": "Secondary Swell Period",
            "secondary_Dir_deg": "Secondary Swell Direction",
            "dominant_surf_score": "Dominant Surf Score (1-10)",
            "secondary_surf_score": "Secondary Surf Score (1-10)",
            "wind_score": "Wind Score (1-10)",
            "tide_score": "Tide Score (1-10)",
            "dominant_swell_score": "Dominant Swell Score (1-10)",
            "secondary_swell_score": "Secondary Swell Score (1-10)",
            "dominant_swell_propagation": "Dominant Propagation Score (0-1)",
            "secondary_swell_propagation": "Secondary Propagation Score (0-1)",
        }
    )
    status(f"Processing complete. Final forecast rows: {len(forecast_df)}")

    return forecast_df
