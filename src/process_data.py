# IMPORTS
import pandas as pd
import numpy as np
from reference_functions import status
from zoneinfo import ZoneInfo

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


def expand_timeseries_data(df: pd.DataFrame, freq="h", limit=6):
    """
    Expand and interpolate wave or wind timeseries data to a higher frequency
    using circular math for directions and linear math for magnitudes.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe indexed by datetime. Supports both 'flat' data
        (like wind/MOP) and 'partitioned' data containing a 'swell_idx' column.
    freq : str, optional
        The pandas frequency string for the new index (e.g., 'h' for hourly,
        '15min' for 15-minute intervals). Default is 'h'.
    limit : int, optional
        Maximum number of consecutive NaNs to fill. Gaps larger than this
        threshold will remain as NaNs to avoid over-interpolation. Default is 6.

    Returns
    -------
    pandas.DataFrame
        The expanded dataframe with a uniform time index. Directional columns
        are interpolated via vector components (sin/cos) to handle 0/360 crossovers,
        while numeric magnitudes are interpolated linearly.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Setup Index
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    full_index.name = "time"

    # Handle Partitioning (Swell 1, 2, 3) vs Flat Data
    if "swell_idx" in df.columns:
        expanded = df.groupby("swell_idx", group_keys=False).apply(
            lambda g: g.reindex(full_index), include_groups=False
        )
    else:
        expanded = df.reindex(full_index)

    # Categorize Columns (Added 'wind' keywords)
    dir_keywords = ["dir", "dp", "deg", "direction"]

    dir_cols = [
        c
        for c in expanded.columns
        if any(key in c.lower() for key in dir_keywords) and not c.startswith("_")
    ]

    linear_cols = [
        c
        for c in expanded.select_dtypes(include=[np.number]).columns
        if c not in dir_cols and c != "swell_idx"
    ]

    # Apply Linear Interpolation (Speeds, Heights, Periods)
    if linear_cols:
        expanded[linear_cols] = expanded[linear_cols].interpolate(
            method="linear", limit=limit, limit_direction="both"
        )

    # Apply Circular Interpolation (Wave & Wind Directions)
    for col in dir_cols:
        if expanded[col].isna().all():
            continue

        rad = np.deg2rad(expanded[col])
        # Interpolate the vector components separately
        s_sin = np.sin(rad).interpolate(
            method="linear", limit=limit, limit_direction="both"
        )
        s_cos = np.cos(rad).interpolate(
            method="linear", limit=limit, limit_direction="both"
        )

        # Reconstruct the angle
        expanded[col] = (np.rad2deg(np.arctan2(s_sin, s_cos)) + 360) % 360

    return expanded


def compute_swell_score(ds_swell, surf_model):
    """
    Compute a 1-10 swell quality score for each timestep in a swell-filtered
    MOP dataset based on wave energy density and spectral parameters.

    The score evaluates how clean and powerful the swell will by combining
    significant wave height (Hs), peak period (Tp), and directional spread.
    It applies non-linear square-root ramps to prioritize long-period
    groundswells and penalize disorganized spectral spread.

    Parameters
    ----------
    ds_swell : xarray.Dataset
        CDIP MOP dataset pre-filtered to the swell frequency band.
        Must contain variables:
        - waveEnergyDensity
        - waveA1Value / waveB1Value (for directional spread)
        - waveBandwidth
        - waveDp
        - waveTime
    surf_model: config with spectral parameters

    Returns
    -------
    pandas.DataFrame
        Time-indexed DataFrame containing:
        - hs_swell : Significant swell height (meters)
        - tp : Peak wave period (seconds)
        - dp : Peak wave direction (degrees)
        - spread : Directional spread (degrees)
        - r1 : First directional moment (unitless)
        - hs_score : Normalized height component [0, 1]
        - tp_score : Normalized period component [0, 1]
        - spread_score : Normalized spread component [0, 1]
        - swell_score : Final composite score [1, 10]
    """
    # Spectral peak info per timestep
    peak_idx = ds_swell.waveEnergyDensity.argmax(dim="waveFrequency")
    a1_peak = ds_swell.waveA1Value.isel(waveFrequency=peak_idx)
    b1_peak = ds_swell.waveB1Value.isel(waveFrequency=peak_idx)
    freq_peak = ds_swell.waveFrequency.isel(waveFrequency=peak_idx)
    r1 = np.sqrt(a1_peak**2 + b1_peak**2)
    spread_deg = np.degrees(np.sqrt(2 * (1 - r1)))

    # Swell Hs
    bw = ds_swell.waveBandwidth
    m0 = (ds_swell.waveEnergyDensity * bw).sum(dim="waveFrequency")
    hs_swell = 4 * np.sqrt(m0)

    # config
    spectral_cfg = surf_model["spectral_scoring"]

    # sqrt curve: ramp up
    hs_range = spectral_cfg["hs_full_credit_m"] - spectral_cfg["hs_min_m"]
    hs_score = np.sqrt(np.clip((hs_swell - spectral_cfg["hs_min_m"]) / hs_range, 0, 1))

    # Peak period
    tp = 1.0 / freq_peak
    tp_range = spectral_cfg["tp_full_credit_s"] - spectral_cfg["tp_min_s"]
    tp_score = np.sqrt(np.clip((tp - spectral_cfg["tp_min_s"]) / tp_range, 0, 1))

    # Spread at peak frequency
    spread_range = spectral_cfg["spread_max_deg"] - spectral_cfg["spread_min_deg"]
    spread_score = np.clip(
        1 - (spread_deg - spectral_cfg["spread_min_deg"]) / spread_range, 0, 1
    )

    # Composite: scale 0-1 to 1-10
    raw = (
        (hs_score * spectral_cfg["w_hs"])
        + (tp_score * spectral_cfg["w_tp"])
        + (spread_score * spectral_cfg["w_sp"])
    )
    score = 1 + raw * 9

    return pd.DataFrame(
        {
            "hs_swell": hs_swell.values,
            "tp": tp.values,
            "dp": ds_swell.waveDp.values,
            "spread": spread_deg.values,
            "r1": r1.values,
            "hs_score": hs_score.values,
            "tp_score": tp_score.values,
            "spread_score": spread_score.values,
            "swell_score": score.values,
        },
        index=pd.DatetimeIndex(ds_swell.waveTime.values, name="time"),
    )


def predict_bolinas_surf_height(
    height_ft: float,
    period_s: float,
    nearshore_cfg: dict,
):
    """
    Estimate the nearshore surf height range at Bolinas using MOP-derived
    nearshore swell height. Applies a dynamic variability range based
    on wave period to account for set consistency and shoaling.

    Parameters
    ----------
    height_ft : float
        Nearshore significant wave height (ft) from CDIP MOP.
    period_s : float
        Peak swell period (s).
    nearshore_cfg : dict
        Configuration block: surf_model.nearshore
        Keys:
            range_factor : float (base % variability)
            range_period_min : float (period threshold for increasing range)
            range_step : float (% increase in range per second of period)

    Returns
    -------
    dict
        {
            "bolinas_surf_min_ft": float,
            "bolinas_surf_max_ft": float
        }
    """
    # Missing inputs or flat ocean → zero
    if pd.isna(height_ft) or pd.isna(period_s) or height_ft <= 0:
        return {
            "bolinas_surf_min_ft": 0.0,
            "bolinas_surf_max_ft": 0.0,
        }

    # Increase the range by X% for every second of period over Ys (e.g., 12s)
    dynamic_rf = nearshore_cfg["range_factor"] + (
        max(0, period_s - nearshore_cfg["range_period_min"])
        * nearshore_cfg["range_step"]
    )

    # Apply to height-MOP height as the median expectation
    h_min = height_ft * (1 - dynamic_rf)
    h_max = height_ft * (1 + dynamic_rf)

    # Clean up for readability
    to_half = lambda x: round(x * 2) / 2

    # Enforce a minimum floor of 0.5ft for any active swell
    h_min_final = to_half(max(0.5, h_min))
    h_max_final = to_half(max(h_min_final, h_max))

    return {
        "bolinas_surf_min_ft": h_min_final,
        "bolinas_surf_max_ft": h_max_final,
    }


def calculate_surf_score(
    swell_score: float,
    wind_speed: float,
    wind_gust: float,
    wind_category: str,
    tide_height: float,
    surf_model: dict,
):
    """
    Compute a 1-10 final surf quality score using parameterized wind and tide
    penalties. Logic is multiplicative: Final = Swell * Wind_P * Tide_P.

    Parameters
    ----------
    swell_score : float
        The 1-10 quality score from compute_swell_score().
    wind_speed : float
        Wind speed in mph.
    wind_gust : float
        Wind gust in mph.
    wind_category : str
        'offshore', 'crosshore', or 'onshore'.
    tide_height : float
        Tide height in feet (MLLW).
    surf_model : dict
        The 'surf_model' block from your config.

    Returns
    -------
    tuple
        (final_score, swell_score, wind_score, tide_score)
    """
    # WIND COMPONENT
    w_cfg = surf_model["wind"]

    # Map the category to the penalty weights in the config
    w_dir_map = {
        "offshore": w_cfg["offshore_penalty_weight"],
        "crosshore": w_cfg["crosshore_penalty_weight"],
        "onshore": w_cfg["onshore_penalty_weight"],
    }
    # don't penalize scores if wind is null
    if pd.isna(wind_speed) or wind_category is np.nan:
        wind_score = np.nan
        w_penalty = 1.0
    else:
        w_dir = w_dir_map.get(wind_category, 1.0)  # Default to onshore if error

        # Calculate gust factor (weighted average)
        gust_weight = w_cfg["gust_weight"]
        adjusted_speed = (wind_speed * (1 - gust_weight)) + (wind_gust * gust_weight)

        # Calculate penalty: (Effective Speed - Floor) / Range
        effective_speed = adjusted_speed * w_dir
        w_penalty = np.clip(
            1 - (effective_speed - w_cfg["speed_floor"]) / w_cfg["speed_range"],
            w_cfg["penalty_min"],
            1.0,
        )
        wind_score = w_penalty * 10.0

    # TIDE COMPONENT
    t_cfg = surf_model["tide"]

    # don't penalize scores if tide  is null
    if pd.isna(tide_height):
        tide_score = np.nan
        t_penalty = 1.0

    else:
        # Gaussian Falloff: exp(-0.5 * ((current - optimal) / sigma)^2)
        t_penalty = np.clip(
            np.exp(
                -0.5 * ((tide_height - t_cfg["optimal_height"]) / t_cfg["sigma"]) ** 2
            ),
            t_cfg["penalty_min"],
            1.0,
        )
        tide_score = t_penalty * 10.0

    # FINAL SCORE
    f_cfg = surf_model["final_scoring"]

    # Calculate individual potential reductions
    w_red = (
        (1.0 - w_penalty) * f_cfg["wind_impact_weight"]
        if wind_score < swell_score
        else 0
    )
    t_red = (
        (1.0 - t_penalty) * f_cfg["tide_impact_weight"]
        if tide_score < swell_score
        else 0
    )

    # Use 100% of the worse penalty + x% of the lesser penalty
    primary_red = max(w_red, t_red)
    secondary_red = min(w_red, t_red)

    combined_reduction = primary_red + (secondary_red * f_cfg["secondary_penalty"])

    final_multiplier = max(f_cfg["min_multiplier"], 1.0 - combined_reduction)

    raw_final = swell_score * final_multiplier

    # But final_score cannot be less than the lowest of any individual score
    lowest_individual_score = min(swell_score, wind_score, tide_score)
    final_score = max(raw_final, lowest_individual_score)

    return (
        round(float(final_score), 1),
        round(float(swell_score), 1),
        round(float(wind_score), 1),
        round(float(tide_score), 1),
    )


def aggregate_hourly_partitions(group: pd.DataFrame):
    """
    Identify and rank the dominant and secondary raw ocean swells from
    WW3 partitions for UI visualization.

    This function processes a group of wave partitions (typically for a single
    hour) and ranks them by Wave Power to ensure that
    energetic long-period groundswells are prioritized over short-period
    wind chop, even if the latter has a slightly higher significant height.

    Parameters
    ----------
    group : pandas.DataFrame
        A dataframe slice containing partitioned swell data for a specific
        timestamp. Expected columns include 'Hs_ft', 'Tp_s', 'Dir_deg',
        and environmental context like 'wind_speed' and 'tide_height'.

    Returns
    -------
    pandas.Series
        A flattened record containing environmental metadata and the
        diagnostics for both the 'dominant' and 'secondary' swells.
        Includes calculated 'power_idx' for both partitions to assist
        in UI ranking and visualization.
    """

    def _scalar(x):
        return x.iloc[0] if isinstance(x, (pd.Series, np.ndarray)) else x

    # Handle Empty Data
    if len(group) == 0:
        return pd.Series({"dominant_Hs_ft": np.nan, "secondary_Hs_ft": np.nan})

    # Rank by Power (Hs^2 * Tp)
    group["power"] = (group["Hs_ft"] ** 2) * group["Tp_s"]
    ranked = group.sort_values(by="power", ascending=False)

    dom = ranked.iloc[0]
    sec = ranked.iloc[1] if len(ranked) > 1 else None

    # Flatten into a single row for the UI
    return pd.Series(
        {
            # Environmental Context
            "wind_speed": float(_scalar(group["wind_speed"])),
            "wind_direction": float(_scalar(group["wind_direction"])),
            "wind_category": _scalar(group["wind_category"]),
            "tide_height": float(_scalar(group["tide_height"])),
            # Dominant Swell Diagnostics (Raw WW3)
            "dominant_Hs_ft": float(dom["Hs_ft"]),
            "dominant_Tp_s": float(dom["Tp_s"]),
            "dominant_Dir_deg": float(dom["Dir_deg"]),
            "dominant_power_idx": round(float(dom["power"]), 1),
            # Secondary Swell Diagnostics (Raw WW3)
            "secondary_Hs_ft": float(sec["Hs_ft"]) if sec is not None else np.nan,
            "secondary_Tp_s": float(sec["Tp_s"]) if sec is not None else np.nan,
            "secondary_Dir_deg": float(sec["Dir_deg"]) if sec is not None else np.nan,
            "secondary_power_idx": (
                round(float(sec["power"]), 1) if sec is not None else 0.0
            ),
        }
    )


def process_data_wrapper(fetch_data_output: dict, config: dict):
    """
    Synchronize and process multiple wave and atmospheric data sources into
    a unified forecast for the Bolinas Streamlit UI.

    This wrapper executes the high-level logic of the surf model:
    1. Transforms CDIP MOP spectral data into nearshore 'Swell Quality' scores.
    2. Translates nearshore heights and periods into surfable wave heights (ft).
    3. Aligns and classifies wind data relative to the coastline (Onshore/Offshore).
    4. Ranks global WW3 partitions by wave power to provide offshore context.
    5. Integrates tidal and solar data to determine sessionability and daylight.
    6. Renames all backend diagnostics to user-friendly UI labels.

    Parameters
    ----------
    fetch_data_output : dict
        A dictionary containing raw DataFrames and Datasets for 'cdip_mop',
        'ww3', 'wind', 'tide', and 'sun'.
    config : dict
        A nested configuration dictionary containing model parameters,
        coastline orientation, and wind/tide scoring thresholds.

    Returns
    -------
    pandas.DataFrame
        A fully processed, hourly-indexed forecast table where each row
        contains all necessary metrics for a single UI dashboard entry,
        including final 'Surf Scores', wave heights, and swell diagnostics.
    """
    status("Starting processing pipeline...")

    surf_cfg = config["surf_model"]
    data_src = config["data_sources"]
    coast_orientation = data_src["coast_orientation"]

    # CDIP MOP PROCESSING
    status("Computing MOP spectral diagnostics and swell quality...")
    mop_swell_score = compute_swell_score(fetch_data_output["cdip_mop"], surf_cfg)
    mop_swell_df = expand_timeseries_data(mop_swell_score)

    # Predict heights (ft)
    mop_heights = mop_swell_df.apply(
        lambda row: predict_bolinas_surf_height(
            row["hs_swell"] * 3.28084, row["tp"], surf_cfg["nearshore"]
        ),
        axis=1,
    )
    mop_heights_df = pd.DataFrame(mop_heights.tolist(), index=mop_swell_df.index)
    mop_combined = pd.concat([mop_swell_df, mop_heights_df], axis=1)

    # ENVIRONMENTAL DATA
    tide_df = fetch_data_output["tide"][["tide_height"]]
    wind_df_hourly = expand_timeseries_data(
        fetch_data_output["wind"][["wind_speed", "wind_gust", "wind_direction"]]
    )

    wind_df_hourly["wind_category"] = wind_df_hourly["wind_direction"].apply(
        lambda wd: classify_wind_relative_to_coast(
            wind_direction=wd,
            coast_orientation_deg=coast_orientation,
            offshore_threshold=surf_cfg["wind"]["offshore_threshold_deg"],
            onshore_threshold=surf_cfg["wind"]["onshore_threshold_deg"],
        )
    )

    # Merge
    forecast_df = mop_combined.join(tide_df, how="left")
    forecast_df = forecast_df.join(wind_df_hourly, how="left")

    # limit df to foreward looking
    now_pacific = pd.Timestamp.now(pacific).floor("h").tz_localize(None)
    forecast_df = forecast_df[forecast_df.index >= now_pacific]

    # FINAL QUALITY SCORING
    status("Applying penalties and generating scores...")
    scoring_results = forecast_df.apply(
        lambda row: calculate_surf_score(
            row["swell_score"],
            row["wind_speed"],
            row["wind_gust"],
            row["wind_category"],
            row["tide_height"],
            surf_cfg,
        ),
        axis=1,
    )

    forecast_df[
        ["final_surf_score", "primary_swell_score", "wind_score", "tide_score"]
    ] = pd.DataFrame(scoring_results.tolist(), index=forecast_df.index)

    # WW3 PARTITION CONTEXT (Diagnostic Only)
    # We use power ranking to pick the Dominant/Secondary for the UI strings
    df_ww3 = expand_timeseries_data(fetch_data_output["ww3"])
    if isinstance(df_ww3.index, pd.MultiIndex):
        df_ww3 = df_ww3.reset_index(level=0, drop=True)
    df_ww3["Hs_ft"] = df_ww3["Hs_m"] * 3.28084

    df_ww3 = df_ww3.join(tide_df, how="left")
    df_ww3 = df_ww3.join(wind_df_hourly, how="left")

    ww3_context = (
        df_ww3.reset_index().groupby("time").apply(aggregate_hourly_partitions)
    )
    forecast_df = forecast_df.join(
        ww3_context[
            [
                "dominant_Hs_ft",
                "dominant_Tp_s",
                "dominant_Dir_deg",
                "dominant_power_idx",
                "secondary_Hs_ft",
                "secondary_Tp_s",
                "secondary_Dir_deg",
                "secondary_power_idx",
            ]
        ],
        how="left",
    )

    # DAYLIGHT LOGIC
    forecast_df["date"] = forecast_df.index.date
    forecast_df = forecast_df.join(
        fetch_data_output["sun"][["first_light", "last_light"]], on="date", how="left"
    )
    forecast_df["is_daylight"] = (forecast_df.index >= forecast_df["first_light"]) & (
        forecast_df.index <= forecast_df["last_light"]
    )

    # UI RENAMING (Mapping Backend to Streamlit)
    rename_map = {
        "final_surf_score": "Surf Score (1-10)",
        "bolinas_surf_min_ft": "Surf Height Min (ft)",
        "bolinas_surf_max_ft": "Surf Height Max (ft)",
        "primary_swell_score": "Dominant Swell Score (1-10)",  # Matches UI Breakdown
        "wind_score": "Wind Score (1-10)",
        "tide_score": "Tide Score (1-10)",
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
        "hs_swell": "MOP Hs (m)",
        "tp": "Peak Period (s)",
        "dp": "Peak Direction (deg)",
        "spread": "Directional Spread (deg)",
    }

    forecast_df = forecast_df.rename(columns=rename_map)
    status(f"Processing complete. Final forecast rows: {len(forecast_df)}")

    return forecast_df
