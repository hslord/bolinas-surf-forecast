# IMPORTS
import requests
from datetime import datetime, timedelta
from noaa_coops import Station
from zoneinfo import ZoneInfo
import xarray as xr
import pandas as pd
import numpy as np
from typing import Dict, Any, List

pacific = ZoneInfo("America/Los_Angeles")

def detect_var(
    ds: xr.Dataset,
    keywords: List[str]
):
    """
    Find the first data variable name containing all given keywords.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset whose variable names will be searched.
    keywords : list of str
        List of case-insensitive substrings that must all appear in
        the variable name.

    Returns
    -------
    str or None
        Name of the first matching variable, or None if no match is found.
    """
    for v in ds.data_vars:
        name = v.lower()
        if all(k in name for k in keywords):
            return v
    return None


def fetch_ww3_timeseries(
    lat_valid: float,
    lon_valid: float,
    debug: bool = False
):
    """
    Extract a WW3 timeseries at a validated grid point.

    Parameters
    ----------
    lat_valid : float
        Latitude of a known valid WW3 ocean grid point.
    lon_valid : float
        Longitude in 0-360 convention for the WW3 grid point.
    debug : bool, optional
        If True, prints extra information useful for development.

    Returns
    -------
    pandas.DataFrame
        Hourly WW3 time series with columns:
        - Hs_m : Significant wave height (m)
        - Tp_s : Peak period (s)
        - Dir_deg : Mean wave direction (degrees)
        Index is datetime64 (tz-naive, Pacific-local timestamps).
    """
    if debug:
        return pd.DataFrame({
            "Hs_m":[1,1.5,2,2.5],
            "Tp_s":[10,11,12,13],
            "Dir_deg":[270,260,250,240]
        }, index=pd.date_range("2025-01-01", periods=4, freq="H"))

    # Open remote WW3 dataset
    url = "https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/WW3/Global/Best"
    ds = xr.open_dataset(url, engine="netcdf4")

    # Detect coordinate names
    latname = next(c for c in ds.coords if "lat" in c.lower())
    lonname = next(c for c in ds.coords if "lon" in c.lower())

    # Detect WW3 variables robustly  
    var_hs  = detect_var(ds, ["significant", "combined", "wave"])
    var_tp  = detect_var(ds, ["primary", "mean", "period"])
    var_dir = detect_var(ds, ["primary", "direction"])

    if not all([var_hs, var_tp, var_dir]):
        raise RuntimeError(
            f"Could not detect WW3 wave variables. Available vars:\n{list(ds.data_vars)}"
        )

    # Extract raw time series
    hs = ds[var_hs].sel({latname: lat_valid, lonname: lon_valid}, method="nearest").to_pandas()
    tp = ds[var_tp].sel({latname: lat_valid, lonname: lon_valid}, method="nearest").to_pandas()
    dr = ds[var_dir].sel({latname: lat_valid, lonname: lon_valid}, method="nearest").to_pandas()

    # Build dataframe
    df = pd.DataFrame({
        "Hs_m": hs,
        "Tp_s": tp,
        "Dir_deg": dr
    }).dropna()

    # Convert UTC → Pacific, then drop timezone
    df.index = (
        df.index.tz_localize("UTC")
                .tz_convert(pacific)
                .tz_localize(None)
    )

    return df


def fetch_cdip_029():
    """
    Fetch and process CDIP 029 (Point Bonita) historical swell data.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        3-hourly CDIP observations indexed by datetime (UTC-naive)
        with columns:
        - Hs_029_m : Significant wave height (m)
        - Tp_029_s : Peak period (s)
        - Dir_029_deg : Mean wave direction (degrees)
    """
    cdip_url = (
        "https://thredds.cdip.ucsd.edu/thredds/dodsC/"
        "cdip/archive/029p1/029p1_historic.nc"
    )

    ds = xr.open_dataset(cdip_url, engine="netcdf4")

    # --- Detect time + variables ---
    time_var = next(c for c in ds.coords if "time" in c.lower())
    var_hs  = next(v for v in ds.data_vars if "wavehs"  in v.lower())
    var_tp  = next(v for v in ds.data_vars if "wavetp"  in v.lower())
    var_dir = next(v for v in ds.data_vars if "wavedir" in v.lower() or "wavedp" in v.lower())

    # --- Convert to pandas ---
    t = pd.to_datetime(ds[time_var].values).tz_localize(None)

    cdip_df = pd.DataFrame({
        "Hs_029_m": ds[var_hs].values,
        "Tp_029_s": ds[var_tp].values,
        "Dir_029_deg": ds[var_dir].values,
    }, index=t).dropna()

    # --- Downsample to match WW3's 3-hourly spacing ---
    cdip_3h = cdip_df.resample("3h").mean()

    return cdip_3h

def fetch_tide_predictions(
    station_id: str,
     days: int = 14):
    """
    Fetch tide predictions from NOAA CO-OPS and return hourly tide heights
    in Pacific local time (PST/PDT).

    Parameters
    ----------
    station_id : str
        NOAA CO-OPS station identifier (e.g., "9414958").
    days : int, optional
        Number of days of tide predictions to request. Default is 14.

    Returns
    -------
    pandas.DataFrame
        Tide forecast indexed by localized datetime (Pacific), containing:
        - tide_height : float  Tide height in feet (MLLW)
    """
    station = Station(id=station_id)

    # Use Pacific time for begin/end dates
    begin = datetime.now(pacific)
    end = begin + timedelta(days=days)

    # Fetch data - use lst_ldt which returns local time for the station
    df = station.get_data(
        begin_date=begin.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        product="predictions",
        datum="MLLW",
        interval="h",
        units="english",
        time_zone="lst_ldt"  # Local standard/daylight time
    )

    df.rename(columns={'v':'tide_height'}, inplace=True)

    # The noaa_coops library returns timezone-naive datetimes
    if df.index.tz is None:
        # Localize to Pacific time (assumes the times are already in Pacific)
        df.index = df.index.tz_localize(pacific).tz_localize(None)
    else:
        # If it somehow has timezone info, convert to Pacific
        df.index = df.index.tz_convert(pacific).tz_localize(None)

    return df

def fetch_wind_forecast(
    lat: float, 
    lon: float):
    """
    Fetch NWS wind forecast and return hourly wind speed and direction
    in Pacific local time (PST/PDT).

    Parameters
    ----------
    lat : float
        Latitude of forecast location (decimal degrees).
    lon : float
        Longitude of forecast location (decimal degrees).

    Returns
    -------
    pandas.DataFrame
        Wind forecast indexed by localized datetime (Pacific), containing:
        - wind_speed : float   Wind speed in mph
        - wind_direction : str Wind direction as compass abbreviation (e.g., "NW")
    """
    headers = {'User-Agent': 'BolinasSurfForecast/1.0 (surfforecast@example.com)'}

    # Get grid point
    point_url = f"https://api.weather.gov/points/{lat},{lon}"
    point_response = requests.get(point_url, headers=headers, timeout=30)
    point_response.raise_for_status()
    point_data = point_response.json()

    # Get hourly forecast
    forecast_url = point_data['properties']['forecastHourly']
    forecast_response = requests.get(forecast_url, headers=headers, timeout=30)
    forecast_response.raise_for_status()
    forecast_data = forecast_response.json()

    # Parse into DataFrame
    periods = forecast_data['properties']['periods']

    wind_data = []
    for period in periods:
        # Parse datetime with timezone awareness, then convert to Pacific
        dt = pd.to_datetime(period['startTime'], utc=True).tz_convert(pacific).tz_localize(None)

        wind_data.append({
            'datetime': dt,
            'wind_speed': float(period['windSpeed'].split()[0]),
            'wind_direction': period['windDirection'],
            'temperature': period['temperature']
        })

    df = pd.DataFrame(wind_data).set_index("datetime").sort_index()
    return df

def fetch_sunrise_sunset(
    lat: float, 
    lon: float, 
    days: int = 14):
    """
    Fetch sunrise and sunset times for a given location, convert timestamps
    to Pacific local time (PST/PDT), and return daily first-light and
    last-light estimates.

    Parameters
    ----------
    lat : float
        Latitude of location (decimal degrees).
    lon : float
        Longitude of location (decimal degrees).
    days : int, optional
        Number of days to retrieve sunrise/sunset data for. Default is 14.

    Returns
    -------
    pandas.DataFrame
        Daily sunrise/sunset table indexed by date, containing:
        - first_light : datetime (Pacific)
        - last_light  : datetime (Pacific)
    """
    sun_data = []
    start_date = datetime.now(pacific).date()  # Use Pacific time for start date

    for i in range(days):
        date = start_date + timedelta(days=i)
        url = f"https://api.sunrise-sunset.org/json?lat={lat}&lng={lon}&date={date}&formatted=0"

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        sunrise = pd.to_datetime(data['results']['sunrise'], utc=True).tz_convert(pacific).tz_localize(None)
        sunset = pd.to_datetime(data['results']['sunset'], utc=True).tz_convert(pacific).tz_localize(None)
        first_light = pd.to_datetime(data['results']['civil_twilight_begin'], utc=True).tz_convert(pacific).tz_localize(None)
        last_light = pd.to_datetime(data['results']['civil_twilight_end'], utc=True).tz_convert(pacific).tz_localize(None)

        sun_data.append({
            'date': date,
            'sunrise': sunrise,
            'sunset': sunset,
            'first_light': first_light,
            'last_light': last_light,
        })

    df = pd.DataFrame(sun_data).set_index("date").sort_index()
    return df


def fetch_data_wrapper(data_sources: Dict):
    """
    Fetch all raw forecast input datasets (swell, tides, wind, sun),
    using the configuration values contained in `data_sources`.

    Parameters
    ----------
    data_sources : dict
        Configuration dictionary for required data-fetch parameters.

    Returns
    -------
    dict
        Dictionary of raw dataframes with keys:
            "ww3" : pd.DataFrame offshore wave forecast
            "cdip": pd.DataFrame observed buoy history
            "tide": pd.DataFrame tide predictions
            "wind": pd.DataFrame wind forecast
            "sun" : pd.DataFrame sunrise / sunset times
    """

    # 1. offshore swell (fast extractor)
    ww3_df = fetch_ww3_timeseries(
        data_sources["ww3_lat"],
        data_sources["ww3_lon"],
        debug=data_sources.get("debug", False)
    )

    # 2. buoy observations (CDIP 029)
    cdip_df = fetch_cdip_029()

    # 3. tides
    tide_df = fetch_tide_predictions(
        data_sources["tide_station"]
    )

    # 4. wind (NWS API)
    wind_df = fetch_wind_forecast(
        data_sources["location_lat"],
        data_sources["location_lon"]
    )

    # 5. sunrise/sunset (NOAA external API)
    sun_df = fetch_sunrise_sunset(
        data_sources["location_lat"],
        data_sources["location_lon"]
    )

    return {
        "ww3": ww3_df,
        "cdip": cdip_df,
        "tide": tide_df,
        "wind": wind_df,
        "sun": sun_df,
    }
