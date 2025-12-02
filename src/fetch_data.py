# IMPORTS
import requests
from datetime import datetime, timedelta
from noaa_coops import Station
from zoneinfo import ZoneInfo
import xarray as xr
import pandas as pd
import numpy as np

pacific = ZoneInfo("America/Los_Angeles")

# DATA FETCHING FUNCTIONS

def fetch_ww3_timeseries(lat, lon, start_deg=0.3, step=0.05, max_deg=2.0):
    """
    High-level WW3 extraction function.

    Automatically:
      • Converts lon → 0–360
      • Detects coord names
      • Detects bulk wave variables (Hs, Tp, Dir)
      • Finds nearest ocean grid cell (expanding search)
      • Extracts time series and returns a clean DataFrame

    Returns:
        DataFrame with columns:
            Hs_m, Tp_s, Dir_deg
    """
    url="https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/WW3/Global/Best"

    # --- Standardize longitude to WW3 convention ---
    lon_pt = lon + 360 if lon < 0 else lon

    # --- Open dataset ---
    ds = xr.open_dataset(url, engine="netcdf4")

    # --- Detect coordinate names ---
    latname = next(c for c in ds.coords if "lat" in c.lower())
    lonname = next(c for c in ds.coords if "lon" in c.lower())

    # --- Detect WW3 variables robustly ---
    def detect_var(keywords):
        for v in ds.data_vars:
            name = v.lower()
            if all(k in name for k in keywords):
                return v
        return None

    var_hs  = detect_var(["significant", "combined", "wave"])
    var_tp  = detect_var(["primary", "mean", "period"])
    var_dir = detect_var(["primary", "direction"])

    if not all([var_hs, var_tp, var_dir]):
        raise RuntimeError("Could not detect WW3 variable names.")

    # --- Detect time dimension (time, time1, etc.) ---
    var0 = ds[var_hs]
    time_dim = next(d for d in var0.dims if "time" in d.lower())

    # --- Search for nearest valid ocean grid ---
    def try_point(lat_try, lon_try):
        try:
            sub = ds[var_hs].sel({latname: lat_try, lonname: lon_try}, method="nearest")
            v = sub.isel(**{time_dim: 0}).values
            return sub if not np.isnan(v) else None
        except Exception:
            return None

    found = None
    radius = start_deg

    while radius <= max_deg and found is None:
        for dlat in np.arange(-radius, radius + step, step):
            for dlon in np.arange(-radius, radius + step, step):
                candidate = try_point(lat + dlat, lon_pt + dlon)
                if candidate is not None:
                    found = candidate
                    break
            if found is not None:
                break
        radius += step

    if found is None:
        raise RuntimeError(f"No valid WW3 ocean grid found within ±{max_deg}°")

    # Extract the valid lat/lon used
    lat_valid = float(found[latname])
    lon_valid = float(found[lonname])

    # --- Extract full time series from valid grid point ---
    hs  = ds[var_hs ].sel({latname: lat_valid, lonname: lon_valid}, method="nearest").to_pandas()
    tp  = ds[var_tp ].sel({latname: lat_valid, lonname: lon_valid}, method="nearest").to_pandas()
    dr  = ds[var_dir].sel({latname: lat_valid, lonname: lon_valid}, method="nearest").to_pandas()

    df = pd.DataFrame({
        "Hs_m": hs,
        "Tp_s": tp,
        "Dir_deg": dr
    }).dropna()

    df.index = (
    df.index.tz_localize("UTC")    
           .tz_convert(pacific).tz_localize(None)    
)

    return df

def fetch_cdip_029():
    """
    Fetch and process CDIP 029 (Point Bonita) historical swell data.
    Output:
        cdip_3h : DataFrame indexed by datetime (UTC-naive),
                  with columns: Hs_029_m, Tp_029_s, Dir_029_deg
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

def fetch_tide_predictions(station_id, days=14):
    """Fetch tide predictions in Pacific time (PST/PDT)"""
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

def fetch_wind_forecast(lat, lon):
    """Fetch NWS wind forecast in Pacific time (PST/PDT)"""
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

def fetch_sunrise_sunset(lat, lon, days=14):
    """Fetch sunrise/sunset data and convert to Pacific time (PST/PDT)"""
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