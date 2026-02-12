# IMPORTS
import xarray as xr
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

pacific = ZoneInfo("America/Los_Angeles")


def status(msg: str):
    """Timestamped print for logs."""
    now = datetime.now(pacific).strftime("%H:%M:%S")
    print(f"[{now}] 🌊 {msg}")


def fetch_ww3_timeseries_latlon(lat, lon, start_deg=0.3, step=0.05, max_deg=2.0):
    """
    Returns the nearest valid WW3 ocean model grid coordinates to (lat, lon).
    Use this once, save the result in config.yaml, and skip future searches.
    """

    url = "https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/WW3/Global/Best"

    # Convert lon → 0–360
    lon_pt = lon + 360 if lon < 0 else lon

    # Load dataset
    ds = xr.open_dataset(url, engine="netcdf4")

    # Detect coordinate names
    latname = next(c for c in ds.coords if "lat" in c.lower())
    lonname = next(c for c in ds.coords if "lon" in c.lower())

    # Detect Hs variable
    var_hs = next(v for v in ds.data_vars if "significant" in v.lower())

    # Detect time dimension
    time_dim = next(d for d in ds[var_hs].dims if "time" in d.lower())

    # Try nearest point search
    def try_point(lat_try, lon_try):
        try:
            sub = ds[var_hs].sel({latname: lat_try, lonname: lon_try}, method="nearest")
            arr = sub.isel(**{time_dim: 0}).values
            val = np.ravel(arr)[0]  # <-- SAFE SCALAR EXTRACTION
            return sub if not np.isnan(val) else None
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
        raise RuntimeError(
            f"No valid WW3 grid found within ±{max_deg}° of {lat}, {lon}"
        )

    # Extract the final valid WW3 grid cell
    lat_valid = float(found[latname])
    lon_valid = float(found[lonname])

    return lat_valid, lon_valid
