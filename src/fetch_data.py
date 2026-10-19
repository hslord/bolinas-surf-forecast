# IMPORTS
import ssl
import certifi
import requests
from requests.adapters import HTTPAdapter
from datetime import datetime, timedelta
from noaa_coops import Station
from zoneinfo import ZoneInfo
import xarray as xr
import pandas as pd
import time
from typing import Dict
from reference_functions import status

pacific = ZoneInfo("America/Los_Angeles")

# THREDDS servers (UCAR, CDIP) have started rejecting requests with missing
# or generic User-Agent headers (e.g. "python-requests", "GitHub-Actions") as
# bot/scraper traffic. The netcdf4 engine can't set custom headers, so OPeNDAP
# datasets are opened via the pydap engine with a descriptive User-Agent instead.
DAP_USER_AGENT = "bolinas-surf-forecast/1.0 (+https://github.com/hslord/bolinas-surf-forecast)"

# thredds.ucar.edu's cert chains through this intermediate CA (emSign, issued
# under their 2024 "Root TLS CA - G1"), which certifi doesn't ship yet even
# though it's a legitimate, currently-valid CA cert (expires 2039). Neither
# certifi nor the OS default trust store on GitHub Actions' hosted Python
# recognizes it, causing CERTIFICATE_VERIFY_FAILED even though the leaf cert
# itself is fine. Pin it explicitly alongside certifi's normal bundle rather
# than depending on ambient trust-store contents.
# To refresh/verify: openssl s_client -connect thredds.ucar.edu:443 -showcerts
_EXTRA_TRUSTED_CA_PEM = """\
-----BEGIN CERTIFICATE-----
MIIFuzCCBKOgAwIBAgIQAPerZVWeEIEt9khDR1hqJTANBgkqhkiG9w0BAQsFADBn
MQswCQYDVQQGEwJJTjETMBEGA1UECxMKZW1TaWduIFBLSTElMCMGA1UEChMcZU11
ZGhyYSBUZWNobm9sb2dpZXMgTGltaXRlZDEcMBoGA1UEAxMTZW1TaWduIFJvb3Qg
Q0EgLSBHMTAeFw0yNDA3MTExNDUyMjFaFw0zOTA3MDkxODMwMDBaMFYxCzAJBgNV
BAYTAklOMSUwIwYDVQQKDBxlTXVkaHJhIFRlY2hub2xvZ2llcyBMaW1pdGVkMSAw
HgYDVQQDDBdlbVNpZ24gUm9vdCBUTFMgQ0EgLSBHMTCCAiIwDQYJKoZIhvcNAQEB
BQADggIPADCCAgoCggIBAMPm5mm136ImUw1EdfpP+tY0OmCgnH1eu6MvMr4P/isE
dsRtmVjH653JHnPpPeUjGSbvLKGCabv9YvVBShZOvrI3GDIPrTguyUhONiygrjYS
McbkG0WouoPFybNE1IjYleAuQ8mdz5gsZ96lCfaAfmy1C9pidUBweF9KWQ31HzQb
Sak5T+0yqCyxNGZNdnrXodEwuH9kj+MOt3wL86K8KYH6CJv3uyhfa0UQw+Ho3tNH
/DwdXs1m9FUViimqWd9GZcscYB7um484iVDVjimNd9cMY1bEMAFpCwzyycWvrjdj
V74q48eCITrg41GSS6Wr0OxKBY3Mh1Yt+tl30MWQVcb0yZgt8e0Oq2a25LZYpTz7
eMBJ9+Zm+AJKE75vGKd0lwa68j9m8vb2+l1w3tbl1OAZGGA5D7FcBcGiIFAXm/aq
s0lEYSZmp/UzGrwtswtPKnqhR87EolO+AMdvGJAnbhqsvrsX2FcxaV7Of2vIWDJ6
hdpWaHa5YjNXpCgDpiaKRd5F6opQsb8WmQdUswZgIocLb4egKzR/ndYX6xlxnL/h
U7W2HLqXckLR1HNCFz6wLgLuHmi7Zf3x760BbP0HwRIWryGUDJhAmh+gg3JNT/HR
BL9bYpBKL7yjM+IJ70Q4tRFWwlpGMziRNNOLeOkem1DBxDl6+Q3A0v5R/HLVMG0p
AgMBAAGjggFyMIIBbjAfBgNVHSMEGDAWgBT77w2GnrDj3am58SEXfz788HcrGjAd
BgNVHQ4EFgQUZB3J2PjE7ARLIoH1MppeueeTUvkwHQYDVR0lBBYwFAYIKwYBBQUH
AwEGCCsGAQUFBwMCMD0GA1UdIAQ2MDQwMgYEVR0gADAqMCgGCCsGAQUFBwIBFhxo
dHRwOi8vcmVwb3NpdG9yeS5lbXNpZ24uY29tMHUGCCsGAQUFBwEBBGkwZzAiBggr
BgEFBQcwAYYWaHR0cDovL29jc3AuZW1zaWduLmNvbTBBBggrBgEFBQcwAoY1aHR0
cDovL3JlcG9zaXRvcnkuZW1zaWduLmNvbS9jZXJ0cy9lbVNpZ25Sb290Q0FHMS5j
cnQwMwYDVR0fBCwwKjAooCagJIYiaHR0cDovL2NybC5lbXNpZ24uY29tP1Jvb3RD
QUcxLmNybDAOBgNVHQ8BAf8EBAMCAQYwEgYDVR0TAQH/BAgwBgEB/wIBATANBgkq
hkiG9w0BAQsFAAOCAQEABj/hFjkky/uOQZ0k3peoiVG7+qwhyC8TgaTLuDZisrnX
7S1GvSeiOs8xeKyp8dnCbE6S1XHFtJbZi8PtSi+3haElYjkSiGixXd8PYLtNmCn9
cK5ymSBVssavkolI3tONt1t2kzhyAaZItZqGOVJoeRxbSkM/vTKWKqlFYHn/J00u
FG7ii70vEyF3Ly0scGHCpD1BFoYajQksxvQ1m3rtKKMwBucZNTejdA0Npf0iSiDX
9HUrEEqodKnRLUMSQz4K0Bd1kB3AVn8tT5JAc1HlgmxQ1OTTz5NcX60ATo4eQIHT
a9Fb1/Op2vpQ/BhGF0vxOvg3nwleIchABUbyj9XnxQ==
-----END CERTIFICATE-----
"""


def _dap_ssl_context() -> ssl.SSLContext:
    """SSL context using certifi's CA bundle plus the pinned emSign intermediate."""
    context = ssl.create_default_context(cafile=certifi.where())
    context.load_verify_locations(cadata=_EXTRA_TRUSTED_CA_PEM)
    return context


class _PinnedCAAdapter(HTTPAdapter):
    """HTTPAdapter that verifies TLS against certifi + the pinned extra CA."""

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = _dap_ssl_context()
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs["ssl_context"] = _dap_ssl_context()
        return super().proxy_manager_for(*args, **kwargs)


def _retry(func, retries=3, delay=2, backoff=2):
    """Retry a function with exponential backoff."""
    for attempt in range(retries):
        try:
            return func()
        except (requests.RequestException, OSError):
            if attempt == retries - 1:
                raise
            status(f"Retry {attempt + 1}/{retries} after {delay}s...")
            time.sleep(delay)
            delay *= backoff


def _open_dap_dataset(url: str):
    """Open an OPeNDAP dataset via pydap with a descriptive User-Agent, with retries."""

    def _open():
        session = requests.Session()
        session.headers.update({"User-Agent": DAP_USER_AGENT})
        session.mount("https://", _PinnedCAAdapter())
        store = xr.backends.PydapDataStore.open(url, session=session)
        return xr.open_dataset(store)

    return _retry(_open)


def fetch_ww3_timeseries(
    lat_valid: float,
    lon_valid: float,
    max_partitions: int,
    forecast_hours: int,
    url: str,
):
    """
    Extract a partitioned WW3 swell timeseries at a validated grid point.

    Returns
    -------
    pandas.DataFrame
        Long-form dataframe with:
        - index: forecast-valid time (tz-naive, Pacific)
        - columns: swell_idx, Hs_m, Tp_s, Dir_deg
        - one row per (time, swell_idx)
    """
    status(f"Opening WW3 Dataset at {url[:50]}...")

    try:

        # Open dataset
        ds = _open_dap_dataset(url)

        # Detect coordinate names
        latname = next(c for c in ds.coords if "lat" in c.lower())
        lonname = next(c for c in ds.coords if "lon" in c.lower())

        # Longitude convention handling
        lon_coord = ds[lonname]
        lon_val = float(lon_valid)
        if float(lon_coord.max()) <= 180.0 and lon_val > 180.0:
            lon_val -= 360.0

        # Partitioned swell variables
        var_hs = "Significant_height_of_swell_waves_ordered_sequence_of_data"
        var_tp = "Mean_period_of_swell_waves_ordered_sequence_of_data"
        var_dir = "Direction_of_swell_waves_ordered_sequence_of_data"

        part_dim = next(d for d in ds[var_hs].dims if "sequence" in d.lower())

        # Forecast time window (UTC, tz-naive)
        start_pacific = datetime.now(pacific).replace(minute=0, second=0, microsecond=0)
        end_pacific = start_pacific + timedelta(hours=int(forecast_hours))

        start_utc = pd.Timestamp(start_pacific).tz_convert("UTC").tz_localize(None)
        end_utc = pd.Timestamp(end_pacific).tz_convert("UTC").tz_localize(None)

        # Lazy subset (forecast-valid time ONLY)
        sub = xr.Dataset(
            {
                "Hs_m": ds[var_hs],
                "Tp_s": ds[var_tp],
                "Dir_deg": ds[var_dir],
            }
        )

        # Nearest spatial point
        sub = sub.sel(
            {latname: lat_valid, lonname: lon_val},
            method="nearest",
        )

        # Forecast window — ONLY forecast-valid time
        sub = sub.sel({"time": slice(start_utc, end_utc)})

        # Partition limit
        sub = sub.isel({part_dim: slice(0, int(max_partitions))})

        # Align any variables that use time1 → time
        tvals = sub["time"].values

        for var in ["Hs_m", "Tp_s", "Dir_deg"]:
            if "time1" in sub[var].dims:
                aligned = (
                    sub[var]
                    .sel(time1=tvals)
                    .assign_coords({"time": ("time1", tvals)})
                    .swap_dims({"time1": "time"})
                )
                sub[var] = aligned

        # Drop time1 everywhere (hard guarantee)
        if "time1" in sub.coords:
            sub = sub.drop_vars("time1")

        # Load aligned dataset
        sub = sub.load()

        # Convert to long-form pandas
        sub = sub.reset_coords(drop=True)
        stacked = sub.stack(_row=("time", part_dim))

        df = stacked.to_dataframe()[["Hs_m", "Tp_s", "Dir_deg"]].reset_index()

        df = df.rename(columns={part_dim: "swell_idx"})

        df = df.dropna(subset=["Hs_m", "Tp_s", "Dir_deg"])

        # Convert time to Pacific tz-naive
        df["time"] = (
            pd.to_datetime(df["time"], utc=True)
            .dt.tz_convert(pacific)
            .dt.tz_localize(None)
        )

        df = df.set_index("time").sort_index()

        status(f"WW3 subset loaded: {len(df)} swell partitions found.")

        return df

    except (RuntimeError, KeyError, ValueError) as e:
        status(f"CRITICAL: WW3 data format is invalid or missing keys: {e}")
        raise RuntimeError(f"WW3 data validation failed: {e}") from e

    except Exception as e:
        status(f"CRITICAL: Unspecified ww3 load error: {e}")
        raise SystemExit(f"Pipeline stopped due to ww3 fetch failure: {e}") from e


def fetch_cdip_mop_forecast(mop_number, min_swell_frequency, forecast_model="ecmwf"):
    """
    Fetch the CDIP MOP spectral wave forecast for a nearshore output point.

    Connects to the CDIP THREDDS OPeNDAP server, selects the appropriate
    forecast model (ECMWF or NCEP), and filters to the swell frequency band.

    Parameters
    ----------
    mop_number : str
        MOP station identifier (e.g., 'MA147').
    min_swell_frequency : float
        Maximum frequency (Hz) to include in the swell band filter.
        Frequencies above this threshold are excluded.
    forecast_model : str, optional
        Which global wave model drives the MOP forecast boundary conditions.
        'ecmwf' (default) uses ECMWF HRES-WAM; 'ncep' uses NCEP WW3.

    Returns
    -------
    xarray.Dataset
        Swell-filtered MOP spectral dataset containing energy density,
        Fourier directional coefficients (a1, b1), bandwidth, and
        peak direction for each forecast timestep.
    """
    status(f"Fetching swell forecast for MOP {mop_number}...")

    try:
        # MOP THREDDS OPeNDAP
        if forecast_model == "ncep":
            fc_url = (
                "https://thredds.cdip.ucsd.edu/thredds/dodsC/"
                f"cdip/model/MOP_alongshore/{mop_number}_forecast.nc"
            )
        else:
            fc_url = (
                "https://thredds.cdip.ucsd.edu/thredds/dodsC/"
                f"cdip/model/MOP_alongshore/{mop_number}_ecmwf_fc.nc"
            )

        fc_swell = _open_dap_dataset(fc_url)

        # Filter to swell band (periods > 10s, i.e. frequencies < 0.1 Hz)
        fc_swell = fc_swell.sel(
            waveFrequency=fc_swell.waveFrequency[
                fc_swell.waveFrequency <= min_swell_frequency
            ]
        )

        status("Retrieved swell forecast.")

        return fc_swell

    except (RuntimeError, KeyError, ValueError) as e:
        status(f"CRITICAL: CDIP MOP data format is invalid or missing keys: {e}")
        raise RuntimeError(f"CDIP MOP data validation failed: {e}") from e

    except Exception as e:
        status(f"CRITICAL: Unspecified CDIP MOP load error: {e}")
        raise SystemExit(f"Pipeline stopped due to CDIP MOP fetch failure: {e}") from e


def fetch_tide_predictions(station_id: str, days: int = 14):
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
    status(f"Fetching tides for station {station_id}...")
    station = Station(id=station_id)

    # Use Pacific time for begin/end dates
    begin = datetime.now(pacific)
    end = begin + timedelta(days=days)

    try:
        # Fetch data - use lst_ldt which returns local time for the station
        df = station.get_data(
            begin_date=begin.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            product="predictions",
            datum="MLLW",
            interval="h",
            units="english",
            time_zone="lst_ldt",  # Local standard/daylight time
        )

        df.rename(columns={"v": "tide_height"}, inplace=True)

        # The noaa_coops library returns timezone-naive datetimes already in
        # Pacific local time (lst_ldt), so no localization is needed here.
        # Round-tripping through tz_localize(pacific) raises on DST transition
        # days, when a naive hour is ambiguous (fall back) or nonexistent
        # (spring forward).
        if df.index.tz is None:
            pass
        else:
            # If it somehow has timezone info, convert to Pacific
            df.index = df.index.tz_convert(pacific).tz_localize(None)

        status(f"Retrieved {len(df)} tide data points.")

        return df

    except (RuntimeError, KeyError, ValueError) as e:
        status(f"CRITICAL: Tide data format is invalid or missing keys: {e}")
        raise RuntimeError(f"Tide data validation failed: {e}") from e

    except Exception as e:
        status(f"CRITICAL: Unspecified tide load error: {e}")
        raise SystemExit(f"Pipeline stopped due to tide fetch failure: {e}") from e


def fetch_wind_forecast(lat: float, lon: float, days: int = 14):
    """
    Fetch wind forecast from Open-Meteo (hourly) with sustained wind and gusts.
    Uses 'best_match' to ensure a full 7-day forecast without gaps.

    Parameters
    ----------
    lat : float
        location latitude
    lon : float
        location longitude

    Returns
    -------
    pandas.DataFrame
        Wind forecast with sustained and gust forecasts in MPH
    """
    status(f"Fetching Open-Meteo wind data at {lat}, {lon}...")

    # Open-Meteo API URL - specifically requesting wind at 10m height
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"hourly=wind_speed_10m,wind_gusts_10m,wind_direction_10m&"
        f"wind_speed_unit=mph&"
        f"timezone=America/Los_Angeles&"
        f"forecast_days={days}"
    )

    try:

        def _fetch_wind():
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()

        data = _retry(_fetch_wind)

        hourly = data["hourly"]

        # Open-Meteo returns a dictionary of lists
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(hourly["time"]),
                "wind_speed": hourly["wind_speed_10m"],
                "wind_gust": hourly["wind_gusts_10m"],
                "wind_direction": hourly["wind_direction_10m"],
            }
        )

        # Open-Meteo returns Pacific times (timezone=America/Los_Angeles).
        # Just ensure they are tz-naive for consistency with the rest of the pipeline.
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()

        # Physical sanity check: gusts must be >= speed
        df["wind_gust"] = df[["wind_speed", "wind_gust"]].max(axis=1)

        status(f"Retrieved {len(df)} wind data points from Open-Meteo.")

        return df

    except (RuntimeError, KeyError, ValueError) as e:
        status(f"CRITICAL: Wind data format is invalid or missing keys: {e}")
        raise RuntimeError(f"Wind data validation failed: {e}") from e

    except Exception as e:
        status(f"CRITICAL: Unspecified wind load error: {e}")
        raise SystemExit(f"Pipeline stopped due to wind fetch failure: {e}") from e


def fetch_sunrise_sunset(lat: float, lon: float, days: int = 14):
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
    status(f"Fetching daylight for {days} days at {lat}, {lon}...")

    sun_data = []
    start_date = datetime.now(pacific).date()  # Use Pacific time for start date

    try:
        for i in range(days):
            date = start_date + timedelta(days=i)
            url = f"https://api.sunrise-sunset.org/json?lat={lat}&lng={lon}&date={date}&formatted=0"

            def _fetch_sun(_url=url):
                resp = requests.get(_url, timeout=30)
                resp.raise_for_status()
                return resp.json()

            data = _retry(_fetch_sun, retries=2, delay=1)

            sunrise = (
                pd.to_datetime(data["results"]["sunrise"], utc=True)
                .tz_convert(pacific)
                .tz_localize(None)
            )
            sunset = (
                pd.to_datetime(data["results"]["sunset"], utc=True)
                .tz_convert(pacific)
                .tz_localize(None)
            )
            first_light = (
                pd.to_datetime(data["results"]["civil_twilight_begin"], utc=True)
                .tz_convert(pacific)
                .tz_localize(None)
            )
            last_light = (
                pd.to_datetime(data["results"]["civil_twilight_end"], utc=True)
                .tz_convert(pacific)
                .tz_localize(None)
            )

            sun_data.append(
                {
                    "date": date,
                    "sunrise": sunrise,
                    "sunset": sunset,
                    "first_light": first_light,
                    "last_light": last_light,
                }
            )
            time.sleep(
                0.5
            )  # Wait half a second between requests to avoid getting blocked

        df = pd.DataFrame(sun_data).set_index("date").sort_index()

        status(f"Retrieved {len(df)} daylight data points.")

        return df

    except (RuntimeError, KeyError, ValueError) as e:
        status(f"Warning: sunrise/sunset data not as expected. Continuing without: {e}")
        return pd.DataFrame()

    except Exception as e:  # pylint: disable=broad-exception-caught
        status(f"Warning: Unspecified sunrise/sunset load error. Using empty data: {e}")
        return pd.DataFrame()


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
    results = {
        "ww3": None,
        "cdip_mop": None,
        "tide": None,
        "wind": None,
        "sun": None,
    }

    # Helper to run a fetch function and catch errors
    def safe_fetch(key, func, *args, **kwargs):
        results[key] = func(*args, **kwargs)
        status(f"Successfully fetched {key}")

    # offshore swell
    safe_fetch(
        "ww3",
        fetch_ww3_timeseries,
        data_sources["ww3_lat"],
        data_sources["ww3_lon"],
        data_sources["max_partitions"],
        data_sources["forecast_hours"],
        data_sources["ww3_url"],
    )

    # shallow water (15M) mop forecast
    safe_fetch(
        "cdip_mop",
        fetch_cdip_mop_forecast,
        data_sources["mop_number"],
        data_sources["min_swell_frequency"],
        data_sources["forecast_model"],
    )

    # tides
    safe_fetch("tide", fetch_tide_predictions, data_sources["tide_station"])

    # wind
    safe_fetch(
        "wind",
        fetch_wind_forecast,
        data_sources["location_lat"],
        data_sources["location_lon"],
    )

    # sunrise/sunset
    safe_fetch(
        "sun",
        fetch_sunrise_sunset,
        data_sources["location_lat"],
        data_sources["location_lon"],
    )

    return results
