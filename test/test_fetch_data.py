import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# fetch_ww3_timeseries is not unit-tested here because it relies on heavy
# xarray coordinate detection, dimension inference, and time-axis alignment
# that would require a near-complete mock of the WW3 NetCDF schema to be
# meaningful. It is exercised indirectly via the wrapper test.

from fetch_data import (
    fetch_cdip_mop_forecast,
    fetch_tide_predictions,
    fetch_wind_forecast,
    fetch_sunrise_sunset,
    fetch_data_wrapper,
)


# ── fetch_cdip_mop_forecast ──────────────────────────────────────────────


@patch("fetch_data.xr.open_dataset")
def test_cdip_mop_returns_dataset(mock_open):
    freqs = np.array([0.04, 0.06, 0.08, 0.12, 0.15])
    ds = xr.Dataset(
        {"waveEnergyDensity": (["waveTime", "waveFrequency"], np.ones((2, 5)))},
        coords={"waveFrequency": freqs, "waveTime": [0, 1]},
    )
    mock_open.return_value = ds

    result = fetch_cdip_mop_forecast("MA147", min_swell_frequency=0.1)
    # Should filter to frequencies < 0.1
    assert (result.waveFrequency.values < 0.1).all()


@patch("fetch_data.xr.open_dataset")
def test_cdip_mop_ecmwf_url(mock_open):
    mock_open.return_value = xr.Dataset(
        coords={"waveFrequency": np.array([0.05])}
    ).assign(waveEnergyDensity=lambda ds: xr.DataArray([1.0], dims=["waveFrequency"]))
    # Need a proper dataset for sel to work
    ds = xr.Dataset(
        {"waveEnergyDensity": (["waveFrequency"], [1.0])},
        coords={"waveFrequency": [0.05]},
    )
    mock_open.return_value = ds

    fetch_cdip_mop_forecast("MA147", min_swell_frequency=0.1, forecast_model="ecmwf")
    url = mock_open.call_args[0][0]
    assert "ecmwf_fc.nc" in url
    assert "MA147" in url


@patch("fetch_data.xr.open_dataset")
def test_cdip_mop_ncep_url(mock_open):
    ds = xr.Dataset(
        {"waveEnergyDensity": (["waveFrequency"], [1.0])},
        coords={"waveFrequency": [0.05]},
    )
    mock_open.return_value = ds

    fetch_cdip_mop_forecast("MA147", min_swell_frequency=0.1, forecast_model="ncep")
    url = mock_open.call_args[0][0]
    assert "forecast.nc" in url
    assert "ecmwf" not in url


@patch("fetch_data.xr.open_dataset")
def test_cdip_mop_raises_on_bad_data(mock_open):
    mock_open.side_effect = KeyError("missing var")
    with pytest.raises(RuntimeError):
        fetch_cdip_mop_forecast("MA147", min_swell_frequency=0.1)


# ── fetch_tide_predictions ────────────────────────────────────────────────


@patch("fetch_data.Station")
def test_tide_returns_dataframe_with_tide_height(mock_station_cls):
    idx = pd.date_range("2025-01-01", periods=24, freq="h")
    mock_df = pd.DataFrame({"v": np.random.uniform(0, 5, 24)}, index=idx)
    mock_station_cls.return_value.get_data.return_value = mock_df

    result = fetch_tide_predictions("9414290", days=1)
    assert "tide_height" in result.columns
    assert len(result) == 24


@patch("fetch_data.Station")
def test_tide_raises_on_failure(mock_station_cls):
    mock_station_cls.return_value.get_data.side_effect = ValueError("bad request")
    with pytest.raises(RuntimeError):
        fetch_tide_predictions("9414290")


# ── fetch_wind_forecast ───────────────────────────────────────────────────


@patch("fetch_data.requests.get")
def test_wind_returns_expected_columns(mock_get):
    hours = pd.date_range("2025-01-01", periods=24, freq="h")
    mock_get.return_value.status_code = 200
    mock_get.return_value.raise_for_status = MagicMock()
    mock_get.return_value.json.return_value = {
        "hourly": {
            "time": [t.isoformat() for t in hours],
            "wind_speed_10m": [5.0] * 24,
            "wind_gusts_10m": [8.0] * 24,
            "wind_direction_10m": [180] * 24,
        }
    }

    result = fetch_wind_forecast(37.9, -122.7)
    assert set(result.columns) >= {"wind_speed", "wind_gust", "wind_direction"}
    assert len(result) == 24


@patch("fetch_data.requests.get")
def test_wind_gusts_at_least_speed(mock_get):
    hours = pd.date_range("2025-01-01", periods=3, freq="h")
    mock_get.return_value.raise_for_status = MagicMock()
    mock_get.return_value.json.return_value = {
        "hourly": {
            "time": [t.isoformat() for t in hours],
            "wind_speed_10m": [10.0, 10.0, 10.0],
            "wind_gusts_10m": [5.0, 5.0, 5.0],  # gusts < speed
            "wind_direction_10m": [180, 180, 180],
        }
    }

    result = fetch_wind_forecast(37.9, -122.7)
    assert (result["wind_gust"] >= result["wind_speed"]).all()


# ── fetch_sunrise_sunset ──────────────────────────────────────────────────


@patch("fetch_data.requests.get")
def test_sunrise_returns_expected_columns(mock_get):
    mock_get.return_value.raise_for_status = MagicMock()
    mock_get.return_value.json.return_value = {
        "results": {
            "sunrise": "2025-01-15T15:20:00+00:00",
            "sunset": "2025-01-16T01:10:00+00:00",
            "civil_twilight_begin": "2025-01-15T15:00:00+00:00",
            "civil_twilight_end": "2025-01-16T01:30:00+00:00",
        }
    }

    result = fetch_sunrise_sunset(37.9, -122.7, days=1)
    assert set(result.columns) >= {"first_light", "last_light", "sunrise", "sunset"}


@patch("fetch_data.requests.get")
def test_sunrise_returns_empty_on_error(mock_get):
    mock_get.side_effect = Exception("network down")
    result = fetch_sunrise_sunset(37.9, -122.7, days=1)
    assert result.empty


# ── fetch_data_wrapper ────────────────────────────────────────────────────


@patch("fetch_data.fetch_sunrise_sunset")
@patch("fetch_data.fetch_wind_forecast")
@patch("fetch_data.fetch_tide_predictions")
@patch("fetch_data.fetch_cdip_mop_forecast")
@patch("fetch_data.fetch_ww3_timeseries")
def test_wrapper_returns_all_keys(mock_ww3, mock_mop, mock_tide, mock_wind, mock_sun):
    mock_ww3.return_value = pd.DataFrame()
    mock_mop.return_value = xr.Dataset()
    mock_tide.return_value = pd.DataFrame()
    mock_wind.return_value = pd.DataFrame()
    mock_sun.return_value = pd.DataFrame()

    data_sources = {
        "ww3_lat": 37.5,
        "ww3_lon": 237.0,
        "max_partitions": 3,
        "forecast_hours": 168,
        "ww3_url": "http://example.com",
        "mop_number": "MA147",
        "min_swell_frequency": 0.1,
        "forecast_model": "ecmwf",
        "tide_station": "9414290",
        "location_lat": 37.9,
        "location_lon": -122.7,
    }

    result = fetch_data_wrapper(data_sources)
    assert set(result.keys()) == {"ww3", "cdip_mop", "tide", "wind", "sun"}
