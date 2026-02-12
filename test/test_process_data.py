import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from process_data import (
    classify_wind_relative_to_coast,
    expand_timeseries_data,
    compute_swell_score,
    predict_bolinas_surf_height,
    calculate_surf_score,
    aggregate_hourly_partitions,
)


# ── classify_wind_relative_to_coast ──────────────────────────────────────


def test_wind_offshore():
    result = classify_wind_relative_to_coast(
        wind_direction=345,
        coast_orientation_deg=165,
        offshore_threshold=45.0,
        onshore_threshold=135.0,
    )
    assert result == "offshore"


def test_wind_onshore():
    result = classify_wind_relative_to_coast(
        wind_direction=165,
        coast_orientation_deg=165,
        offshore_threshold=45.0,
        onshore_threshold=135.0,
    )
    assert result == "onshore"


def test_wind_crosshore():
    result = classify_wind_relative_to_coast(
        wind_direction=75,
        coast_orientation_deg=165,
        offshore_threshold=45.0,
        onshore_threshold=135.0,
    )
    assert result == "crosshore"


def test_wind_nan_input():
    result = classify_wind_relative_to_coast(
        wind_direction=np.nan,
        coast_orientation_deg=165,
        offshore_threshold=45.0,
        onshore_threshold=135.0,
    )
    assert pd.isna(result)


# ── expand_timeseries_data ───────────────────────────────────────────────


def test_expand_fills_gaps():
    idx = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 03:00"])
    df = pd.DataFrame({"val": [1.0, 4.0]}, index=idx)
    result = expand_timeseries_data(df, freq="h")
    assert len(result) == 4  # 00, 01, 02, 03


def test_expand_empty_returns_empty():
    result = expand_timeseries_data(pd.DataFrame())
    assert result.empty


def test_expand_handles_direction_wrap():
    """Direction interpolation should handle 350->10 without going through 180."""
    idx = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 02:00"])
    df = pd.DataFrame({"Dir_deg": [350.0, 10.0]}, index=idx)
    result = expand_timeseries_data(df, freq="h")
    # Midpoint should be near 0/360, not near 180
    mid = result["Dir_deg"].iloc[1]
    assert mid < 30 or mid > 330


# ── compute_swell_score ──────────────────────────────────────────────────


@pytest.fixture
def _spectral_config():
    return {
        "spectral_scoring": {
            "hs_min_m": 0.3,
            "hs_full_credit_m": 1.5,
            "tp_min_s": 10.0,
            "tp_full_credit_s": 16.0,
            "spread_min_deg": 5.0,
            "spread_max_deg": 20.0,
            "w_hs": 0.40,
            "w_tp": 0.40,
            "w_sp": 0.20,
        }
    }


def _make_mop_dataset(n_times=3, n_freqs=5, energy_scale=1.0):
    """Build a minimal MOP-like xarray dataset for testing."""
    freqs = np.linspace(0.04, 0.09, n_freqs)
    times = pd.date_range("2025-01-01", periods=n_times, freq="3h")
    bw = np.full(n_freqs, 0.01)

    energy = np.full((n_times, n_freqs), energy_scale)
    a1 = np.full((n_times, n_freqs), 0.9)  # high directional concentration
    b1 = np.full((n_times, n_freqs), 0.1)

    return xr.Dataset(
        {
            "waveEnergyDensity": (["waveTime", "waveFrequency"], energy),
            "waveA1Value": (["waveTime", "waveFrequency"], a1),
            "waveB1Value": (["waveTime", "waveFrequency"], b1),
            "waveBandwidth": (["waveFrequency"], bw),
            "waveDp": (["waveTime"], [270.0] * n_times),
        },
        coords={
            "waveFrequency": freqs,
            "waveTime": times.values,
        },
    )


def test_swell_score_returns_expected_columns(_spectral_config):
    ds = _make_mop_dataset()
    result = compute_swell_score(ds, _spectral_config)
    expected_cols = {"hs_swell", "tp", "dp", "spread", "swell_score"}
    assert expected_cols.issubset(result.columns)


def test_swell_score_range(_spectral_config):
    ds = _make_mop_dataset()
    result = compute_swell_score(ds, _spectral_config)
    assert (result["swell_score"] >= 1).all()
    assert (result["swell_score"] <= 10).all()


def test_bigger_swell_scores_higher(_spectral_config):
    small = compute_swell_score(_make_mop_dataset(energy_scale=0.1), _spectral_config)
    big = compute_swell_score(_make_mop_dataset(energy_scale=10.0), _spectral_config)
    assert big["swell_score"].mean() > small["swell_score"].mean()


# ── predict_bolinas_surf_height ──────────────────────────────────────────


@pytest.fixture
def _nearshore_cfg():
    return {"range_factor": 0.15, "range_period_min": 12, "range_step": 0.01}


def test_surf_height_returns_min_max(_nearshore_cfg):
    result = predict_bolinas_surf_height(4.0, 14.0, _nearshore_cfg)
    assert "bolinas_surf_min_ft" in result
    assert "bolinas_surf_max_ft" in result
    assert result["bolinas_surf_max_ft"] >= result["bolinas_surf_min_ft"]


def test_surf_height_zero_for_flat(_nearshore_cfg):
    result = predict_bolinas_surf_height(0.0, 10.0, _nearshore_cfg)
    assert result["bolinas_surf_min_ft"] == 0.0
    assert result["bolinas_surf_max_ft"] == 0.0


def test_surf_height_nan_input(_nearshore_cfg):
    result = predict_bolinas_surf_height(np.nan, 14.0, _nearshore_cfg)
    assert result["bolinas_surf_min_ft"] == 0.0


def test_surf_height_longer_period_wider_range(_nearshore_cfg):
    short = predict_bolinas_surf_height(4.0, 10.0, _nearshore_cfg)
    long = predict_bolinas_surf_height(4.0, 18.0, _nearshore_cfg)
    short_range = short["bolinas_surf_max_ft"] - short["bolinas_surf_min_ft"]
    long_range = long["bolinas_surf_max_ft"] - long["bolinas_surf_min_ft"]
    assert long_range >= short_range


# ── calculate_surf_score ─────────────────────────────────────────────────


@pytest.fixture
def _surf_model_cfg():
    return {
        "wind": {
            "offshore_penalty_weight": 0.7,
            "crosshore_penalty_weight": 1.0,
            "onshore_penalty_weight": 1.0,
            "gust_weight": 0.3,
            "speed_floor": 4.0,
            "speed_range": 12.0,
            "penalty_min": 0.25,
        },
        "tide": {
            "optimal_height": 1.0,
            "sigma": 2.5,
            "penalty_min": 0.1,
        },
        "final_scoring": {
            "wind_impact_weight": 0.60,
            "tide_impact_weight": 0.60,
            "min_multiplier": 0.20,
            "secondary_penalty": 0.2,
        },
    }


def test_surf_score_returns_four_values(_surf_model_cfg):
    result = calculate_surf_score(7.0, 5.0, 8.0, "offshore", 1.0, _surf_model_cfg)
    assert len(result) == 4


def test_surf_score_range(_surf_model_cfg):
    final, _, wind, tide = calculate_surf_score(
        8.0, 3.0, 5.0, "offshore", 1.0, _surf_model_cfg
    )
    assert 1 <= final <= 10
    assert 1 <= wind <= 10
    assert 1 <= tide <= 10


def test_onshore_wind_penalizes_more(_surf_model_cfg):
    _, _, wind_off, _ = calculate_surf_score(
        7.0, 10.0, 15.0, "offshore", 1.0, _surf_model_cfg
    )
    _, _, wind_on, _ = calculate_surf_score(
        7.0, 10.0, 15.0, "onshore", 1.0, _surf_model_cfg
    )
    assert wind_off >= wind_on


def test_optimal_tide_no_penalty(_surf_model_cfg):
    _, _, _, tide_score = calculate_surf_score(
        7.0, 3.0, 5.0, "offshore", 1.0, _surf_model_cfg
    )
    assert tide_score == 10.0


# ── aggregate_hourly_partitions ──────────────────────────────────────────


def test_aggregate_ranks_by_power():
    group = pd.DataFrame(
        {
            "Hs_ft": [2.0, 3.0],
            "Tp_s": [8.0, 14.0],
            "Dir_deg": [270.0, 200.0],
            "wind_speed": [5.0, 5.0],
            "wind_direction": [180.0, 180.0],
            "wind_category": ["offshore", "offshore"],
            "tide_height": [2.0, 2.0],
        }
    )
    result = aggregate_hourly_partitions(group)
    # 3ft @ 14s has more power than 2ft @ 8s
    assert result["dominant_Hs_ft"] == 3.0
    assert result["dominant_Tp_s"] == 14.0


def test_aggregate_single_partition():
    group = pd.DataFrame(
        {
            "Hs_ft": [4.0],
            "Tp_s": [12.0],
            "Dir_deg": [260.0],
            "wind_speed": [3.0],
            "wind_direction": [350.0],
            "wind_category": ["offshore"],
            "tide_height": [1.5],
        }
    )
    result = aggregate_hourly_partitions(group)
    assert result["dominant_Hs_ft"] == 4.0
    assert pd.isna(result["secondary_Hs_ft"])


# ── Coverage gap: expand_timeseries_data with partitioned (swell_idx) data


def test_expand_partitioned_data():
    idx = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 03:00"] * 2)
    df = pd.DataFrame(
        {
            "swell_idx": [0, 0, 1, 1],
            "Hs_m": [1.0, 2.0, 0.5, 1.5],
            "Tp_s": [12.0, 14.0, 8.0, 10.0],
            "Dir_deg": [270.0, 275.0, 200.0, 210.0],
        },
        index=idx,
    )
    result = expand_timeseries_data(df, freq="h")
    # Each partition should be expanded from 2 to 4 rows
    assert len(result) == 8


# ── Coverage gap: expand with all-NaN direction column


def test_expand_skips_all_nan_direction():
    idx = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 02:00"])
    df = pd.DataFrame({"Dir_deg": [np.nan, np.nan], "val": [1.0, 2.0]}, index=idx)
    result = expand_timeseries_data(df, freq="h")
    assert result["Dir_deg"].isna().all()


# ── Coverage gap: calculate_surf_score with NaN wind


def test_surf_score_nan_wind(_surf_model_cfg):
    final, _, wind, _ = calculate_surf_score(
        7.0, np.nan, np.nan, np.nan, 1.0, _surf_model_cfg
    )
    assert pd.isna(wind)
    assert not pd.isna(final)


# ── Coverage gap: calculate_surf_score with NaN tide


def test_surf_score_nan_tide(_surf_model_cfg):
    final, _, _, tide = calculate_surf_score(
        7.0, 5.0, 8.0, "offshore", np.nan, _surf_model_cfg
    )
    assert pd.isna(tide)
    assert not pd.isna(final)
