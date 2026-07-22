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
    detect_swell_partitions,
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


# ── compute_swell_score / detect_swell_partitions ────────────────────────


@pytest.fixture
def _spectral_config():
    return {
        "spectral_scoring": {
            "hs_min_m": 0.3,
            "hs_center_m": 1.0,
            "hs_steepness": 6,
            "tp_min_s": 10.0,
            "tp_center_s": 13.0,
            "tp_steepness": 0.8,
            "spread_min_deg": 5.0,
            "spread_max_deg": 20.0,
            "w_hs": 0.40,
            "w_tp": 0.45,
            "w_sp": 0.15,
        },
        "partitioning": {
            "min_peak_prominence": 0.2,
            "min_peak_distance_hz": 0.02,
            "max_partitions_mop": 2,
        },
    }


# MA147's real swell band: 0.04-0.095 Hz in uniform 0.005 Hz bins, 12 bins total.
MOP_FREQS = np.arange(0.04, 0.0951, 0.005)


def _make_mop_dataset(n_times=3, peaks=None, energy_scale=1.0):
    """
    Build a minimal MOP-like xarray dataset with one or more Gaussian energy
    peaks placed in the real MA147 swell-band bin grid (12 bins, 0.005 Hz
    spacing). Each peak's own (a1, b1) is pinned exactly at its own bin so
    that overlapping Gaussian tails can't contaminate the direction/spread
    recovered at a *different* peak's frequency.

    peaks : list of dict, each with keys 'freq', 'width', 'amp', 'a1', 'b1'.
        Defaults to a single peak at 0.07 Hz.
    """
    if peaks is None:
        peaks = [{"freq": 0.07, "width": 0.006, "amp": 1.0, "a1": 0.9, "b1": 0.1}]

    freqs = MOP_FREQS
    n_freqs = len(freqs)
    times = pd.date_range("2025-01-01", periods=n_times, freq="3h")
    bw = np.full(n_freqs, 0.005)

    energy = np.zeros(n_freqs)
    for p in peaks:
        energy += (
            p["amp"]
            * energy_scale
            * np.exp(-0.5 * ((freqs - p["freq"]) / p["width"]) ** 2)
        )

    a1 = np.full(n_freqs, peaks[0]["a1"])
    b1 = np.full(n_freqs, peaks[0]["b1"])
    for p in peaks:
        idx = int(np.argmin(np.abs(freqs - p["freq"])))
        a1[idx] = p["a1"]
        b1[idx] = p["b1"]

    energy_2d = np.tile(energy, (n_times, 1))
    a1_2d = np.tile(a1, (n_times, 1))
    b1_2d = np.tile(b1, (n_times, 1))

    return xr.Dataset(
        {
            "waveEnergyDensity": (["waveTime", "waveFrequency"], energy_2d),
            "waveA1Value": (["waveTime", "waveFrequency"], a1_2d),
            "waveB1Value": (["waveTime", "waveFrequency"], b1_2d),
            "waveBandwidth": (["waveFrequency"], bw),
        },
        coords={
            "waveFrequency": freqs,
            "waveTime": times.values,
        },
    )


def test_swell_score_returns_expected_columns(_spectral_config):
    ds = _make_mop_dataset()
    result = compute_swell_score(ds, _spectral_config)
    expected_cols = {
        "hs_swell",
        "tp",
        "dp",
        "spread",
        "hs_secondary",
        "tp_secondary",
        "dp_secondary",
        "spread_secondary",
        "swell_score",
    }
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


# ── detect_swell_partitions ───────────────────────────────────────────────


def test_two_well_separated_peaks_give_independent_hs_tp_dp_spread():
    """Two well-separated peaks (different freq, different a1/b1) should
    produce two distinct partitions, each with its own correctly recovered
    Hs/Tp/Dp/spread -- not a blend of the two."""
    freq = MOP_FREQS
    bw = np.full(len(freq), 0.005)
    energy = np.zeros(len(freq))
    energy[1] = 4.0  # freq[1] = 0.045 Hz -- long period, more energy
    energy[9] = 2.0  # freq[9] = 0.085 Hz -- short period, less energy, 8 bins away
    a1 = np.full(len(freq), 0.9)
    b1 = np.full(len(freq), 0.1)
    # Distinctly different direction *and* magnitude (r1) for the second peak,
    # so spread as well as dp are expected to differ.
    a1[9], b1[9] = 0.3, 0.4

    partitions = detect_swell_partitions(
        freq,
        energy,
        a1,
        b1,
        bw,
        min_peak_prominence=0.2,
        min_peak_distance_hz=0.02,
        max_partitions=2,
    )

    assert len(partitions) == 2
    dominant, secondary = partitions[0], partitions[1]

    # Isolated spikes on an otherwise-zero background -> each partition's
    # window integral is exact regardless of boundary placement.
    assert dominant["hs"] == pytest.approx(4 * np.sqrt(4.0 * 0.005))
    assert secondary["hs"] == pytest.approx(4 * np.sqrt(2.0 * 0.005))
    assert dominant["tp"] == pytest.approx(1 / freq[1])
    assert secondary["tp"] == pytest.approx(1 / freq[9])
    assert dominant["dp"] == pytest.approx(np.degrees(np.arctan2(0.1, 0.9)) % 360)
    assert secondary["dp"] == pytest.approx(np.degrees(np.arctan2(0.4, 0.3)) % 360)
    assert dominant["spread"] != pytest.approx(secondary["spread"])


def test_single_peak_still_produces_one_partition():
    """A single-peak spectrum should still produce exactly one partition."""
    freq = MOP_FREQS
    bw = np.full(len(freq), 0.005)
    energy = np.zeros(len(freq))
    energy[5] = 3.0
    a1 = np.full(len(freq), 0.9)
    b1 = np.full(len(freq), 0.1)

    partitions = detect_swell_partitions(
        freq,
        energy,
        a1,
        b1,
        bw,
        min_peak_prominence=0.2,
        min_peak_distance_hz=0.02,
        max_partitions=2,
    )
    assert len(partitions) == 1
    assert partitions[0]["hs"] > 0
    assert not np.isnan(partitions[0]["tp"])


def test_flat_spectrum_yields_degenerate_zero_partition():
    """A flat/near-zero-energy spectrum (no real swell) should produce a
    single zero-height partition with NaN period/direction/spread, rather
    than crashing or fabricating a fake peak."""
    freq = MOP_FREQS
    bw = np.full(len(freq), 0.005)
    energy = np.zeros(len(freq))
    a1 = np.full(len(freq), 0.9)
    b1 = np.full(len(freq), 0.1)

    partitions = detect_swell_partitions(
        freq,
        energy,
        a1,
        b1,
        bw,
        min_peak_prominence=0.2,
        min_peak_distance_hz=0.02,
        max_partitions=2,
    )
    assert len(partitions) == 1
    assert partitions[0]["hs"] == 0.0
    assert np.isnan(partitions[0]["tp"])
    assert np.isnan(partitions[0]["dp"])
    assert np.isnan(partitions[0]["spread"])


def test_flat_spectrum_end_to_end_via_compute_swell_score(_spectral_config):
    """The degenerate case should also flow cleanly through compute_swell_score:
    no crash, hs_swell == 0, tp/dp/spread NaN, secondary NaN, worst-case score."""
    ds = _make_mop_dataset(
        peaks=[{"freq": 0.07, "width": 0.006, "amp": 0.0, "a1": 0.9, "b1": 0.1}]
    )
    result = compute_swell_score(ds, _spectral_config)
    row = result.iloc[0]
    assert row["hs_swell"] == 0.0
    assert pd.isna(row["tp"])
    assert pd.isna(row["dp"])
    assert pd.isna(row["spread"])
    assert pd.isna(row["hs_secondary"])
    assert row["swell_score"] == pytest.approx(1.0)


@pytest.mark.parametrize("edge_freq", [MOP_FREQS[0], MOP_FREQS[-1]])
def test_edge_of_band_peak_is_not_missed(edge_freq):
    """A real peak sitting at bin 0 or bin 11 of the 12-bin swell band has no
    room on one side to establish prominence -- confirm it isn't missed or
    mis-scored as a result."""
    freq = MOP_FREQS
    bw = np.full(len(freq), 0.005)
    # Gaussian centered exactly on the edge bin: only one-sided falloff exists
    # within the band, mirroring a genuine edge-of-band swell peak.
    energy = 2.0 * np.exp(-0.5 * ((freq - edge_freq) / 0.01) ** 2)
    a1 = np.full(len(freq), 0.9)
    b1 = np.full(len(freq), 0.1)

    partitions = detect_swell_partitions(
        freq,
        energy,
        a1,
        b1,
        bw,
        min_peak_prominence=0.2,
        min_peak_distance_hz=0.02,
        max_partitions=2,
    )

    assert len(partitions) == 1
    edge_idx = int(np.argmin(np.abs(freq - edge_freq)))
    assert partitions[0]["tp"] == pytest.approx(1 / freq[edge_idx])
    assert partitions[0]["hs"] > 0


# ── predict_bolinas_surf_height ──────────────────────────────────────────


@pytest.fixture
def _nearshore_cfg():
    return {
        "range_factor": 0.15,
        "range_period_min": 12,
        "range_step": 0.01,
        "dp_neutral": 215.0,
        "dp_slope": 0.04,
        "dp_max_boost": 2.00,
        "dp_min_factor": 0.80,
    }


def _predict_single(height_ft, period_s, dp_deg, nearshore_cfg):
    """predict_bolinas_surf_height with no secondary partition detected --
    the single-swell case. Mathematically identical to the dominant-only
    formula since a NaN/absent secondary contributes zero to the combined
    (RSS) height."""
    return predict_bolinas_surf_height(
        height_ft, period_s, dp_deg, np.nan, np.nan, np.nan, nearshore_cfg
    )


def test_surf_height_returns_min_max(_nearshore_cfg):
    result = _predict_single(4.0, 14.0, 215.0, _nearshore_cfg)
    assert "bolinas_surf_min_ft" in result
    assert "bolinas_surf_max_ft" in result
    assert result["bolinas_surf_max_ft"] >= result["bolinas_surf_min_ft"]


def test_surf_height_zero_for_flat(_nearshore_cfg):
    result = _predict_single(0.0, 10.0, 215.0, _nearshore_cfg)
    assert result["bolinas_surf_min_ft"] == 0.0
    assert result["bolinas_surf_max_ft"] == 0.0


def test_surf_height_nan_input(_nearshore_cfg):
    result = _predict_single(np.nan, 14.0, 215.0, _nearshore_cfg)
    assert result["bolinas_surf_min_ft"] == 0.0


def test_surf_height_longer_period_wider_range(_nearshore_cfg):
    short = _predict_single(4.0, 10.0, 215.0, _nearshore_cfg)
    long = _predict_single(4.0, 18.0, 215.0, _nearshore_cfg)
    short_range = short["bolinas_surf_max_ft"] - short["bolinas_surf_min_ft"]
    long_range = long["bolinas_surf_max_ft"] - long["bolinas_surf_min_ft"]
    assert long_range >= short_range


# ── predict_bolinas_surf_height: direction factor tests ──────────────────


def test_surf_height_south_swell_boosts(_nearshore_cfg):
    """More southerly dp (< neutral) should produce taller surf."""
    neutral = _predict_single(4.0, 14.0, 215.0, _nearshore_cfg)
    south = _predict_single(4.0, 14.0, 190.0, _nearshore_cfg)
    assert south["bolinas_surf_max_ft"] > neutral["bolinas_surf_max_ft"]
    assert south["bolinas_surf_min_ft"] >= neutral["bolinas_surf_min_ft"]


def test_surf_height_west_swell_reduces(_nearshore_cfg):
    """More westerly dp (> neutral) should produce shorter surf."""
    neutral = _predict_single(4.0, 14.0, 215.0, _nearshore_cfg)
    west = _predict_single(4.0, 14.0, 250.0, _nearshore_cfg)
    assert west["bolinas_surf_max_ft"] < neutral["bolinas_surf_max_ft"]


def test_surf_height_dp_factor_clamped_to_max_boost(_nearshore_cfg):
    """Extremely south dp should be clamped to dp_max_boost, not go infinite."""
    extreme_south = _predict_single(4.0, 14.0, 100.0, _nearshore_cfg)
    very_south = _predict_single(4.0, 14.0, 150.0, _nearshore_cfg)
    # With dp_max_boost=2.0 and dp_slope=0.04, dp=100 would give factor=5.6 unclamped
    # Both should be clamped to the same dp_max_boost
    assert extreme_south["bolinas_surf_max_ft"] == very_south["bolinas_surf_max_ft"]


def test_surf_height_dp_factor_clamped_to_min_factor(_nearshore_cfg):
    """Extremely west dp should be clamped to dp_min_factor, not go below."""
    extreme_west = _predict_single(4.0, 14.0, 350.0, _nearshore_cfg)
    very_west = _predict_single(4.0, 14.0, 300.0, _nearshore_cfg)
    # Both should be clamped to dp_min_factor
    assert extreme_west["bolinas_surf_max_ft"] == very_west["bolinas_surf_max_ft"]


def test_surf_height_dp_nan_input(_nearshore_cfg):
    """NaN dp should still return zero (handled by the NaN check)."""
    result = _predict_single(4.0, 14.0, np.nan, _nearshore_cfg)
    assert result["bolinas_surf_min_ft"] == 0.0
    assert result["bolinas_surf_max_ft"] == 0.0


# ── predict_bolinas_surf_height: dominant + secondary combination ────────


def test_dominant_and_secondary_each_apply_own_dp_multiplier(_nearshore_cfg):
    """Each partition's own direction multiplier must apply to its own height
    before combining -- not raw heights combined first with one multiplier
    based on the dominant direction only."""
    height_dom, period_dom, dp_dom = 4.0, 14.0, 190.0  # southerly -> boosted
    height_sec, period_sec, dp_sec = 3.0, 10.0, 300.0  # westerly -> reduced

    result = predict_bolinas_surf_height(
        height_dom, period_dom, dp_dom, height_sec, period_sec, dp_sec, _nearshore_cfg
    )

    def multiplier(dp):
        factor = 1.0 + _nearshore_cfg["dp_slope"] * (_nearshore_cfg["dp_neutral"] - dp)
        return np.clip(
            factor, _nearshore_cfg["dp_min_factor"], _nearshore_cfg["dp_max_boost"]
        )

    transformed_dom = height_dom * multiplier(dp_dom)
    transformed_sec = height_sec * multiplier(dp_sec)
    expected_combined = np.sqrt(transformed_dom**2 + transformed_sec**2)

    dynamic_rf = (
        _nearshore_cfg["range_factor"]
        + max(0, period_dom - _nearshore_cfg["range_period_min"])
        * _nearshore_cfg["range_step"]
    )
    to_half = lambda x: round(x * 2) / 2
    expected_min = to_half(max(0.5, expected_combined * (1 - dynamic_rf)))
    expected_max = to_half(max(expected_min, expected_combined * (1 + dynamic_rf)))

    assert result["bolinas_surf_min_ft"] == pytest.approx(expected_min)
    assert result["bolinas_surf_max_ft"] == pytest.approx(expected_max)


def test_worse_wrap_secondary_is_discounted_by_its_own_angle(_nearshore_cfg):
    """A synthetic case where the secondary partition has a worse wrap angle
    than the dominant shows the combined height discounted accordingly --
    the secondary's own multiplier applies to it, not the dominant's."""
    height_dom, period_dom, dp_dom = 4.0, 14.0, 215.0  # neutral dominant angle
    height_sec, period_sec = 4.0, 12.0
    dp_sec_good = 190.0  # southerly -> higher multiplier
    dp_sec_bad = 350.0  # far north/west -> clamped low

    result_good = predict_bolinas_surf_height(
        height_dom,
        period_dom,
        dp_dom,
        height_sec,
        period_sec,
        dp_sec_good,
        _nearshore_cfg,
    )
    result_bad = predict_bolinas_surf_height(
        height_dom,
        period_dom,
        dp_dom,
        height_sec,
        period_sec,
        dp_sec_bad,
        _nearshore_cfg,
    )

    assert result_bad["bolinas_surf_max_ft"] < result_good["bolinas_surf_max_ft"]

    # If the secondary had wrongly inherited the dominant's (neutral) multiplier
    # instead of its own (poor) one, the discount above wouldn't show up.
    dominant_multiplier = 1.0 + _nearshore_cfg["dp_slope"] * (
        _nearshore_cfg["dp_neutral"] - dp_dom
    )
    wrongly_combined = np.sqrt(
        (height_dom * dominant_multiplier) ** 2
        + (height_sec * dominant_multiplier) ** 2
    )
    correctly_combined = np.sqrt(
        (height_dom * dominant_multiplier) ** 2
        + (
            height_sec
            * np.clip(
                1.0
                + _nearshore_cfg["dp_slope"]
                * (_nearshore_cfg["dp_neutral"] - dp_sec_bad),
                _nearshore_cfg["dp_min_factor"],
                _nearshore_cfg["dp_max_boost"],
            )
        )
        ** 2
    )
    assert not np.isclose(wrongly_combined, correctly_combined)
    assert correctly_combined < wrongly_combined


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
