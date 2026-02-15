"""
Config sensitivity tests using CDIP MOP MA147 hindcast data.

These tests load real historical wave data via OPeNDAP and verify that
config parameter changes produce expected directional effects on scores.

"""

import sys
import copy
from pathlib import Path
import pytest
import numpy as np
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from process_data import (
    compute_swell_score,
    predict_bolinas_surf_height,
    calculate_surf_score,
)

HINDCAST_URL = (
    "https://thredds.cdip.ucsd.edu/thredds/dodsC/"
    "cdip/model/MOP_alongshore/MA147_hindcast.nc"
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "surf_config.yaml"


@pytest.fixture(scope="module")
def _hindcast_swell():
    """Load a 48-hour slice of MA147 hindcast, filtered to swell band."""
    ds = xr.open_dataset(HINDCAST_URL, engine="netcdf4")
    ds = ds.sel(waveFrequency=ds.waveFrequency[ds.waveFrequency < 0.1])
    # Take the last 48 timesteps to keep it fast
    ds = ds.isel(waveTime=slice(-48, None))
    return ds.load()


@pytest.fixture(scope="module")
def _baseline_config():
    """Load the current production config."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _modify_config(config, path, value):
    """Return a deep copy of config with a nested key changed.

    path is a dot-separated string, e.g. 'surf_model.spectral_scoring.hs_min_m'
    """
    cfg = copy.deepcopy(config)
    keys = path.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value
    return cfg


# -- Swell score sensitivity ----------------------------------------------


class TestSwellScoreSensitivity:

    def test_higher_hs_min_lowers_scores(self, _hindcast_swell, _baseline_config):
        """Raising the minimum height threshold should lower average scores."""
        baseline = compute_swell_score(_hindcast_swell, _baseline_config["surf_model"])
        stricter = _modify_config(
            _baseline_config, "surf_model.spectral_scoring.hs_min_m", 0.8
        )
        modified = compute_swell_score(_hindcast_swell, stricter["surf_model"])
        assert modified["swell_score"].mean() <= baseline["swell_score"].mean()

    def test_higher_tp_weight_favors_long_period(
        self, _hindcast_swell, _baseline_config
    ):
        """Increasing period weight should raise scores when long-period swell is present."""
        cfg_tp = _modify_config(
            _baseline_config, "surf_model.spectral_scoring.w_tp", 0.70
        )
        cfg_tp = _modify_config(cfg_tp, "surf_model.spectral_scoring.w_hs", 0.10)
        cfg_hs = _modify_config(
            _baseline_config, "surf_model.spectral_scoring.w_tp", 0.10
        )
        cfg_hs = _modify_config(cfg_hs, "surf_model.spectral_scoring.w_hs", 0.70)

        scores_tp = compute_swell_score(_hindcast_swell, cfg_tp["surf_model"])
        scores_hs = compute_swell_score(_hindcast_swell, cfg_hs["surf_model"])

        # With real data containing mixed periods, these should differ
        assert not np.isclose(
            scores_tp["swell_score"].mean(), scores_hs["swell_score"].mean(), atol=0.01
        )

    def test_wider_spread_max_is_more_lenient(self, _hindcast_swell, _baseline_config):
        """A wider spread tolerance should produce equal or higher scores."""
        baseline = compute_swell_score(_hindcast_swell, _baseline_config["surf_model"])
        lenient = _modify_config(
            _baseline_config, "surf_model.spectral_scoring.spread_max_deg", 40.0
        )
        modified = compute_swell_score(_hindcast_swell, lenient["surf_model"])
        assert modified["swell_score"].mean() >= baseline["swell_score"].mean()

    def test_lower_hs_full_credit_raises_scores(
        self, _hindcast_swell, _baseline_config
    ):
        """Lowering the full-credit height threshold should raise average scores."""
        baseline = compute_swell_score(_hindcast_swell, _baseline_config["surf_model"])
        easier = _modify_config(
            _baseline_config, "surf_model.spectral_scoring.hs_full_credit_m", 0.8
        )
        modified = compute_swell_score(_hindcast_swell, easier["surf_model"])
        assert modified["swell_score"].mean() >= baseline["swell_score"].mean()


# -- Wind score sensitivity ------------------------------------------------


class TestWindScoreSensitivity:

    @pytest.mark.parametrize(
        "speed_range,expected_direction",
        [
            (6.0, "stricter"),
            (24.0, "more_lenient"),
        ],
    )
    def test_speed_range_controls_penalty(
        self, _baseline_config, speed_range, expected_direction
    ):
        """A smaller speed_range means wind penalizes faster; larger is more lenient."""
        cfg = _modify_config(
            _baseline_config, "surf_model.wind.speed_range", speed_range
        )
        _, _, wind_score, _ = calculate_surf_score(
            7.0, 10.0, 15.0, "onshore", 1.0, cfg["surf_model"]
        )
        _, _, baseline_wind, _ = calculate_surf_score(
            7.0, 10.0, 15.0, "onshore", 1.0, _baseline_config["surf_model"]
        )
        if expected_direction == "stricter":
            assert wind_score <= baseline_wind
        else:
            assert wind_score >= baseline_wind

    def test_higher_speed_floor_is_more_lenient(self, _baseline_config):
        """Raising the speed floor means more wind is tolerated before penalty kicks in."""
        cfg = _modify_config(_baseline_config, "surf_model.wind.speed_floor", 10.0)
        _, _, lenient_wind, _ = calculate_surf_score(
            7.0, 8.0, 12.0, "onshore", 1.0, cfg["surf_model"]
        )
        _, _, baseline_wind, _ = calculate_surf_score(
            7.0, 8.0, 12.0, "onshore", 1.0, _baseline_config["surf_model"]
        )
        assert lenient_wind >= baseline_wind


# -- Tide score sensitivity ------------------------------------------------


class TestTideScoreSensitivity:

    @pytest.mark.parametrize(
        "optimal,tide_height",
        [
            (0.0, 0.0),
            (3.0, 3.0),
            (5.0, 5.0),
        ],
    )
    def test_score_peaks_at_optimal(self, _baseline_config, optimal, tide_height):
        """Tide score should be 10.0 when tide height equals optimal."""
        cfg = _modify_config(
            _baseline_config, "surf_model.tide.optimal_height", optimal
        )
        _, _, _, tide_score = calculate_surf_score(
            7.0, 3.0, 5.0, "offshore", tide_height, cfg["surf_model"]
        )
        assert tide_score == 10.0

    def test_wider_sigma_is_more_tolerant(self, _baseline_config):
        """A larger sigma means the Gaussian is wider, so off-optimal tides score higher."""
        cfg_narrow = _modify_config(_baseline_config, "surf_model.tide.sigma", 1.0)
        cfg_wide = _modify_config(_baseline_config, "surf_model.tide.sigma", 5.0)
        _, _, _, narrow_score = calculate_surf_score(
            7.0, 3.0, 5.0, "offshore", 5.0, cfg_narrow["surf_model"]
        )
        _, _, _, wide_score = calculate_surf_score(
            7.0, 3.0, 5.0, "offshore", 5.0, cfg_wide["surf_model"]
        )
        assert wide_score > narrow_score


# -- Surf height sensitivity -----------------------------------------------


class TestSurfHeightSensitivity:

    @pytest.mark.parametrize("range_factor", [0.05, 0.15, 0.30])
    def test_range_factor_scales_spread(self, _baseline_config, range_factor):
        """Larger range_factor should produce a wider min-max spread."""
        cfg = _modify_config(
            _baseline_config, "surf_model.nearshore.range_factor", range_factor
        )
        result = predict_bolinas_surf_height(
            4.0, 14.0, 215.0, cfg["surf_model"]["nearshore"]
        )
        spread = result["bolinas_surf_max_ft"] - result["bolinas_surf_min_ft"]
        assert spread >= 0
        if range_factor > 0.15:
            baseline = predict_bolinas_surf_height(
                4.0, 14.0, 215.0, _baseline_config["surf_model"]["nearshore"]
            )
            baseline_spread = (
                baseline["bolinas_surf_max_ft"] - baseline["bolinas_surf_min_ft"]
            )
            assert spread >= baseline_spread
