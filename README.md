# Bolinas Surf Forecast

A fully automated surf forecasting pipeline for **Bolinas, CA**. Uses nearshore spectral wave predictions from the CDIP MOP system, combined with offshore swell context, tidal data, and wind forecasts to produce a scored surf forecast updated four times daily.

## Live Forecast
**[View the interactive dashboard](https://bolinas-surf-forecast-helenlord.streamlit.app/)**

---

## How It Works

Bolinas is a unique wrap break - deep-water swells from the West and Northwest must refract around Duxbury Reef to reach the Patch and the Channel. Accurately predicting what arrives at the beach requires modeling how offshore wave energy transforms as it crosses the continental shelf and bends around local bathymetry.

### CDIP MOP

This pipeline uses **CDIP MOP (MOnitoring and Prediction)** data from the Coastal Data Information Program at Scripps Institution of Oceanography, UC San Diego. MOP is a linear spectral refraction wave model that transforms deep-water wave energy to thousands of nearshore output points along the California coast using precomputed transformation coefficients derived from backward ray tracing over high-resolution bathymetry (O'Reilly & Guza, 1991; O'Reilly et al., 2016).

The model accounts for refraction, shoaling, and island/headland sheltering. The forecast variant used here is driven by **ECMWF HRES-WAM** global wave model output as its offshore boundary condition.

MOP station **MA147** sits at **15-meter water depth** near the Patch - meaning the refraction, shoaling, and shadowing from Duxbury Reef are already encoded in the spectral output. The pipeline reads the full directional spectrum (energy density, Fourier coefficients a1/b1, bandwidth) from MA147 and computes swell quality metrics directly from the nearshore-transformed spectra.

### Pipeline Steps

1. **Fetch** CDIP MOP spectral forecasts for MA147 (nearshore) and WW3 partitioned swell data (offshore context) via OPeNDAP.
2. **Score** swell quality (1-10) from the MOP spectrum - evaluating significant height (Hs), peak period (Tp), and directional spread at the swell frequency band.
3. **Predict** surfable wave-height range (ft) from MOP Hs with period-adjusted variability.
4. **Classify** winds as offshore / cross-shore / onshore relative to Bolinas' coastline orientation and apply parameterized wind and tide penalties.
5. **Rank** WW3 offshore partitions by wave power to provide dominant/secondary swell context for the dashboard.
6. **Combine** swell, wind, and tide scores into a final Surf Score (1-10) for each hourly forecast window.
7. **Automate** via GitHub Actions (4x daily) - no manual runs required.

---

## Features
- **CDIP MOP Nearshore Model:** Wave heights derived from Scripps' spectral refraction model at 15m depth. Supports ECMWF and NCEP forcing.
- **Spectral Swell Scoring:** Quality score from energy density, peak period, and directional spread - prioritizes clean, long-period groundswells over disorganized short-period sea.
- **Tide Integration:** NOAA CO-OPS predictions scored with a Gaussian curve around an optimal tide height.
- **Wind Alignment:** Classified relative to Bolinas' coastal orientation (165 SSE) to identify true offshore flow.
- **Daylight Awareness:** Best-session recommendations limited to daylight hours.
- **User Feedback:** In-app form to report observed conditions and help calibrate the model.

---

## Installation & Local Setup

1. **Clone the repo:**
```bash
git clone https://github.com/hslord/bolinas-surf-forecast.git
cd bolinas-surf-forecast
```

2. **Install dependencies:**
```bash
make install
```

3. **Run the forecast pipeline:**
```bash
make run
```

4. **Launch the Streamlit dashboard:**
```bash
make ui
```

### Makefile Commands

| Command          | Description                                              |
|------------------|----------------------------------------------------------|
| `make install`   | Install/upgrade Python dependencies                      |
| `make run`       | Run the data fetch + processing pipeline                 |
| `make ui`        | Launch the Streamlit dashboard                           |
| `make test`      | Run all tests (unit + config sensitivity)                            |
| `make coverage`  | Run unit tests with coverage report                      |
| `make lint`      | Run Pylint on `src/` and `test/`                         |
| `make format`    | Auto-format code with Black                              |
| `make clean`     | Remove `__pycache__`, `.pytest_cache`, and temp files    |

---

## Testing

Unit tests and config sensitivity tests live in `test/` and cover data fetching, processing logic, scoring functions, UI helpers, and config parameter behaviors.

```bash
make test          # all 56 tests (~3s)
make coverage      # tests with line-by-line coverage report
```

Config sensitivity tests (`test/test_config_sensitivity.py`) load real MA147 hindcast data via OPeNDAP and verify that config parameter changes produce expected directional effects on scores (e.g. raising `hs_min_m` lowers average swell scores, wider `sigma` makes tide scoring more tolerant).

---

## Config Tuning

The `simulations/config_tuning.ipynb` notebook provides an interactive environment for experimenting with config parameter values against real hindcast data.

It loads the MA147 hindcast, auto-selects representative days spanning a range of conditions (biggest/smallest swell, longest/shortest period, cleanest/messiest spread, plus median days), and produces side-by-side comparison tables showing baseline vs experiment scores. Separate sections cover swell scoring, wind penalties, tide penalties, and combined effects.

---

## Repository Structure

```
bolinas-surf-forecast/
├── .github/
│   └── workflows/
│       └── update_forecast.yml       # GitHub Actions: runs pipeline 4x daily
├── config/
│   └── surf_config.yaml              # All model parameters and data source config
├── data/
│   ├── forecast_df.parquet           # Latest forecast output (auto-committed by CI)
│   └── user_feedback.csv             # Crowd-sourced observed condition reports
├── simulations/
│   └── config_tuning.ipynb           # Interactive config tuning notebook
├── src/
│   ├── fetch_data.py                 # Data fetchers: WW3, CDIP MOP, tides, wind, sun
│   ├── process_data.py               # Spectral scoring, surf height, wind/tide penalties
│   ├── reference_functions.py        # Shared utilities (status logging, WW3 grid lookup)
│   ├── run_surf_app.py               # Pipeline entry point: fetch -> process -> save
│   ├── surf_app_streamlit.py         # Streamlit dashboard UI
│   └── __init__.py
├── test/
│   ├── test_fetch_data.py            # Unit tests for data fetching
│   ├── test_process_data.py          # Unit tests for scoring and processing logic
│   ├── test_config_sensitivity.py    # Config sensitivity tests (uses CDIP hindcast)
│   ├── test_surf_app_streamlit.py    # Unit tests for UI helper functions
│   └── test_run_surf_app.py          # Notes on pipeline entry point coverage
├── Makefile
├── requirements.txt                  # Streamlit app dependencies
├── requirements-pipeline.txt         # Full pipeline dependencies (used by CI)
├── LICENSE
└── README.md
```

---

## Data Sources

| Source | What | Endpoint |
|--------|------|----------|
| **CDIP MOP** | Nearshore spectral wave forecast (MA147, 15m depth) | UCSD THREDDS OPeNDAP |
| **WW3** | Offshore partitioned swell (deep water context) | UCAR THREDDS OPeNDAP |
| **NOAA CO-OPS** | Tide predictions (SF Golden Gate, stn 9414290) | noaa-coops API |
| **Open-Meteo** | Hourly wind speed, gusts, direction | Open-Meteo REST API |
| **Sunrise-Sunset** | Civil twilight / daylight windows | sunrise-sunset.org |

---

## Configuration

All tunable parameters live in `config/surf_config.yaml`:

- **Spectral scoring weights** - height, period, and spread contributions to the swell quality score
- **Nearshore range** - variability factor for translating MOP Hs to a surfable wave-height range
- **Wind scoring** - offshore/onshore angle cutoffs, gust weighting, penalty curve
- **Tide scoring** - optimal height, Gaussian sigma, penalty floor
- **Final scoring** - wind/tide impact weights, secondary penalty blending

---

## References

- O'Reilly, W.C. et al. (2016). "The California coastal wave monitoring and prediction system." *Coastal Engineering*, 116, 118-132.
- O'Reilly, W.C. & Guza, R.T. (1991, 1993). Spectral refraction of surface gravity waves.
- [CDIP MOP Documentation](https://cdip.ucsd.edu/MOP_v1.1/)

## License

MIT License -- see the `LICENSE` file for details.
