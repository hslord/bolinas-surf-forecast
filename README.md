# 🌊 Bolinas Surf Forecast

A fully automated, physics-aware surf forecasting pipeline for **Bolinas, CA**. This tool fetches deep-water wave data and NWS weather forecasts, applies a custom propagation model to account for local bathymetry, and updates a live dashboard every 12 hours.

## 🚀 Live Forecast
**View the interactive dashboard here: [Live App Link](https://bolinas-surf-forecast-helenlord.streamlit.app/)**

---

## 🛠 How It Works
Bolinas is a unique "wrap" break. Deep-water swells from the West and Northwest must refract around Duxbury Reef to reach the Patch and the Channel. 

This pipeline:
1. **Fetches** raw WW3 (Wave Watch III) data and NWS wind grids.
2. **Filters** swells based on "Sweet Spot" angles (S/SW) vs "Wrap" angles (W/NW).
3. **Calculates** a custom **Surf Score (1-10)** by weighting tide height, wind velocity/direction, and swell period.
4. **Automates** the entire process via GitHub Actions—no manual local runs required.



---

## ✨ Features
- **Local Calibration:** Configurable "shadowing" and "wrap" coefficients in `surf_config.yaml`.
- **Tide Integration:** Real-time NOAA MLLW tide predictions.
- **Daylight Awareness:** Specifically highlights "Best Session" windows during daylight hours.
- **Wind Alignment:** Calibrated for Bolinas' specific coastal orientation to identify true offshore flow.

---

## 📦 Installation & Local Setup

If you want to run the model locally:

1. **Clone the Repo:**
```bash
   git clone [https://github.com/hslord/bolinas-surf-forecast.git](https://github.com/hslord/bolinas-surf-forecast.git)
   cd bolinas-surf-forecast
```

2. **Run the Forecast:**
```bash
   python src/run_surf_app.py
```

3. **Launch the Streamlit UI**
```bash
streamlit run src/surf_app_streamlit.py

http://localhost:8501
```

## Repository Structure

```
bolinas-surf-forecast/
├── config/
│   └── surf_config.yaml
│
├── src/
│   ├── fetch_data.py
│   ├── process_data.py
│   ├── reference_functions.py
│   ├── surf_app_streamlit.py
│   ├── run_surf_app_local.py
│   └── __init__.py
│
├── requirements-pipeline.txt
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## License

MIT License — See the `LICENSE` file for details.
