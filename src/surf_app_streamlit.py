import streamlit as st
import pandas as pd
import altair as alt
from fetch_data import fetch_data_wrapper
from process_data import process_data_wrapper
import yaml

with open("../config/surf_config.yaml", "r") as f:
    config = yaml.safe_load(f)

st.set_page_config(page_title="Bolinas Surf Forecast", layout="wide")

# -------------------------
# LOAD DATA
# -------------------------

@st.cache_data(show_spinner=True)
def load_forecast():

    raw = fetch_data_wrapper(config['data_sources'])
    df = process_data_wrapper(raw, config)
    return df

forecast_df = load_forecast()

st.title("🌊 Bolinas Surf Forecast")
st.caption("Interactive explorer for surf, swell, wind, and tide data")

# -------------------------
# SIDEBAR FILTERS
# -------------------------

st.sidebar.header("Filters")

# Date filter (unique dates in index)
available_dates = (
    pd.Series(forecast_df.index.date)
    .drop_duplicates()
    .sort_values()
    .tolist()
)

selected_dates = st.sidebar.multiselect(
    "Select dates",
    options=available_dates,
    default=[]
)

# Daylight filter
daylight_only = st.sidebar.checkbox("Daylight hours only", value=True)

# -------------------------
# APPLY FILTERS
# -------------------------

filtered = forecast_df.copy()

if selected_dates:
    # Convert index to a Series so .isin works
    idx_dates = pd.Series(filtered.index.date, index=filtered.index)
    filtered = filtered[idx_dates.isin(selected_dates)]

if daylight_only:
    filtered = filtered[filtered["is_daylight"] == True]

# -------------------------
# MAIN DISPLAY
# -------------------------

st.subheader("Forecast Data")
st.dataframe(filtered, height=400)

# -------------------------
# TOP TIMES AND DAILY AVERAGES
# -------------------------

st.subheader("🏄 Top 10 Surf Times (Daylight Only)")

# Filter for daylight
daylight_df = forecast_df[forecast_df["is_daylight"]].copy()\
			.reset_index().rename(columns={"index": "datetime"})

# Top 10 by surf score
top10 = daylight_df.nlargest(10, "Surf Score (1-10)")

st.dataframe(top10, use_container_width=True)


# ----------------------------------------------------
# DAILY AVERAGES
# ----------------------------------------------------
st.subheader("📊 Daily Average Surf Metrics")

# Ensure datetime column exists (we already reset_index earlier)
daylight_df["day"] = daylight_df["datetime"].dt.date

# Select only key numeric columns for aggregation
numeric_cols = ['Surf Score (1-10)',  'Surf Height Min (ft)',
 'Surf Height Max (ft)', 'Swell Size (ft)', 'Swell Period (Seconds)', 'Swell Direction (Degrees)']

# Compute daily averages
daily_avg_df = (
    daylight_df
    .groupby("day")[numeric_cols]
    .mean()
    .round(2)
    .reset_index()
)

st.dataframe(daily_avg_df, use_container_width=True)

# ==========================
# TIME SERIES PLOTS (Altair)
# ==========================

def alt_chart(df, y_col, y_title, domain=None, color="steelblue"):
    """Reusable chart builder with optional y-axis domain."""
    y_encoding = alt.Y(
        f"{y_col}:Q",
        title=y_title,
        scale=alt.Scale(domain=domain) if domain else alt.Undefined  # <-- key line
    )

    chart = (
        alt.Chart(df.reset_index())
        .mark_line(color=color)
        .encode(
            x=alt.X("index:T", title="Date/Time", axis=alt.Axis(format="%b %d")),
            y=y_encoding,
        )
        .properties(height=200)
    )
    return chart

st.subheader("📈 Time Series Plots")

tab1, tab2, tab3, tab4 = st.tabs([
    "Surf Score", "Wave Height", "Wind Speed", "Tide Height"
])

with tab1:
    st.altair_chart(
        alt_chart(filtered, "Surf Score (1-10)", "Surf Score", domain=[0, 10]),
        use_container_width=True
    )

with tab2:
    st.altair_chart(
        alt_chart(filtered, "Surf Height Max (ft)", "Surf Height (ft)", domain=[0, 6]),
        use_container_width=True
    )

with tab3:
    st.altair_chart(
        alt_chart(filtered, "Wind Speed (MPH)", "Wind Speed (mph)", domain=[0, 30], color="green"),
        use_container_width=True
    )

with tab4:
    st.altair_chart(
        alt_chart(filtered, "Tide Height (ft)", "Tide Height (ft)", domain=[-2, 7], color="teal"),
        use_container_width=True
    )