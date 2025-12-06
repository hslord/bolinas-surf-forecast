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

#@st.cache_data
#def load_data():
#    df = pd.read_pickle("forecast_df.pkl")
#
#    # Ensure index is DatetimeIndex
#    if not isinstance(df.index, pd.DatetimeIndex):
#        raise ValueError("forecast_df.pkl must have a DatetimeIndex as index.")
#
#    return df

#forecast_df = load_data()

@st.cache_data(show_spinner=True)
def load_forecast():

    raw = fetch_data_wrapper(config)
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

# Select only numeric columns for aggregation
numeric_cols = daylight_df.select_dtypes(include=["float", "int"]).columns

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

def alt_chart(df, y_col, y_title, color="steelblue"):
    """Reusable chart builder."""
    chart = alt.Chart(df.reset_index()).mark_line(color=color).encode(
        x=alt.X("index:T", title="Date/Time", axis=alt.Axis(format="%b %d")),
        y=alt.Y(f"{y_col}:Q", title=y_title)
    ).properties(height=200)
    return chart

st.subheader("📈 Time Series Plots")

tab1, tab2, tab3, tab4 = st.tabs([
    "Surf Score", "Wave Height", "Wind Speed", "Tide Height"
])

with tab1:
    st.altair_chart(alt_chart(filtered, "Surf Score (1-10)", "Surf Score"), use_container_width=True)

with tab2:
    st.altair_chart(alt_chart(filtered, "Surf Height Max (ft)", "Surf Height (ft)"), use_container_width=True)

with tab3:
    st.altair_chart(alt_chart(filtered, "Wind Speed (MPH)", "Wind Speed (mph)", color="green"), use_container_width=True)

with tab4:
    st.altair_chart(alt_chart(filtered, "Tide Height (ft)", "Tide Height (ft)", color="teal"), use_container_width=True)
