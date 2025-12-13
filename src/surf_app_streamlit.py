import streamlit as st
import pandas as pd
import altair as alt

# ------------------------------------
# DATA LOADING (pkl)
# ------------------------------------
@st.cache_data(show_spinner=True)
def load_forecast():
    try:
        df = pd.read_pickle('data/forecast_df.pkl')
        return df
    except Exception as e:
        st.sidebar.error(f"Failed loading forecast_df.pkl")
        st.stop()

forecast_df = load_forecast()
# Ensure datetime index (important for .pkl loads)
if not isinstance(forecast_df.index, pd.DatetimeIndex):
    forecast_df.index = pd.to_datetime(forecast_df.index)

st.title("🌊 Bolinas Surf Forecast")
st.caption("Your personalized surf, swell, wind, and tide dashboard.")

# =======================================================================================
# DAY/NIGHT SHADING UTILITIES
# =======================================================================================

def build_night_rects(df):
    """
    Build Altair rectangle blocks for nighttime using is_daylight == False.
    Daytime remains unshaded for clarity.
    """
    if "is_daylight" not in df.columns:
        return pd.DataFrame(columns=["start", "end"])

    df = df.reset_index().rename(columns={"index": "datetime"})
    rects = []
    in_block = False
    start_time = None

    for _, row in df.iterrows():
        if not row["is_daylight"] and not in_block:
            in_block = True
            start_time = row["datetime"]

        if row["is_daylight"] and in_block:
            in_block = False
            rects.append({"start": start_time, "end": row["datetime"]})

    # If ending at night
    if in_block:
        rects.append({"start": start_time, "end": df["datetime"].iloc[-1]})

    return pd.DataFrame(rects)


def add_daylight_shading(line_chart, df):
    """Overlay darker shading for nighttime blocks."""
    rect_df = build_night_rects(df)

    if rect_df.empty:
        return line_chart

    rect = (
        alt.Chart(rect_df)
        .mark_rect(opacity=0.35, color="#d0d0d0")
        .encode(
            x="start:T",
            x2="end:T",
        )
    )

    return rect + line_chart


def alt_chart(df, y_col, y_title, domain=None, color="steelblue"):
    """Unified line chart with visible night shading."""
    base = df.reset_index()

    y_enc = alt.Y(
        f"{y_col}:Q",
        title=y_title,
        scale=alt.Scale(domain=domain) if domain else alt.Undefined
    )

    line = (
        alt.Chart(base)
        .mark_line(color=color)
        .encode(
            x=alt.X("index:T", title="Date/Time"),
            y=y_enc,
            tooltip=[
                alt.Tooltip("index:T", title="Time"),
                alt.Tooltip(f"{y_col}:Q", title=y_title),
            ],
        )
        .properties(height=220)
    )

    return add_daylight_shading(line, base)

# =======================================================================================
# SIDEBAR FILTERS
# =======================================================================================

st.sidebar.header("Filters")

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

daylight_only = st.sidebar.checkbox("Daylight hours only (tables only)", value=True)

filtered = forecast_df.copy()
idx_dates = pd.Series(filtered.index.date, index=filtered.index)

if selected_dates:
    filtered = filtered[idx_dates.isin(selected_dates)]

# ❗ Only apply daylight filter to tables — NOT to time series
filtered_for_tables = filtered.copy()
if daylight_only:
    filtered_for_tables = filtered_for_tables[filtered_for_tables["is_daylight"]]

# =======================================================================================
# SUMMARY CARDS
# =======================================================================================

st.subheader("🏄 Quick Surf Summary")

col1, col2, col3 = st.columns(3)

current = forecast_df.iloc[0]
best_row = forecast_df.loc[forecast_df["Surf Score (1-10)"].idxmax()]

col1.metric(
    "Current Surf",
    f"{current['Surf Height Min (ft)']}–{current['Surf Height Max (ft)']} ft",
)

col2.metric(
    "Current Score",
    f"{current['Surf Score (1-10)']}/10",
)

col3.metric(
    "Best Upcoming Window",
    f"{best_row['Surf Height Min (ft)']}–{best_row['Surf Height Max (ft)']} ft",
    help=f"{best_row.name:%b %d, %I:%M %p}",
)

# =======================================================================================
# SURF QUALITY BREAKDOWN PANEL
# =======================================================================================

st.subheader("🔎 Surf Quality Breakdown")

# Build timestamp list for selectbox
timestamps = list(forecast_df.index)

# Safe default: highest surf score timestamp
best_timestamp = forecast_df["Surf Score (1-10)"].idxmax()
default_idx = timestamps.index(best_timestamp)

selected_time = st.selectbox(
    "Select a forecast time to inspect:",
    options=timestamps,
    index=default_idx,
    format_func=lambda t: t.strftime("%b %d, %I:%M %p"),
)

row = forecast_df.loc[selected_time]

# ---- Horizontal Layout ----
col1, col2, col3, col4 = st.columns(4)

# Helper for small secondary text
def small(text):
    return f"<span style='font-size:0.95rem; color:#666;'>{text}</span>"

# Column 1 – Surf + Propagation
col1.metric("Surf Score", f"{row['Surf Score (1-10)']}/10")
col1.markdown(
    small(
        f"Propagation: {row['Propagation Score (0-1)']:.2f} — from "
        f"{round(float(row['Swell Direction (Degrees)']),1)}°"
    ),
    unsafe_allow_html=True
)

# Column 2 – Swell
col2.metric("Swell Score", f"{row['Swell Score (1-10)']}/10")
col2.markdown(
    small(
        f"{round(float(row['Swell Size (ft)']), 1)} ft @ "
        f"{round(float(row['Swell Period (Seconds)']), 1)}s"
    ),
    unsafe_allow_html=True
)

# Column 3 – Wind
col3.metric("Wind Score", f"{row['Wind Score (1-10)']}/10")
col3.markdown(
    small(f"{row['Wind Speed (MPH)']} mph {row['Wind Direction (Degrees)']}°"),
    unsafe_allow_html=True
)

# Column 4 – Tide
col4.metric("Tide Score", f"{row['Tide Score (1-10)']}/10")
col4.markdown(
    small(f"{row['Tide Height (ft)']} ft"),
    unsafe_allow_html=True
)


# =======================================================================================
# TOP 10 SURF WINDOWS
# =======================================================================================

st.subheader("🏆 Top Surf Windows (Daylight Only)")

daylight_df = (
    forecast_df[forecast_df["is_daylight"]]
    .reset_index()
    .rename(columns={"index": "datetime"})
)

top10 = daylight_df.nlargest(10, "Surf Score (1-10)")
st.dataframe(top10, use_container_width=True, height=300)

# =======================================================================================
# DAILY AVERAGES
# =======================================================================================

st.subheader("📊 Daily Surf Averages")

daylight_df["day"] = daylight_df["datetime"].dt.date

numeric_cols = [
    "Surf Score (1-10)", 
    "Surf Height Min (ft)",
    "Surf Height Max (ft)",
    "Swell Size (ft)",
    "Swell Period (Seconds)",
    "Swell Direction (Degrees)"
]

daily_avg_df = (
    daylight_df.groupby("day")[numeric_cols]
    .mean()
    .round(2)
    .reset_index()
)

st.dataframe(daily_avg_df, use_container_width=True)

# =======================================================================================
# TIME SERIES EXPLORER (NO DAYLIGHT FILTER)
# =======================================================================================

st.subheader("📈 Time Series Explorer")

# Important: ALWAYS use unfiltered forecast_df for shading accuracy
unfiltered = forecast_df.copy()

tab1, tab2, tab3, tab4 = st.tabs([
    "Surf Score", "Wave Height", "Wind Speed", "Tide Height"
])

with tab1:
    st.altair_chart(
        alt_chart(unfiltered, "Surf Score (1-10)", "Surf Score", domain=[0, 10]),
        use_container_width=True
    )

with tab2:
    st.altair_chart(
        alt_chart(unfiltered, "Surf Height Max (ft)", "Surf Height (ft)", domain=[0, 6]),
        use_container_width=True
    )

with tab3:
    st.altair_chart(
        alt_chart(unfiltered, "Wind Speed (MPH)", "Wind Speed (mph)", domain=[0, 30], color="green"),
        use_container_width=True
    )

with tab4:
    st.altair_chart(
        alt_chart(unfiltered, "Tide Height (ft)", "Tide Height (ft)", domain=[-2, 7], color="teal"),
        use_container_width=True
    )

# =======================================================================================
# RAW TABLE (WITH OPTIONAL DAYLIGHT FILTER)
# =======================================================================================

st.subheader("📄 Full Forecast Data")
st.dataframe(filtered_for_tables, use_container_width=True, height=350)
