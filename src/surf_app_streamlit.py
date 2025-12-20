import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

# ------------------------------------
# DATA LOADING (pkl)
# ------------------------------------
@st.cache_data(show_spinner=True)
def load_forecast():
    base_dir = Path(__file__).resolve().parents[1]  # repo root
    pkl_path = base_dir / "data" / "forecast_df.pkl"

    if not pkl_path.exists():
        st.error(f"PKL file not found at: {pkl_path}")
        st.stop()

    return pd.read_pickle(pkl_path)

forecast_df = load_forecast()
# Ensure datetime index (important for .pkl loads)
if not isinstance(forecast_df.index, pd.DatetimeIndex):
    forecast_df.index = pd.to_datetime(forecast_df.index)

st.title("🌊 Bolinas Surf Forecast")
st.caption("Your personalized surf, swell, wind, and tide dashboard.")


# =======================================================================================
# SUMMARY CARDS
# =======================================================================================

st.subheader("🏄 Quick Surf Summary")

col1, col2, col3 = st.columns(3)

current = forecast_df.iloc[0]
daylight_df = (
    forecast_df[forecast_df["is_daylight"]]
    .reset_index(names="datetime")
)
best_row = daylight_df.loc[
    daylight_df["Surf Score (1-10)"].idxmax()
]

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
    f"{best_row['Surf Score (1-10)']}/ {best_row['Surf Height Min (ft)']}–{best_row['Surf Height Max (ft)']} ft",
    help=best_row["datetime"].strftime("%b %d, %I:%M %p"),
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

# Helper for small secondary text
def small(text, size):
    return f"<span style='font-size:{size}rem; color:#666;'>{text}</span>"

# ---- Horizontal Layout ----
col1, col2, col3, col4 = st.columns(4)

# Column 1 – Surf + Propagation
col1.metric("Surf Score", f"{row['Surf Score (1-10)']}/10")
col1.markdown(
    small(
        f"Propagation {row['Dominant Propagation Score (0-1)']:.2f} from "
        f"{row['Dominant Swell Direction']:.1f}°",
        0.90
    ),
    unsafe_allow_html=True
)

# Column 2 – Swell
col2.metric("Swell Score", f"{row['Dominant Swell Score (1-10)']}/10")
col2.markdown(
    small(
        f"{round(float(row['Dominant Swell Size (ft)']), 1)} ft @ "
        f"{round(float(row['Dominant Swell Period']), 1)}s",
        0.95
    ),
    unsafe_allow_html=True
)

# Column 3 – Wind
col3.metric("Wind Score", f"{row['Wind Score (1-10)']}/10")
col3.markdown(
    small(f"{int(row['Wind Speed (MPH)'])} mph w gusts to {int(row['Wind Gust (MPH)'])} {row['Wind Direction']}",
          0.9
          ),
    unsafe_allow_html=True
)

# Column 4 – Tide
col4.metric("Tide Score", f"{row['Tide Score (1-10)']}/10")
col4.markdown(
    small(f"{row['Tide Height (ft)']} ft",
          0.95
          ),
    unsafe_allow_html=True
)


# =======================================================================================
# TOP 10 SURF WINDOWS
# =======================================================================================

st.subheader("🏆 Top Surf Windows (Daylight Only)")

#round key columns for visualization
def format_for_display(df: pd.DataFrame):
    df = df.copy()
    df["Wind Speed (MPH)"] = df["Wind Speed (MPH)"].round(0)
    df["Wind Gust (MPH)"] = df["Wind Gust (MPH)"].round(0)
    df["Tide Height (ft)"] = df["Tide Height (ft)"].round(1)
    df["Dominant Swell Size (ft)"] = df["Dominant Swell Size (ft)"].round(1)
    df["Dominant Swell Period"] = df["Dominant Swell Period"].round(0)
    df["Dominant Swell Direction"] = df["Dominant Swell Direction"].round(0)
    df["Dominant Propagation Score (0-1)"] = df["Dominant Propagation Score (0-1)"].round(1)
    df["Secondary Swell Size (ft)"] = df["Secondary Swell Size (ft)"].round(1)
    df["Secondary Swell Period"] = df["Secondary Swell Period"].round(0)
    df["Secondary Swell Direction"] = df["Secondary Swell Direction"].round(0)
    df["Secondary Propagation Score (0-1)"] = df["Secondary Propagation Score (0-1)"].round(1)
    return df

daylight_df = format_for_display(daylight_df)

top10 = daylight_df.nlargest(10, "Surf Score (1-10)")
st.dataframe(top10, use_container_width=True, height=300)

# =======================================================================================
# DAILY AVERAGES
# =======================================================================================

st.subheader("📊 Daily Surf Averages (Daylight Only)")

daylight_df["day"] = daylight_df["datetime"].dt.date

numeric_cols = [
    "Surf Score (1-10)", 
    "Surf Height Min (ft)",
    "Surf Height Max (ft)",
    "Dominant Swell Size (ft)",
    "Dominant Swell Period",
    "Dominant Swell Direction"
]

daily_avg_df = (
    daylight_df.groupby("day")[numeric_cols]
    .mean()
    .round(2)
    .reset_index()
)

st.dataframe(daily_avg_df, use_container_width=True)

# =======================================================================================
# TIME SERIES EXPLORER 
# =======================================================================================

st.subheader("📈 Time Series Explorer")

# UTILITIES

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
    base = df.reset_index(names="datetime")

    y_enc = alt.Y(
        f"{y_col}:Q",
        title=y_title,
        scale=alt.Scale(domain=domain) if domain else alt.Undefined
    )

    line = (
        alt.Chart(base)
        .mark_line(color=color)
        .encode(
            x=alt.X("datetime:T", title="Date/Time"),
            y=y_enc,
            tooltip=[
                alt.Tooltip("datetime:T", title="Time"),
                alt.Tooltip(f"{y_col}:Q", title=y_title),
            ],
        )
        .properties(height=220)
    )

    return add_daylight_shading(line, base)

def alt_wind_with_gusts(df):
    """
    Wind chart with sustained wind + gust overlay.
    """
    base = df.reset_index(names="datetime")


    # Sustained wind
    wind_line = (
        alt.Chart(base)
        .mark_line(color="green")
        .encode(
            x=alt.X("datetime:T", title="Date/Time"),
            y=alt.Y(
                "Wind Speed (MPH):Q",
                title="Wind Speed (mph)",
                scale=alt.Scale(domain=[0, 30]),
            ),
            tooltip=[
                alt.Tooltip("datetime:T", title="Time"),
                alt.Tooltip("Wind Speed (MPH):Q", title="Wind Speed (mph)", format=".0f"),
            ],
        )
    )

    # Gusts (dashed)
    gust_line = (
        alt.Chart(base)
        .mark_line(color="darkgreen", strokeDash=[4, 4], opacity=0.7)
        .encode(
            x="datetime:T",
            y=alt.Y("Wind Gust (MPH):Q"),
            tooltip=[
                alt.Tooltip("Wind Gust (MPH):Q", title="Wind Gust (mph)", format=".0f"),
            ],
        )
    )

    chart = (wind_line + gust_line).properties(height=220)

    return add_daylight_shading(chart, base)


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
        alt_wind_with_gusts(unfiltered),
        use_container_width=True
    )

with tab4:
    st.altair_chart(
        alt_chart(unfiltered, "Tide Height (ft)", "Tide Height (ft)", domain=[-2, 7], color="teal"),
        use_container_width=True
    )

# =======================================================================================
# RAW TABLE 
# =======================================================================================

st.subheader("📄 Full Forecast Data")

#round key columns for visualization
unfiltered = format_for_display(unfiltered)

st.dataframe(unfiltered, use_container_width=True, height=350)
