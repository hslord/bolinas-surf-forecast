import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import yaml
import numpy as np

st.title("🌊 Bolinas Surf Forecast")

# =======================================================================================
# DATA LOADING (pkl)
# =======================================================================================
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


@st.cache_data
def load_config():
    with open("../config/surf_config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
prop_cfg = config['surf_model']['propagation']

# =======================================================================================
# DATA FORMATTING
# =======================================================================================

# Convert degrees to simple cardinal directions or arrows
def categorize_bolinas_swell(deg):
    # Pull ranges from config
    s_sweet = prop_cfg['south_sweet_spot']
    s_edges = prop_cfg['south_edges']
    w_range = prop_cfg['west_range']
    nw_range = prop_cfg['nw_range']

    if s_sweet[0] <= deg <= s_sweet[1]:
        return "🎯 S Sweet Spot"
    elif s_edges[0] <= deg <= s_edges[1]:
        return "🌊 S Edge"
    elif w_range[0] <= deg <= w_range[1]:
        return "🌀 W Wrap"
    elif nw_range[0] <= deg <= nw_range[1]:
        return "🛡️ NW Shadowed"
    else:
        return "❌ Blocked"

# Create a color styling function for scores
def get_score_color(val):
    """Returns an RGB color string based on a 1-10 score."""
    score = max(1, min(10, val))
    if score <= 5:
        f = (score - 1) / 4
        r, g, b = int(231 + f*(241-231)), int(76 + f*(196-76)), int(60 + f*(15-60))
    else:
        f = (score - 5) / 5
        r, g, b = int(241 + f*(46-241)), int(196 + f*(204-196)), int(15 + f*(113-15))
    return f'rgb({r}, {g}, {b})'

# This is the specific version for pandas .style (tables)
def style_surf_score(val):
    color = get_score_color(val)
    return f'color: {color}; font-weight: bold;'

# Compact “summary” columns that read like English

forecast_df = forecast_df.reset_index(names="datetime")
forecast_df["When"] = forecast_df["datetime"].dt.strftime("%a %b %d %I:%M %p")

forecast_df["Surf (ft)"] = (
    forecast_df["Surf Height Min (ft)"].round(1).astype(str)
    + "–"
    + forecast_df["Surf Height Max (ft)"].round(1).astype(str)
)

forecast_df["Dominant"] = (
    forecast_df["Dominant Swell Size (ft)"].round(1).astype(str) + "ft @ " +
    forecast_df["Dominant Swell Period"].round(0).astype(int).astype(str) + "s " +
    forecast_df["Dominant Swell Direction"].apply(categorize_bolinas_swell)
)

forecast_df["Secondary"] = (
    forecast_df["Secondary Swell Size (ft)"].round(1).astype(str) + "ft @ " +
    forecast_df["Secondary Swell Period"].round(0).astype(int).astype(str) + "s " +
    forecast_df["Secondary Swell Direction"].apply(categorize_bolinas_swell)
)

forecast_df["Wind"] = (
    forecast_df["Wind Speed (MPH)"].round(0).astype(int).astype(str) + "g" +
    forecast_df["Wind Gust (MPH)"].round(0).astype(int).astype(str) + " " +
    forecast_df["Wind Direction"].astype(str)
)

# Get current and next tide to find trend
tide_diff = forecast_df["Tide Height (ft)"].diff(periods=-1) # periods=-1 looks at the NEXT row

def get_tide_trend(diff):
    if pd.isna(diff): return "" # Last row handling
    return " Dropping" if diff > 0 else " Rising"

forecast_df["tide_trend"] = tide_diff.apply(get_tide_trend)

# 2. Create the "English" Tide column
forecast_df["Tide (ft)"] = (
    forecast_df["Tide Height (ft)"].round(1).astype(str) + 
    "ft " + 
    forecast_df["tide_trend"]
)

# add columns with shortened names for readability
forecast_df["Surf Score"] = forecast_df["Surf Score (1-10)"]

# =======================================================================================
# SUMMARY CARDS
# =======================================================================================

st.subheader("🏄 Quick Surf Summary")
c1, c2, c3 = st.columns(3)

# 1. Get the data
current = forecast_df.iloc[0]
curr_score = current['Surf Score (1-10)']

daylight_df = forecast_df[forecast_df["is_daylight"]]
best_row = daylight_df.loc[daylight_df["Surf Score (1-10)"].idxmax()]
best_score = best_row['Surf Score (1-10)']

# 2. Define a consistent card template
def summary_card(column, title, value, color="inherit", help_text=None):
    with column:
        st.markdown(f"**{title}**")
        st.markdown(
            f"<h2 style='color:{color}; margin-top:-15px; font-size:1.8rem;'>{value}</h2>", 
            unsafe_allow_html=True,
            help=help_text
        )

# 3. Render Cards
summary_card(c1, "Current Surf", f"{current['Surf Height Min (ft)']}–{current['Surf Height Max (ft)']} ft")

summary_card(c2, "Current Score", f"{curr_score}/10", color=get_score_color(curr_score))

summary_card(c3, "Best Upcoming Session", f"{best_score}/10", 
             color=get_score_color(best_score),
             help_text=f"Best window: {best_row['datetime'].strftime('%b %d, %I:%M %p')}")

# =======================================================================================
# TOP SESSIONS
# =======================================================================================

st.subheader("🏆 Best Upcoming Sessions")

# 1. User Input for filtering
min_score = st.slider("Minimum Surf Score to include:", 1.0, 10.0, 5.0, step=0.5)

# 2. Filter for daylight and user's score threshold
good_windows = forecast_df[
    (forecast_df["is_daylight"]) & 
    (forecast_df["Surf Score (1-10)"] >= min_score)
].copy()

if not good_windows.empty:
    # 2. Identify contiguous blocks of time
    # We check if the current row's time is more than 1 hour away from the previous row
    good_windows = good_windows.sort_values("datetime")
    session_id = (good_windows["datetime"].diff() > pd.Timedelta(hours=1)).cumsum()
    
    # 3. Aggregate by Session
    sessions = good_windows.groupby(session_id).agg({
        "datetime": ["min", "max"],
        "Surf Score (1-10)": "max",
        "Surf Height Min (ft)": "mean",
        "Surf Height Max (ft)": "mean",
        "Wind": "first", # Show the wind at the start of the session
        "Dominant": "first",
        "Secondary": "first",
        "Tide (ft)": "first"
    })

    # Flatten columns
    sessions.columns = ["start", "end", "max_score", "surf_min", "surf_max", "wind", "dom_swell", "sec_swell", "tide"]

    # Calculate Duration
    # a window from 8am to 10am actually represents the 8-9, 9-10, and 10-11 blocks
    sessions["actual_end"] = sessions["end"] + pd.Timedelta(hours=1)
    sessions["Duration"] = (sessions["actual_end"] - sessions["start"])
    sessions["Hours"] = (sessions["Duration"].dt.total_seconds() / 3600).astype(int)
    
    # Format Duration String
    sessions["Length"] = sessions["Hours"].apply(lambda x: f"{x} hr" if x == 1 else f"{x} hrs")
    
    # 4. Format for display
    sessions["Window"] = (
        sessions["start"].dt.strftime("%a %b %d: %I %p") + 
        " - " + 
        sessions["actual_end"].dt.strftime("%I %p")
    )
    
    sessions["Surf (ft)"] = (
        sessions["surf_min"].round(1).astype(str) + "–" + 
        sessions["surf_max"].round(1).astype(str)
    )

    # Sort by score to show the "Top 10" sessions
    top_sessions = sessions.sort_values("max_score", ascending=False).head(10)

    # 5. Display
    st.dataframe(
        top_sessions[["Window", "Length", "max_score", "Surf (ft)", "wind", "dom_swell", "sec_swell", "tide"]]\
            .style.applymap(style_surf_score, subset=['max_score']),
        column_config={
            "Window": "Time Block",
            "Length": "Length",
            "max_score": "Peak Score",
            "Surf (ft)": "Avg Size (ft)",
            "wind": "Start Wind",
            "dom_swell": "Dominant Swell",
            "sec_swell": "Secondary Swell",
            "tide": "Start Tide"

        },
        use_container_width=True,
        hide_index=True
    )
else:
    st.write("No high-quality sessions found in the current forecast.")

# =======================================================================================
# DAILY OUTLOOK
# =======================================================================================
st.subheader("📊 Daily Outlook (Daylight Hours)")

# 1. Prepare the data
# daylight_df["Day"] = daylight_df["datetime"].dt.strftime("%a, %b %d")
daylight_df["Date"] = daylight_df["datetime"].dt.date

# Define how we want to aggregate each column
daily_summary = daylight_df.groupby("Date").agg({
    "Surf Score (1-10)": "max",
    "Surf Height Min (ft)": "min",
    "Surf Height Max (ft)": "max",
    "Dominant Swell Period": "mean",
    "Dominant Swell Size (ft)": "mean",
    "Dominant Swell Direction": "mean",
    "Secondary Swell Period": "mean",
    "Secondary Swell Size (ft)": "mean",
    "Secondary Swell Direction": "mean",
    "Wind Speed (MPH)": ["min", "max"]
})

daily_summary.columns = [
    "_".join(col).strip("_") if isinstance(col, tuple) else col 
    for col in daily_summary.columns.values
]

# 2. Create "Human Readable" columns
daily_summary["Surf Range"] = (
    daily_summary["Surf Height Min (ft)_min"].round(1).astype(str) + 
    " - " + 
    daily_summary["Surf Height Max (ft)_max"].round(1).astype(str) + " ft"
)

daily_summary["Avg Dominant Swell"] = daily_summary["Dominant Swell Size (ft)_mean"].round(0).astype(int).astype(str) + "ft @ " \
                                        + daily_summary["Dominant Swell Period_mean"].round(0).astype(int).astype(str) + "s " \
                                        + daily_summary["Dominant Swell Direction_mean"].apply(categorize_bolinas_swell)

daily_summary["Avg Secondary Swell"] = daily_summary["Secondary Swell Size (ft)_mean"].round(0).astype(int).astype(str) + "ft @ " \
                                        + daily_summary["Secondary Swell Period_mean"].round(0).astype(int).astype(str) + "s " \
                                        + daily_summary["Secondary Swell Direction_mean"].apply(categorize_bolinas_swell)

daily_summary["Wind Range"] = (
    daily_summary["Wind Speed (MPH)_min"].round(0).astype(str) + 
    " - " + 
    daily_summary["Wind Speed (MPH)_max"].round(0).astype(str) + " MPH"
)

# 3. Final selection for display
display_df = daily_summary[["Surf Score (1-10)_max", "Surf Range", "Avg Dominant Swell", "Avg Secondary Swell", "Wind Range"]].copy()
display_df.columns = ["Max Surf Score", "Surf Range", "Avg Dominant Swell", "Avg Secondary Swell", "Wind Range"]

st.dataframe(display_df.style.applymap(style_surf_score, subset=['Max Surf Score']), 
            use_container_width=True)

# =======================================================================================
# SURF QUALITY BREAKDOWN PANEL
# =======================================================================================

st.subheader("🔎 Surf Quality Breakdown")

# Build timestamp list for selectbox
timestamps = forecast_df["datetime"].tolist()

# Safe default: highest surf score timestamp
best_timestamp = forecast_df.loc[
    forecast_df["Surf Score (1-10)"].idxmax(),
    "datetime"
]

default_idx = timestamps.index(best_timestamp)

selected_time = st.selectbox(
    "Select a forecast time to inspect:",
    options=timestamps,
    index=default_idx,
    format_func=lambda t: t.strftime("%b %d, %I:%M %p"),
)

# 1. Row selection
row = forecast_df.loc[forecast_df["datetime"] == selected_time].iloc[0]

row = forecast_df.loc[forecast_df["datetime"] == selected_time].iloc[0]

with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)

    # Helper function to keep this section clean
    def breakdown_item(column, score, label, detail_text, is_multi=False):
        color = get_score_color(score)
        with column:
            # Score Heading
            st.markdown(f"<h3 style='color:{color}; margin-bottom: 0px;'>{score}/10</h3>", unsafe_allow_html=True)
            # Label
            st.caption(label)
            # Details
            if is_multi:
                st.markdown(detail_text, unsafe_allow_html=True)
            else:
                st.write(f"**{detail_text}**")

    # Column 1: Overall
    breakdown_item(col1, row['Surf Score (1-10)'], "Overall Grade", row['Surf (ft)'])

    # Column 2: Swell (using is_multi for the secondary swell styling)
    swell_html = f"**Dom:** {row['Dominant']}<br><span style='font-size:0.8rem; color:grey;'>**Sec:** {row['Secondary']}</span>"
    breakdown_item(col2, row['Dominant Swell Score (1-10)'], "Swell Quality", swell_html, is_multi=True)

    # Column 3: Wind
    breakdown_item(col3, row['Wind Score (1-10)'], "Wind Quality", row['Wind'])

    # Column 4: Tide
    breakdown_item(col4, row['Tide Score (1-10)'], "Tide Quality", f"{row['Tide (ft)']}")

# 4. Pro-Tip: Add a "Why this score?" helper text
with st.expander("How are these scores calculated?"):
    # 1. Dynamically build the Swell help text from config
    # This loops through west_range, nw_range, south_sweet_spot, etc.
    swell_notes = []
    for key, value in prop_cfg.items():
        if isinstance(value, list) and len(value) == 2:
            # Format name: "west_range" -> "West Range"
            range_name = key.replace("_", " ").title()
            swell_notes.append(f"{range_name} ({value[0]}°-{value[1]}°)")
    
    swell_help = " • ".join(swell_notes)

    # 2. Render the Markdown
    st.write(f"""
    - **Swell:** Optimized for propagation from:  
      {swell_help}
    - **Wind:** Optimized for offshore flow relative to the coast orientation of **{config['data_sources']['coast_orientation']}°**.
    - **Tide:** The "Tide Score" is highest when the height is between **{config['surf_model']['tide']['optimal_low']}ft** and **{config['surf_model']['tide']['optimal_high']}ft**.
    """)

# =======================================================================================
# TIME SERIES EXPLORER 
# =======================================================================================

st.subheader("📈 Time Series Explorer")

# TIME SERIES UTILITIES

def build_night_rects(df):
    """
    Build Altair rectangle blocks for nighttime using is_daylight == False.
    Daytime remains unshaded for clarity.
    """
    if "is_daylight" not in df.columns:
        return pd.DataFrame(columns=["start", "end"])

    #df = df.reset_index().rename(columns={"index": "datetime"})
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
    #base = df.reset_index(names="datetime")

    y_enc = alt.Y(
        f"{y_col}:Q",
        title=y_title,
        scale=alt.Scale(domain=domain) if domain else alt.Undefined
    )

    line = (
        alt.Chart(df)
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

    return add_daylight_shading(line, df)

def alt_wind_with_gusts(df):
    """
    Wind chart with sustained wind + gust overlay.
    """
    #base = df.reset_index(names="datetime")


    # Sustained wind
    wind_line = (
        alt.Chart(df)
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
        alt.Chart(df)
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

    return add_daylight_shading(chart, df)


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

st.dataframe(forecast_df, use_container_width=True, height=350, hide_index=True)

# -------------------------------------------------
# Download as CSV
# -------------------------------------------------
csv = unfiltered.to_csv(index=True).encode("utf-8")

st.download_button(
    label="⬇️ Download forecast as CSV",
    data=csv,
    file_name="bolinas_surf_forecast.csv",
    mime="text/csv",
)
