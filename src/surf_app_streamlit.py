import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import yaml
import os
from datetime import datetime, timedelta
import numpy as np
from streamlit_gsheets import GSheetsConnection

st.title("🌊 Bolinas Surf Forecast")

# =======================================================================================
# DATA LOADING (pkl)
# =======================================================================================
base_dir = Path(__file__).resolve().parents[1]

@st.cache_data(ttl=3600, show_spinner=True)
def load_forecast():
    parquet_path = base_dir / "data" / "forecast_df.parquet"

    if not parquet_path.exists():
        st.error(f"Parquet file not found at: {parquet_path}")
        st.stop()

    return pd.read_parquet(parquet_path)


forecast_df = load_forecast()
if not isinstance(forecast_df.index, pd.DatetimeIndex):
    forecast_df.index = pd.to_datetime(forecast_df.index)


@st.cache_data
def load_config():
    config_path = base_dir / "config" / "surf_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


try:
    config = load_config()
    surf_cfg = config["surf_model"]
except Exception as e:
    st.error(f"Config load failed: {type(e).__name__}: {e}")
    st.stop()

# =======================================================================================
# DATA FORMATTING
# =======================================================================================


# Convert degrees to simple cardinal directions or arrows
def categorize_bolinas_swell(deg):
    # Pull ranges from config
    s_sweet = surf_cfg["ui_ranges"]["south_swell"]
    s_edges = surf_cfg["ui_ranges"]["southwest_wrap"]
    w_range = surf_cfg["ui_ranges"]["west_range"]
    nw_range = surf_cfg["ui_ranges"]["nw_range"]

    if s_sweet[0] <= deg <= s_sweet[1]:
        return "🎯 South Swell"
    elif s_edges[0] <= deg <= s_edges[1]:
        return "🌊 SW Edge"
    elif w_range[0] <= deg <= w_range[1]:
        return "🌀 West Wrap"
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
        r, g, b = (
            int(231 + f * (241 - 231)),
            int(76 + f * (196 - 76)),
            int(60 + f * (15 - 60)),
        )
    else:
        f = (score - 5) / 5
        r, g, b = (
            int(241 + f * (46 - 241)),
            int(196 + f * (204 - 196)),
            int(15 + f * (113 - 15)),
        )
    return f"rgb({r}, {g}, {b})"


def circular_mean_deg(series):
    """Compute mean of compass directions using vector averaging."""
    rads = np.deg2rad(series.dropna())
    if len(rads) == 0:
        return np.nan
    return np.rad2deg(np.arctan2(np.sin(rads).mean(), np.cos(rads).mean())) % 360


circular_mean_deg.__name__ = "mean"


# This is the specific version for pandas .style (tables)
def style_surf_score(val):
    color = get_score_color(val)
    return f"color: {color}; font-weight: bold;"


# Compact “summary” columns that read like English

forecast_df = forecast_df.reset_index(names="datetime")
forecast_df["When"] = forecast_df["datetime"].dt.strftime("%a %b %d %I:%M %p")

forecast_df["Surf (ft)"] = (
    forecast_df["Surf Height Min (ft)"].round(1).astype(str)
    + "–"
    + forecast_df["Surf Height Max (ft)"].round(1).astype(str)
)

forecast_df["Dominant"] = (
    forecast_df["Dominant Swell Size (ft)"].fillna(0).round(1).astype(str)
    + "ft @ "
    + forecast_df["Dominant Swell Period"].fillna(0).round(0).astype(int).astype(str)
    + "s "
    + forecast_df["Dominant Swell Direction"].fillna(0).apply(categorize_bolinas_swell)
)

forecast_df["Secondary"] = (
    forecast_df["Secondary Swell Size (ft)"].fillna(0).round(1).astype(str)
    + "ft @ "
    + forecast_df["Secondary Swell Period"].fillna(0).round(0).astype(int).astype(str)
    + "s "
    + forecast_df["Secondary Swell Direction"].fillna(0).apply(categorize_bolinas_swell)
)

forecast_df["Wind"] = (
    forecast_df["Wind Speed (MPH)"].fillna(0).round(0).astype(int).astype(str)
    + "g"
    + forecast_df["Wind Gust (MPH)"].fillna(0).round(0).astype(int).astype(str)
    + " "
    + forecast_df["Wind Direction"].astype(str)
)

# Get current and next tide to find trend
tide_diff = forecast_df["Tide Height (ft)"].diff(
    periods=-1
)  # periods=-1 looks at the NEXT row


def get_tide_trend(diff):
    if pd.isna(diff):
        return ""  # Last row handling
    return " Dropping" if diff > 0 else " Rising"


forecast_df["tide_trend"] = tide_diff.apply(get_tide_trend)

# 2. Create the "English" Tide column
forecast_df["Tide (ft)"] = (
    forecast_df["Tide Height (ft)"].round(1).astype(str)
    + "ft "
    + forecast_df["tide_trend"]
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
curr_score = current["Surf Score (1-10)"]

daylight_df = forecast_df[forecast_df["is_daylight"]]
best_row = daylight_df.loc[daylight_df["Surf Score (1-10)"].idxmax()]
best_score = best_row["Surf Score (1-10)"]


# 2. Define a consistent card template
def summary_card(column, title, value, color="inherit", help_text=None):
    with column:
        st.markdown(f"**{title}**")
        st.markdown(
            f"<h2 style='color:{color}; margin-top:-15px; font-size:1.8rem;'>{value}</h2>",
            unsafe_allow_html=True,
            help=help_text,
        )


# 3. Render Cards
summary_card(
    c1,
    "Latest Surf",
    f"{current['Surf Height Min (ft)']}–{current['Surf Height Max (ft)']} ft",
)
mtime = os.path.getmtime(base_dir / "data" / "forecast_df.parquet")
last_updated_dt = datetime.fromtimestamp(mtime)
st.caption(f"Last updated: {last_updated_dt.strftime('%b %d, %I:%M %p')}")

summary_card(c2, "Latest Score", f"{curr_score}/10", color=get_score_color(curr_score))

summary_card(
    c3,
    "Best Upcoming Session",
    f"{best_score}/10",
    color=get_score_color(best_score),
    help_text=f"Best window: {best_row['datetime'].strftime('%b %d, %I:%M %p')}",
)


# =======================================================================================
# TOP SESSIONS
# =======================================================================================

st.subheader("🏆 Best Upcoming Sessions")

# 1. User Input for filtering
min_score = st.slider("Minimum Surf Score to include:", 1.0, 10.0, 6.0, step=0.5)

# 2. Filter for daylight and user's score threshold
good_windows = forecast_df[
    (forecast_df["is_daylight"]) & (forecast_df["Surf Score (1-10)"] >= min_score)
].copy()

if not good_windows.empty:
    # 2. Identify contiguous blocks of time
    # We check if the current row's time is more than 1 hour away from the previous row
    good_windows = good_windows.sort_values("datetime")
    session_id = (good_windows["datetime"].diff() > pd.Timedelta(hours=1)).cumsum()

    # 3. Aggregate by Session
    sessions = good_windows.groupby(session_id).agg(
        {
            "datetime": ["min", "max"],
            "Surf Score (1-10)": "max",
            "Surf Height Min (ft)": "mean",
            "Surf Height Max (ft)": "mean",
            "Wind Speed (MPH)": ["min", "max"],
            "Wind Direction": "first",
            "Dominant": "first",
            "Secondary": "first",
            "Tide (ft)": "first",
        }
    )

    # Flatten columns
    sessions.columns = [
        "start",
        "end",
        "max_score",
        "surf_min",
        "surf_max",
        "wind_speed_min",
        "wind_speed_max",
        "wind_direction",
        "dom_swell",
        "sec_swell",
        "tide",
    ]

    def format_wind_range(row):
        w_min = round(row["wind_speed_min"])
        w_max = round(row["wind_speed_max"])
        w_direction = row["wind_direction"]
        if w_min == w_max:
            return f"{w_min} {w_direction}"
        return f"{w_min}–{w_max} {w_direction}"

    sessions["wind_range"] = sessions.apply(format_wind_range, axis=1)

    # Calculate Duration
    # a window from 8am to 10am actually represents the 8-9, 9-10, and 10-11 blocks
    sessions["actual_end"] = sessions["end"] + pd.Timedelta(hours=1)
    sessions["Duration"] = sessions["actual_end"] - sessions["start"]
    sessions["Hours"] = (sessions["Duration"].dt.total_seconds() / 3600).astype(int)

    # Format Duration String
    sessions["Length"] = sessions["Hours"].apply(
        lambda x: f"{x} hr" if x == 1 else f"{x} hrs"
    )

    # 4. Format for display
    sessions["Window"] = (
        sessions["start"].dt.strftime("%a %b %d: %I %p")
        + " - "
        + sessions["actual_end"].dt.strftime("%I %p")
    )

    sessions["Surf (ft)"] = (
        sessions["surf_min"].round(1).astype(str)
        + "–"
        + sessions["surf_max"].round(1).astype(str)
    )

    # Sort by score to show the "Top 10" sessions
    top_sessions = sessions.sort_values("max_score", ascending=False).head(10)

    # 5. Display
    st.dataframe(
        top_sessions[
            [
                "Window",
                "Length",
                "max_score",
                "Surf (ft)",
                "wind_range",
                "dom_swell",
                "sec_swell",
                "tide",
            ]
        ].style.map(style_surf_score, subset=["max_score"]),
        column_config={
            "Window": "Time Block",
            "Length": "Length",
            "max_score": "Peak Score",
            "Surf (ft)": "Avg Size (ft)",
            "wind_range": "Wind Range (MPH)",
            "dom_swell": "Dominant Swell",
            "sec_swell": "Secondary Swell",
            "tide": "Start Tide",
        },
        use_container_width=True,
        hide_index=True,
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
daily_summary = daylight_df.groupby("Date").agg(
    {
        "Surf Score (1-10)": "max",
        "Surf Height Min (ft)": "min",
        "Surf Height Max (ft)": "max",
        "Dominant Swell Period": "mean",
        "Dominant Swell Size (ft)": "mean",
        "Dominant Swell Direction": circular_mean_deg,
        "Secondary Swell Period": "mean",
        "Secondary Swell Size (ft)": "mean",
        "Secondary Swell Direction": circular_mean_deg,
        "Wind Speed (MPH)": ["min", "max"],
    }
)

daily_summary.columns = [
    "_".join(col).strip("_") if isinstance(col, tuple) else col
    for col in daily_summary.columns.values
]

# 2. Create "Human Readable" columns
daily_summary["Surf Range"] = (
    daily_summary["Surf Height Min (ft)_min"].round(1).astype(str)
    + " - "
    + daily_summary["Surf Height Max (ft)_max"].round(1).astype(str)
    + " ft"
)

daily_summary["Avg Dominant Swell"] = (
    daily_summary["Dominant Swell Size (ft)_mean"]
    .fillna(0)
    .round(0)
    .astype(int)
    .astype(str)
    + "ft @ "
    + daily_summary["Dominant Swell Period_mean"]
    .fillna(0)
    .round(0)
    .astype(int)
    .astype(str)
    + "s "
    + daily_summary["Dominant Swell Direction_mean"]
    .fillna(0)
    .apply(categorize_bolinas_swell)
)

daily_summary["Avg Secondary Swell"] = (
    daily_summary["Secondary Swell Size (ft)_mean"]
    .fillna(0)
    .round(0)
    .astype(int)
    .astype(str)
    + "ft @ "
    + daily_summary["Secondary Swell Period_mean"]
    .fillna(0)
    .round(0)
    .astype(int)
    .astype(str)
    + "s "
    + daily_summary["Secondary Swell Direction_mean"]
    .fillna(0)
    .apply(categorize_bolinas_swell)
)

daily_summary["Wind Range"] = (
    daily_summary["Wind Speed (MPH)_min"].fillna(0).round(0).astype(str)
    + " - "
    + daily_summary["Wind Speed (MPH)_max"].fillna(0).round(0).astype(str)
    + " MPH"
)

# 3. Final selection for display
display_df = daily_summary[
    [
        "Surf Score (1-10)_max",
        "Surf Range",
        "Avg Dominant Swell",
        "Avg Secondary Swell",
        "Wind Range",
    ]
].copy()
display_df.columns = [
    "Max Surf Score",
    "Surf Range",
    "Avg Dominant Swell",
    "Avg Secondary Swell",
    "Wind Range",
]

st.dataframe(
    display_df.style.map(style_surf_score, subset=["Max Surf Score"]),
    use_container_width=True,
)

# =======================================================================================
# SURF QUALITY BREAKDOWN PANEL
# =======================================================================================

st.subheader("🔎 Surf Quality Breakdown")

# Build timestamp list for selectbox
timestamps = forecast_df["datetime"].tolist()

# Safe default: highest surf score timestamp
best_timestamp = daylight_df.loc[daylight_df["Surf Score (1-10)"].idxmax(), "datetime"]

default_idx = timestamps.index(best_timestamp)

selected_time = st.selectbox(
    "Select a forecast time to inspect:",
    options=timestamps,
    index=default_idx,
    format_func=lambda t: t.strftime("%b %d, %I:%M %p"),
)

# Row selection
time_row = forecast_df.loc[forecast_df["datetime"] == selected_time].iloc[0]

with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)

    # Helper function to keep this section clean
    def breakdown_item(column, score, label, detail_text, is_multi=False):
        color = get_score_color(score)
        with column:
            # Score Heading
            st.markdown(
                f"<h3 style='color:{color}; margin-bottom: 0px;'>{score}/10</h3>",
                unsafe_allow_html=True,
            )
            # Label
            st.caption(label)
            # Details
            if is_multi:
                st.markdown(detail_text, unsafe_allow_html=True)
            else:
                st.write(f"**{detail_text}**")

    # Column 1: Overall
    breakdown_item(
        col1,
        time_row["Surf Score (1-10)"],
        "Overall Grade",
        f"{time_row['Surf (ft)']} ft",
    )

    # Column 2: Swell (using is_multi for the secondary swell styling)
    swell_html = f"**Dom:** {time_row['Dominant']}<br><span style='font-size:0.8rem; color:grey;'>**Sec:** {time_row['Secondary']}</span>"
    breakdown_item(
        col2,
        time_row["Dominant Swell Score (1-10)"],
        "Swell Quality",
        swell_html,
        is_multi=True,
    )

    # Column 3: Wind
    breakdown_item(
        col3, time_row["Wind Score (1-10)"], "Wind Quality", time_row["Wind"]
    )

    # Column 4: Tide
    breakdown_item(
        col4, time_row["Tide Score (1-10)"], "Tide Quality", f"{time_row['Tide (ft)']}"
    )

# Add a "Why this score?" helper text
with st.expander("How are these scores calculated?"):
    # Dynamically build the Swell help text from config
    # This loops through west_range, nw_range, south_sweet_spot, etc.
    swell_notes = []
    for key, ui_range in surf_cfg["ui_ranges"].items():
        if isinstance(ui_range, list) and len(ui_range) == 2:
            # Format name: "west_range" -> "West Range"
            range_name = key.replace("_", " ").title()
            swell_notes.append(f"{range_name} ({ui_range[0]}°-{ui_range[1]}°)")

    swell_help = " • ".join(swell_notes)

    # Render the Markdown
    st.write(
        f"""
    - **Swell:** Optimized for propagation from:  
      {swell_help}
    - **Wind:** Optimized for offshore flow relative to the coast orientation of **{config["data_sources"]["coast_orientation"]}°**.
    - **Tide:** The "Tide Score" is highest when the height is between **-1 and 3 ft**.
    """
    )

# =======================================================================================
# TIME SERIES EXPLORER
# =======================================================================================

st.subheader("📈 Time Series Explorer")

# TIME SERIES UTILITIES


def build_night_rects(light_df):
    """
    Build Altair rectangle blocks for nighttime using is_daylight == False.
    Daytime remains unshaded for clarity.
    """
    if "is_daylight" not in light_df.columns:
        return pd.DataFrame(columns=["start", "end"])

    rects = []
    in_block = False
    start_time = None

    for _, row in light_df.iterrows():
        if not row["is_daylight"] and not in_block:
            in_block = True
            start_time = row["datetime"]

        if row["is_daylight"] and in_block:
            in_block = False
            rects.append({"start": start_time, "end": row["datetime"]})

    # If ending at night
    if in_block:
        rects.append({"start": start_time, "end": light_df["datetime"].iloc[-1]})

    return pd.DataFrame(rects)


def add_daylight_shading(line_chart, light_df):
    """Overlay darker shading for nighttime blocks."""
    rect_df = build_night_rects(light_df)

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


def alt_chart(chart_df, y_col, y_title, domain=None, color="steelblue"):
    """Unified line chart with visible night shading."""

    y_enc = alt.Y(
        f"{y_col}:Q",
        title=y_title,
        scale=alt.Scale(domain=domain) if domain else alt.Undefined,
    )

    line = (
        alt.Chart(chart_df)
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

    return add_daylight_shading(line, chart_df)


def alt_wind_with_gusts(wind_df):
    """
    Wind chart with sustained wind + gust overlay.
    """

    # Sustained wind
    wind_line = (
        alt.Chart(wind_df)
        .mark_line(color="green")
        .encode(
            x=alt.X("datetime:T", title="Date/Time"),
            y=alt.Y(
                "Wind Speed (MPH):Q",
                title="Wind Speed (mph)",
                scale=alt.Scale(domain=[0, 40]),
            ),
            tooltip=[
                alt.Tooltip("datetime:T", title="Time"),
                alt.Tooltip(
                    "Wind Speed (MPH):Q", title="Wind Speed (mph)", format=".0f"
                ),
            ],
        )
    )

    # Gusts (dashed)
    gust_line = (
        alt.Chart(wind_df)
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

    return add_daylight_shading(chart, wind_df)


# Important: ALWAYS use unfiltered forecast_df for shading accuracy
unfiltered = forecast_df.copy()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Surf Score", "Wave Height", "Wind Speed", "Tide Height"]
)

with tab1:
    st.altair_chart(
        alt_chart(unfiltered, "Surf Score (1-10)", "Surf Score", domain=[0, 10]),
        use_container_width=True,
    )

with tab2:
    st.altair_chart(
        alt_chart(
            unfiltered, "Surf Height Max (ft)", "Surf Height (ft)", domain=[0, 8]
        ),
        use_container_width=True,
    )

with tab3:
    st.altair_chart(alt_wind_with_gusts(unfiltered), use_container_width=True)

with tab4:
    st.altair_chart(
        alt_chart(
            unfiltered,
            "Tide Height (ft)",
            "Tide Height (ft)",
            domain=[-2, 7],
            color="teal",
        ),
        use_container_width=True,
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

# =======================================================================================
# FEEDBACK
# =======================================================================================

st.subheader("Report Live Conditions")
st.info("Your feedback helps tune the Bolinas Surf Forecast.")

with st.form("surf_report_form", clear_on_submit=True):
    # Row 1: The Timeline
    report_date = st.date_input("Date", value=datetime.now())
    col_start, col_end = st.columns(2)
    
    with col_start:
        start_time = st.time_input("Start Time", value=datetime.now().time())
    
    with col_end:
        # Defaults to 1 hour later
        default_end = (datetime.combine(report_date, start_time) + timedelta(hours=1)).time()
        end_time = st.time_input("End Time", value=default_end)

    # Row 2: The Conditions
    col_min, col_max = st.columns(2)
    with col_min:
        observed_min = st.number_input("Size Min (ft)", min_value=0.0, value=2.0, step=0.5)
    with col_max:
        observed_max = st.number_input("Size Max (ft)", min_value=0.0, value=3.0, step=0.5)

    # Row 3: Rating & User
    surf_rating = st.select_slider("Surf Rating (1-10)", options=range(1, 11), value=5)
    user_name = st.text_input("User (Optional)", placeholder="Your Name or Initials")

    # Row 4: Qualitative
    notes = st.text_area("Notes", placeholder="Describe the crowd, wind, or 10s+ swell energy...")

    # The button that finally triggers the rerun
    submitted = st.form_submit_button("Submit Report")
# The sheet
conn = st.connection("gsheets", type=GSheetsConnection)
sheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]

if submitted:
    try:
        # Map the form inputs to your GSheet columns
        new_entry = pd.DataFrame([{
            "Date": report_date.strftime("%Y-%m-%d"),
            "Start Time": start_time.strftime("%H:%M"),
            "End Time": end_time.strftime("%H:%M"),
            "Size Min": observed_min,
            "Size Max": observed_max,
            "Surf Rating (1-10)": surf_rating,
            "Notes": notes.replace(",", ";"), # Clean notes for CSV/Spreadsheet safety
            "User": user_name if user_name else "Anonymous"
        }])
        # Read the existing sheet 
        existing_data = conn.read(spreadsheet=sheet_url, worksheet="Sheet1")
        
        # Combine the old data with the new entry
        updated_df = pd.concat([existing_data, new_entry], ignore_index=True)
        
        # Push the updated dataframe back to Google
        conn.update(spreadsheet=sheet_url, worksheet="Sheet1", data=updated_df)
        
        st.success("Thank you for your feedback!")
    
    except Exception as e:
        st.error(f"Something went wrong: {e}")
        
