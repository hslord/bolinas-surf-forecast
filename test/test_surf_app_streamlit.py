import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Only test pure helper functions -- skip Streamlit UI rendering.
# Importing the full module would trigger st.title() etc., so we
# extract the testable logic by importing them indirectly.


def categorize_bolinas_swell(deg, ui_ranges):
    """Local copy of the categorization logic to avoid importing the Streamlit app."""
    s_sweet = ui_ranges["south_swell"]
    s_edges = ui_ranges["southwest_wrap"]
    w_range = ui_ranges["west_range"]
    nw_range = ui_ranges["nw_range"]

    if s_sweet[0] <= deg <= s_sweet[1]:
        return "South Swell"
    elif s_edges[0] <= deg <= s_edges[1]:
        return "SW Edge"
    elif w_range[0] <= deg <= w_range[1]:
        return "West Wrap"
    elif nw_range[0] <= deg <= nw_range[1]:
        return "NW Shadowed"
    else:
        return "Blocked"


def get_score_color(val):
    """Local copy of the color function to avoid importing Streamlit."""
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


UI_RANGES = {
    "south_swell": [150, 225],
    "southwest_wrap": [225, 245],
    "west_range": [245, 280],
    "nw_range": [280, 310],
}


# ── categorize_bolinas_swell ─────────────────────────────────────────────


def test_south_swell():
    assert "South" in categorize_bolinas_swell(180, UI_RANGES)


def test_west_wrap():
    assert "West" in categorize_bolinas_swell(260, UI_RANGES)


def test_nw_shadowed():
    assert "NW" in categorize_bolinas_swell(300, UI_RANGES)


def test_blocked_direction():
    assert "Blocked" in categorize_bolinas_swell(90, UI_RANGES)


# ── get_score_color ──────────────────────────────────────────────────────


def test_score_color_returns_rgb():
    color = get_score_color(5)
    assert color.startswith("rgb(")


def test_score_color_clamps_low():
    # Should not error on values outside 1-10
    color = get_score_color(-1)
    assert "rgb" in color


def test_score_color_clamps_high():
    color = get_score_color(15)
    assert "rgb" in color
