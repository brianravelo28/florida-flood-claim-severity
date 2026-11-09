"""
Florida Flood Claim Severity - interactive Dash app (real FEMA NFIP flood-insurance claims, payouts in 2025 dollars).

A shared filter bar above the tabs (Loss Years, County, Occupancy, Flood Zone) drives every tab.

Tabs:
1. Overview          claims, total paid and median claim by year of loss and by decade; county map
2. Storms            the events FEMA labels on its claims, ranked by claims, dollars paid or median claim
3. Factors           how payouts differ by flood zone, occupancy, elevation and building age

Every chart and table shows the number of claims behind it; groups with fewer than MIN_CLAIMS claims are flagged or left out.

Run locally: python app.py   (port from $PORT, default 8058).  Hosted: gunicorn wsgi:application
Data: deploy_data/ (built by build_deploy_data.py).
"""
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from constants import FACTOR_COLUMNS, FEMA_PULL_DATE, FLOOD_ORDER, MIN_CLAIMS, OCC_ORDER

BASE_DIR = Path(__file__).resolve().parent
DEPLOY = BASE_DIR / "deploy_data"
_T0 = time.time()


def _log(stage):
    """Startup progress to stderr (elapsed seconds + resident memory on Linux) for the host's logs."""
    rss = ""
    try:
        with open("/proc/self/status") as f:
            rss = f", {int(next(l.split()[1] for l in f if l.startswith('VmRSS'))) // 1024} MB"
    except (OSError, StopIteration):
        pass
    print(f"[app] {stage} ({time.time() - _T0:.1f}s{rss})", file=sys.stderr, flush=True)


# Design tokens (light palette)
SURFACE, PAGE_BG, BORDER, GRID = "#fcfcfb", "#f4f4f2", "#e5e4df", "#ecebe7"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#6b6a66"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT = "rgba(42,120,214,0.35)"
SERIES = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#7b61ff"]

# ---------------------------------------------------------------- data ----
# One compact row per paid claim (categoricals + float32, ~4 MB in memory). Every tab filters and aggregates it live, so
# nothing is a precomputed summary of a subset.
claims = pd.read_parquet(DEPLOY / "claims.parquet")
claims["decade"] = (claims["yearOfLoss"] // 10 * 10).astype("int16")
AGG = json.loads((DEPLOY / "aggregates.json").read_text())
REPORT = json.loads((DEPLOY / "build_report.json").read_text())
SUMMARY = json.loads((DEPLOY / "summary.json").read_text())
_log("data loaded")

YEAR_MIN, YEAR_MAX, N_CLAIMS = AGG["year_min"], AGG["year_max"], AGG["n_claims"]
LOG_TICKS = [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6]
CLOSED_WITHOUT_PAYMENT_PCT = 100 * REPORT["rows_dropped"]["amountPaid <= 0 (closed without payment)"] / REPORT["raw_rows"]
AGE_BY_DECADE = {int(k): v for k, v in SUMMARY["median_building_age_at_loss_by_decade"].items()}
UNUSABLE_DATES = REPORT["construction_year_unusable"]
PLACEHOLDER_SHARE = 100 * REPORT["construction_date_placeholder_1492"] / UNUSABLE_DATES

# Short forms for chart x-axis ticks only; full names stay in the dropdown, table, and hover.
AXIS_SHORT_LABELS = {
    "X / B / C (moderate-minimal)": "X / B / C",
    "A / AH / AO / AR / A99 (no base elevation, shallow, special)": "A / AH / AO...",
    "AE / A1-A30 (base elevation known)": "AE / A1-A30", "V / VE (velocity, coastal)": "V / VE",
    "Condo association": "Condo assoc.", "Unit in multi-unit building": "Multi-unit", "Non-residential": "Non-resid.",
    "Three or more floors": "3+ floors", "Manufactured home or trailer on foundation": "Mfd. home / trailer",
    "Townhouse, three or more floors": "Townhouse, 3+", "Not a primary residence": "Not primary",
    "Built before the first flood map": "Before first map", "Built after the first flood map": "After first map",
    "No building coverage": "None",
    "Accumulation of rainfall or snowmelt": "Rainfall / snowmelt", "Tidal water overflow": "Tidal overflow",
    "Stream, river or lake overflow": "Stream / river / lake", "Alluvial fan overflow": "Alluvial fan",
    "Below the base flood level": "Below", "At the base flood level": "At", "3 ft or more above": "3+ ft above",
    "Elevated, no enclosure (post, pile or pier)": "Elevated, open",
    "Elevated with enclosure (post, pile or pier)": "Enclosed, post/pile",
    "Elevated with enclosure (not post, pile or pier)": "Enclosed, other",
}

FACTOR_NOTES = {
    "Flood zone": None,
    "Occupancy type": "FEMA's legacy (1-4, 6) and newer (11-19) occupancy codes describe the same building types, so they are grouped together.",
    "Elevated building": None,
    "Building floors": None,
    "Primary residence": None,
    "Building coverage": None,
    "Foundation type": "FEMA records the foundation type mostly for recent claims; earlier claims are listed as Not recorded.",
    "Elevation vs base flood level": "Feet between the building's lowest floor, as used for rating, and the base flood elevation.",
    "Building deductible": None,
    "Cause of flooding": None,
    "Flood water depth": "Depth in inches. FEMA notes that some measurements were recorded in feet, and negative values are counted as "
                         "not recorded.",
    "Built after first flood map": (
        "FEMA counts a building as built after the first flood map (post-FIRM) when construction started after December 31, 1974 "
        "or after the community's first Flood Insurance Rate Map, whichever is later. These buildings are newer, so the comparison "
        "overlaps with building age."),
    "Building age": (
        f"Age at time of loss. About {REPORT['construction_year_unknown_pct']:.0f}% of buildings have an unusable construction date "
        f"({PLACEHOLDER_SHARE:.0f}% of them FEMA's 1492 placeholder, and every placeholder is flagged pre-FIRM, so they are probably old). "
        "Those buildings are placed using the age of known-date pre-FIRM buildings from the same loss year. "
        "Age and loss year move together: the median building age at loss was "
        f"{AGE_BY_DECADE[1980]:.0f} years in the 1980s, {AGE_BY_DECADE[2010]:.0f} in the 2010s and {AGE_BY_DECADE[2020]:.0f} in the 2020s "
        "(buildings with known dates), and payouts rose over the same period, so larger payouts among older buildings may reflect "
        "when the claim happened. The chart below compares building ages within each decade of loss."),
}
FACTORS = {name: (col, order, FACTOR_NOTES[name]) for name, (col, order) in FACTOR_COLUMNS.items()}

COUNTY_INFO = AGG["counties"]


def _present(col, order=None):
    """Category labels that actually occur in the data (a category with no claims isn't a useful filter option)."""
    have = set(claims[col].dropna().unique())
    return [o for o in (order or sorted(have)) if o in have]


COUNTY_OPTIONS = _present("countyName")
OCC_OPTIONS = _present("occupancy_group", OCC_ORDER)
FLOOD_OPTIONS = _present("flood_zone_group", FLOOD_ORDER)


@lru_cache(maxsize=4)
def _subset(years, counties, occs, floods):
    m = claims["yearOfLoss"].between(years[0], years[1]).to_numpy()
    for col, sel in (("countyName", counties), ("occupancy_group", occs), ("flood_zone_group", floods)):
        if sel:
            m = m & claims[col].isin(sel).to_numpy()
    return claims[m]


def filtered(years, counties, occs, floods):
    """The claims matching the global filter bar (an empty multi-select means 'all')."""
    return _subset(tuple(years or (YEAR_MIN, YEAR_MAX)), tuple(counties or ()), tuple(occs or ()), tuple(floods or ()))


_log("precompute done")


# ------------------------------------------------------------- helpers ----
def money(x):
    x = float(x)
    if abs(x) >= 1e9:
        return f"${x / 1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"${x / 1e6:.1f}M"
    if abs(x) >= 1e4:
        return f"${x / 1e3:.0f}k"
    return f"${x:,.0f}"


def pct(x):
    return f"{x:.1f}%" if isinstance(x, (int, float)) else "-"


def compact_count(n):
    n = int(n)
    return f"{n / 1e3:.0f}k" if n >= 10_000 else f"{n / 1e3:.1f}k" if n >= 1_000 else str(n)


def years_label(years):
    lo, hi = int(min(years)), int(max(years))
    if lo == hi:
        return str(lo)
    return f"{lo}-{str(hi)[-2:]}" if lo // 10 == hi // 10 else f"{lo}-{hi}"


# ---------------------------------------------------------- app + style ----
CSS = f"""
:root {{ color-scheme: light; }}
body {{ margin: 0; background: {PAGE_BG}; color: {INK};
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
.wrap {{ font-size: 15px; max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }}
.card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; padding: 16px 18px; }}
.kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
.kpi {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; padding: 12px 16px; }}
.kpi .v {{ font-size: 28px; font-weight: 650; letter-spacing: -0.01em; }}
.kpi .l {{ font-size: 15px; color: {INK_2}; margin-bottom: 2px; }}
.kpi .s {{ font-size: 15px; color: {INK_3}; margin-top: 2px; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 16px 20px; align-items: flex-end; }}
.controls label {{ display: block; font-size: 15px; color: {INK_2}; margin-bottom: 4px; }}
.note {{ font-size: 15px; color: {INK_2}; line-height: 1.5; margin: 10px 2px 0; }}
details.about {{ margin-top: 12px; }}
details.about summary {{ cursor: pointer; font-weight: 600; color: {INK_2}; }}
details.about ul {{ margin: 8px 0 0; padding-left: 20px; color: {INK_2}; line-height: 1.55; font-size: 15px; }}
details.about li {{ font-style: italic; }}
.dash-dropdown, .dash-dropdown *, .dash-datepicker-input, .DateInput_input, .Select-value-label, .Select-input input {{ font-size: 15px !important; }}
.dash-options-list:not(.dash-checklist) .dash-options-list-option {{ display: flex !important; align-items: center; gap: 10px; width: 100%; box-sizing: border-box; padding: 8px 12px; margin: 0; cursor: pointer; font-size: 15px; }}
.dash-options-list:not(.dash-checklist) .dash-options-list-option:hover {{ background: {PAGE_BG}; }}
.dash-dropdown-content {{ max-height: 226px !important; }}
table.eval {{ border-collapse: collapse; width: 100%; font-size: 15px; display: block; overflow-x: auto; }}
table.eval th, table.eval td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid {BORDER}; }}
table.eval th {{ color: {INK_2}; font-weight: 600; }}
"""

# Interaction tips, shown as bullets in a chart's 'About' section. The legend tip only goes with charts that have a legend.
ZOOM_TIP = "Drag to zoom into part of the chart; double-click the chart to reset the view."
LEGEND_TIP = "Click a legend entry to hide that series; double-click it to isolate it."
MAP_TIP = "Drag to pan the map; scroll to zoom; double-click to reset the view."
CHART_TIPS = (ZOOM_TIP,)
LEGEND_CHART_TIPS = (ZOOM_TIP, LEGEND_TIP)
MAP_TIPS = (MAP_TIP,)


app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Florida Flood Claim Severity"
server = app.server
app.index_string = f"""<!DOCTYPE html>
<html>
<head>
{{%metas%}}
<title>{{%title%}}</title>
{{%favicon%}}
{{%css%}}
<style>{CSS}</style>
</head>
<body>
{{%app_entry%}}
<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>"""


def kpi(label, value, sub=None):
    return html.Div([html.Div(label, className="l"), html.Div(value, className="v")]
                    + ([html.Div(sub, className="s")] if sub else []), className="kpi")


def style_fig(fig, title=None, height=None, unified=True, center_title=False):
    title_pos = dict(x=0.5, xanchor="center") if center_title else dict(x=0)
    fig.update_layout(
        template="plotly_white", title=dict(text=title, font=dict(size=17, color=INK), **title_pos) if title else None,
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=INK_2, size=15), hoverlabel=dict(font_size=15),
        margin=dict(l=76, r=20, t=60 if title else 30, b=80), height=height,
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", yanchor="top", font=dict(color=INK_2)),
        hovermode="x unified" if unified else "closest",
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=BORDER, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=BORDER, zeroline=False, automargin=True)
    return fig


def log_dollar_axis(fig, axis="y", title=None, lo=None, hi=None):
    """Log-scale dollar axis with readable ticks ($1k, $3k, $10k...) instead of Plotly's crowded 2..9 minor labels."""
    ticks = [t for t in LOG_TICKS if (lo is None or t >= lo * 0.99) and (hi is None or t <= hi * 1.01)]
    kw = dict(type="log", tickvals=ticks, ticktext=[money(t) for t in ticks], minor=dict(showgrid=False))
    if title:
        kw["title"] = title
    (fig.update_yaxes if axis == "y" else fig.update_xaxes)(**kw)


def field(label, control):
    return html.Div([html.Label(label), control])


def header():
    return html.Div(
        [
            html.H1("Florida Flood Claim Severity", style={"margin": "0 0 4px", "fontSize": "28px", "letterSpacing": "-0.02em"}),
            html.Div("How much does a paid flood-insurance claim cost in Florida, and how has that changed? Real FEMA (NFIP) claims, "
                     f"{YEAR_MIN} to {YEAR_MAX}, with every payout in constant 2025 dollars.",
                     style={"color": INK_2, "marginBottom": "16px"}),
            html.Div(
                [
                    kpi("Paid claims", f"{N_CLAIMS:,}", f"Florida, {YEAR_MIN}-{YEAR_MAX}"),
                    kpi("Total paid", money(SUMMARY["total_paid"]), "2025 dollars"),
                    kpi("Median paid claim", money(SUMMARY["median_claim"]), "2025 dollars"),
                    kpi("Claims with a FEMA event label", pct(SUMMARY["event_label_pct"]),
                        f"{pct(SUMMARY['named_storm_label_pct'])} name a hurricane or tropical storm"),
                ],
                className="kpi-row",
            ),
            html.Details(
                [
                    html.Summary("About this data"),
                    html.Ul([
                        html.Li([html.Strong("Sources: "),
                                 f"claims are OpenFEMA NFIP Redacted Claims (v3), pulled {FEMA_PULL_DATE}. NFIP is the National Flood Insurance "
                                 "Program. Prices are CPI-U (Consumer Price Index for All Urban Consumers) from FRED, so a 1992 claim and a "
                                 "2024 claim are comparable."]),
                        html.Li([html.Strong("Paid claims only: "),
                                 "payout = building + contents + increased-cost-of-compliance. "
                                 f"About {CLOSED_WITHOUT_PAYMENT_PCT:.0f}% of claims were closed without payment and are excluded, so this "
                                 "describes the cost of a claim given that it was paid."]),
                        html.Li([html.Strong("Event labels: "),
                                 "event names are FEMA's own label for the claim (its floodEvent field). A claim without a label can still be "
                                 "storm-related."]),
                        html.Li([html.Strong("Patterns: "),
                                 "the figures describe associations in past claims. Payouts have grown much faster than inflation; possible reasons "
                                 "include storm severity, rebuilding costs, coverage limits, claims handling and the mix of insured buildings."]),
                        html.Li([html.Strong("Location: "),
                                 "FEMA deliberately blurs locations to 0.1 degree (about 7 miles), so a point can fall in the wrong county. "
                                 f"{REPORT['rows_dropped']['outside Florida bounds']} paid claims with coordinates outside Florida were dropped."]),
                        html.Li([html.Strong("Repeated rows: "),
                                 f"{SUMMARY['duplicates_kept']:,} rows are identical to another row on every field except id. They are kept, "
                                 "because several units of one building can file claims with the same values."]),
                    ]),
                ],
                className="about",
            ),
        ],
        className="card", style={"marginBottom": "16px"},
    )


TAB_STYLE = {"padding": "10px 14px", "fontSize": "15px", "border": f"1px solid {BORDER}", "backgroundColor": PAGE_BG,
             "color": INK_2, "fontWeight": "500"}
TAB_SELECTED = {**TAB_STYLE, "backgroundColor": SURFACE, "color": INK, "fontWeight": "650", "borderTop": f"2px solid {BLUE}"}


def filter_bar():
    def multi(id_, options, placeholder, width, searchable=True):
        return dcc.Dropdown(id=id_, options=options, multi=True, placeholder=placeholder, searchable=searchable,
                            style={"width": width})

    return html.Div(
        [
            html.Div([
                field("Loss Years", html.Div(dcc.RangeSlider(
                    id="f-years", min=YEAR_MIN, max=YEAR_MAX, step=1, value=[YEAR_MIN, YEAR_MAX], allowCross=False,
                    marks={y: str(y) for y in range(1980, YEAR_MAX, 10)}), style={"width": "300px", "paddingBottom": "18px"})),
                field("County", multi("f-county", COUNTY_OPTIONS, "All counties", "240px")),
                field("Occupancy", multi("f-occ", OCC_OPTIONS, "All types", "200px", searchable=False)),
                field("Flood Zone", multi("f-flood", FLOOD_OPTIONS, "All zones", "200px", searchable=False)),
            ], className="controls"),
            html.Div(id="f-count", className="note", style={"marginTop": "8px"}),
        ],
        id="filters", className="card", style={"marginBottom": "12px"},
    )


app.layout = html.Div(
    className="wrap",
    children=[
        header(),
        filter_bar(),
        dcc.Tabs(
            id="tabs", value="tab1", colors={"border": BORDER, "primary": BLUE, "background": PAGE_BG},
            children=[dcc.Tab(label=lbl, value=val, style=TAB_STYLE, selected_style=TAB_SELECTED) for val, lbl in
                      [("tab1", "Overview"), ("tab2", "Storms"), ("tab3", "Factors")]],
        ),
        html.Div(id="tab-content", style={"marginTop": "16px"}),
    ],
)


# ------------------------------------------------------------ tab layouts ----
MEASURES = {"claims": "Number of claims", "total": "Total paid (2025 $)", "median": "Median claim (2025 $)"}


def tab1_layout():
    return html.Div([
        html.Div([
            field("Measure", dcc.Dropdown(id="t1-measure", options=[{"label": v, "value": k} for k, v in MEASURES.items()],
                                          value="total", clearable=False, searchable=False, style={"width": "240px"})),
        ], className="controls card"),
        html.Div(id="t1-kpis", className="kpi-row", style={"margin": "12px 0"}),
        html.Div([dcc.Graph(id="t1-year-chart", config={"displayModeBar": False}), html.Div(id="t1-year-note")],
                 className="card"),
        html.Div(id="t1-decades", className="card", style={"marginTop": "12px"}),
        html.Div([dcc.Graph(id="t1-map", config={"displayModeBar": False}),
                  about([f"Counties with fewer than {MIN_CLAIMS} claims are not shown."], "About this map", tips=MAP_TIPS)],
                 className="card", style={"marginTop": "12px"}),
    ])


def tab2_layout():
    return html.Div([
        html.Div([
            field("Rank events by", dcc.Dropdown(id="t2-metric", options=[{"label": v, "value": k} for k, v in MEASURES.items()],
                                                 value="total", clearable=False, searchable=False, style={"width": "240px"})),
            field("Show Top", dcc.Dropdown(id="t2-n", options=[{"label": str(n), "value": n} for n in (10, 15, 25)],
                                           value=15, clearable=False, searchable=False, style={"width": "110px"})),
        ], className="controls card"),
        html.Div(id="t2-coverage", className="note"),
        html.Div([dcc.Graph(id="t2-chart", config={"displayModeBar": False}), about([])], className="card", style={"marginTop": "12px"}),
        html.Div(id="t2-table", style={"marginTop": "12px"}),
        html.Div(id="t2-other", style={"marginTop": "12px"}),
    ])


def tab3_layout():
    return html.Div([
        html.Div([field("Compare payouts by", dcc.Dropdown(id="t3-factor", options=[{"label": k, "value": k} for k in FACTORS],
                                                           value="Flood zone", clearable=False, searchable=False,
                                                           style={"width": "300px"}))],
                 className="controls card"),
        html.Div([dcc.Graph(id="t3-chart", config={"displayModeBar": False}), html.Div(id="t3-note")],
                 className="card", style={"marginTop": "12px"}),
        html.Div(id="t3-table", className="card", style={"marginTop": "12px"}),
        html.Div(id="t3-extra", style={"marginTop": "12px"}),
    ])


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    return {"tab1": tab1_layout, "tab2": tab2_layout, "tab3": tab3_layout}[tab]()


# -------------------------------------------------------------- callbacks ----
def _measure_series(frame_group, measure):
    if measure == "claims":
        return frame_group.size()
    col = frame_group["amountPaid_real"]
    return col.sum() if measure == "total" else col.median()


FILTERS = [Input("f-years", "value"), Input("f-county", "value"), Input("f-occ", "value"), Input("f-flood", "value")]


def empty_fig(message, height=300):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16, color=INK_2), xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_fig(fig, height=height)


NO_CLAIMS = "No claims match these filters"


def about(items, title="About this chart", tips=CHART_TIPS):
    """A collapsed 'About ...' section, styled like 'About this data': the caveats first, then the chart's interaction tips."""
    entries = [i for i in items if i] + list(tips)
    return html.Details([html.Summary(title), html.Ul([html.Li(e) for e in entries])], className="about") if entries else None


def count_cell(n):
    """A claim count; one under MIN_CLAIMS gets an asterisk and a tooltip."""
    n = int(n)
    return f"{n:,}" if n >= MIN_CLAIMS or n == 0 else (f"{n:,}*", f"Fewer than {MIN_CLAIMS} claims")


def table(headers, rows):
    """A plain results table. A cell may be (text, tooltip)."""
    return html.Table([
        html.Thead(html.Tr([html.Th(h) for h in headers])),
        html.Tbody([html.Tr([html.Td(v[0], title=v[1]) if isinstance(v, tuple) else html.Td(v) for v in row]) for row in rows]),
    ], className="eval")


@app.callback(Output("f-count", "children"), *FILTERS)
def update_filter_count(years, counties, occs, floods):
    sub = filtered(years, counties, occs, floods)
    if len(sub) == N_CLAIMS:
        return f"Showing all {N_CLAIMS:,} paid claims."
    return (f"Showing {len(sub):,} of {N_CLAIMS:,} paid claims ({100 * len(sub) / N_CLAIMS:.1f}%), "
            f"{money(sub['amountPaid_real'].sum())} paid in 2025 dollars.")


def update_tab1(years, counties, occs, floods, measure):
    sub = filtered(years, counties, occs, floods)
    if sub.empty:
        return [kpi("Claims", "0")], empty_fig(NO_CLAIMS), html.Div(), html.Div(NO_CLAIMS, className="note"), empty_fig(NO_CLAIMS, 520)
    labeled = 100 * sub["event"].notna().mean()
    kpis = [kpi("Claims", f"{len(sub):,}"), kpi("Total paid", money(sub["amountPaid_real"].sum()), "2025 dollars"),
            kpi("Median claim", money(sub["amountPaid_real"].median()), "2025 dollars"),
            kpi("With a FEMA event label", pct(labeled), "of the claims shown")]

    by_year = sub.groupby("yearOfLoss")
    y = _measure_series(by_year, measure)
    n = by_year.size().reindex(y.index)
    top = sub[sub["event"].notna()].groupby(["yearOfLoss", "event"], observed=True).size().reset_index(name="n")
    top_event = top.sort_values("n").groupby("yearOfLoss").tail(1).set_index("yearOfLoss")["event"].astype(str).to_dict()
    tops = [top_event.get(int(i), "no event label") for i in y.index]
    small = n < MIN_CLAIMS
    fig = go.Figure(go.Bar(x=y.index, y=y.values, marker_color=[BLUE_LIGHT if s else BLUE for s in small],
                           customdata=np.c_[[f"{int(v):,}" + ("*" if v < MIN_CLAIMS else "") for v in n.values], tops],
                           hovertemplate="%{x}: %{y:,.0f}<br>Claims: %{customdata[0]}<br>Most claims from: %{customdata[1]}<extra></extra>"))
    style_fig(fig, f"{MEASURES[measure]} by year of loss", 380, unified=False)
    fig.update_yaxes(title=MEASURES[measure], tickprefix="" if measure == "claims" else "$")
    note = about([f"Pale bars are years with fewer than {MIN_CLAIMS} claims." if small.any() else None])

    decade_rows = []
    for d, g in sub.groupby("decade"):
        decade_rows.append([years_label(g["yearOfLoss"].unique()), count_cell(len(g)), money(g["amountPaid_real"].median()),
                            money(g["amountPaid_real"].sum())])
    decades = html.Div([html.H3("By decade of loss", style={"margin": "0 0 8px"}),
                        table(["Loss years", "Claims", "Median claim (2025 $)", "Total paid (2025 $)"], decade_rows)])

    g = sub.groupby("countyName", observed=True)
    counts = g.size()
    keep = counts[counts >= MIN_CLAIMS].index
    if len(keep) == 0:
        return kpis, fig, note, decades, empty_fig(f"No county has {MIN_CLAIMS} or more claims", 520)
    size = _measure_series(g, measure)[keep]
    med = g["amountPaid_real"].median()[keep]
    counts = counts[keep]
    ci = pd.DataFrame(COUNTY_INFO).T.loc[size.index]
    scale = 46 / np.sqrt(max(size.max(), 1))
    m = go.Figure(go.Scattermap(
        lat=ci["lat"], lon=ci["lon"], mode="markers", text=size.index,
        marker=dict(size=np.maximum(np.sqrt(size.values) * scale, 5), color=med.values, colorscale="YlOrRd", showscale=True,
                    colorbar=dict(title="Median<br>claim", tickprefix="$"), opacity=0.75),
        customdata=np.c_[size.values, med.values, counts.values],
        hovertemplate="%{text}<br>" + MEASURES[measure] + ": %{customdata[0]:,.0f}<br>Median claim: %{customdata[1]:$,.0f}"
                      "<br>Claims: %{customdata[2]:,.0f}<extra></extra>"))
    m.update_layout(map=dict(style="open-street-map", center=dict(lat=27.9, lon=-83.2), zoom=5.4), height=520,
                    margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=SURFACE)
    return kpis, fig, note, decades, m


app.callback(Output("t1-kpis", "children"), Output("t1-year-chart", "figure"), Output("t1-year-note", "children"),
             Output("t1-decades", "children"), Output("t1-map", "figure"),
             *FILTERS, Input("t1-measure", "value"))(update_tab1)


def storm_label(name, year):
    """A storm's name with its loss year, so it is easy to place ('Hurricane Ian (2022)'); a label that already ends in a year stays as is."""
    name = str(name)
    return name if name.endswith(")") and name[-5:-1].isdigit() else f"{name} ({int(year)})"


def event_stats(sub):
    stats = sub.groupby("event", observed=True).agg(
        claims=("amountPaid_real", "size"), total=("amountPaid_real", "sum"), median=("amountPaid_real", "median"),
        first=("yearOfLoss", "min"), last=("yearOfLoss", "max"))
    return stats[stats["claims"] > 0]


def other_events_card(sub):
    """The other FEMA-labeled events; shown whatever the named storms look like. A claim count under MIN_CLAIMS gets an asterisk."""
    other = event_stats(sub[sub["event_group"] == "Other FEMA-labeled event"])
    if other.empty:
        return html.Div()
    other = other.sort_values("claims", ascending=False)
    rows = []
    for i, r in other.iterrows():
        n = int(r["claims"])
        claims_cell = f"{n:,}" if n >= MIN_CLAIMS else (f"{n:,}*", f"Fewer than {MIN_CLAIMS} claims")
        rows.append([str(i), years_label([r["first"], r["last"]]), claims_cell, money(r["total"]), money(r["median"])])
    return html.Div([
        html.H3("Other FEMA-labeled events", style={"margin": "0 0 8px"}),
        table(["Event", "Loss years", "Claims", "Total paid (2025 $)", "Median claim (2025 $)"], rows),
    ], className="card")


def update_tab2(years, counties, occs, floods, metric, n):
    sub = filtered(years, counties, occs, floods)
    if sub.empty:
        return "", empty_fig(NO_CLAIMS), html.Div(), html.Div()
    coverage = (f"Of the {len(sub):,} claims shown, {pct(100 * sub['event'].notna().mean())} carry a FEMA event label and "
                f"{pct(100 * (sub['event_group'] == 'Named storm').mean())} name a hurricane or tropical storm. Claims with no label are not "
                "ranked below.")

    named = event_stats(sub[sub["event_group"] == "Named storm"])
    big = named[named["claims"] >= MIN_CLAIMS]
    hidden = len(named) - len(big)
    if big.empty:
        return coverage, empty_fig(f"No named storm reaches {MIN_CLAIMS} claims under these filters"), html.Div(), other_events_card(sub)
    top = big.sort_values(metric, ascending=False).head(n)
    names = [storm_label(i, r["first"]) for i, r in top.iterrows()]
    fig = go.Figure(go.Bar(x=top[metric][::-1], y=names[::-1], orientation="h", marker_color=ORANGE,
                           customdata=np.c_[top["claims"][::-1], top["total"][::-1], top["median"][::-1]],
                           hovertemplate="%{y}<br>Claims: %{customdata[0]:,.0f}<br>Total paid: %{customdata[1]:$,.0f}"
                                         "<br>Median claim: %{customdata[2]:$,.0f}<extra></extra>"))
    style_fig(fig, f"Top {len(top)} named storms by {MEASURES[metric].lower()}", max(360, 26 * len(top) + 120), unified=False)
    fig.update_xaxes(title=MEASURES[metric], tickprefix="" if metric == "claims" else "$")
    tbl = html.Div([
        table(["Storm", "Claims", "Total paid (2025 $)", "Median claim (2025 $)"],
              [[name, f"{int(r['claims']):,}", money(r["total"]), money(r["median"])] for name, (_, r) in zip(names, top.iterrows())]),
        about([f"Storms with fewer than {MIN_CLAIMS} claims under these filters ({hidden}) are not shown."] if hidden else [],
              "About this table", tips=()),
    ], className="card")

    return coverage, fig, tbl, other_events_card(sub)


app.callback(Output("t2-coverage", "children"), Output("t2-chart", "figure"), Output("t2-table", "children"),
             Output("t2-other", "children"), *FILTERS, Input("t2-metric", "value"), Input("t2-n", "value"))(update_tab2)


def age_by_decade(sub):
    """Median payout by building age band within each decade of loss (cells with too few claims are left out)."""
    g = sub.groupby(["decade", "age_band"], observed=True)["amountPaid_real"]
    stats = pd.DataFrame({"n": g.size(), "median": g.median()})
    stats = stats[stats["n"] > 0]
    labels = {int(d): years_label(sub.loc[sub["decade"] == d, "yearOfLoss"].unique()) for d in sub["decade"].unique()}
    bands = [b for b in FACTOR_COLUMNS["Building age"][1] if b in set(stats.index.get_level_values("age_band"))]
    fig = go.Figure()
    for k, d in enumerate(sorted(labels)):
        rows = [(b, stats.loc[(d, b)]) for b in bands if (d, b) in stats.index and stats.loc[(d, b), "n"] >= MIN_CLAIMS]
        if not rows:
            continue
        fig.add_trace(go.Scatter(x=[b for b, _ in rows], y=[r["median"] for _, r in rows], mode="lines+markers", name=labels[d],
                                 line=dict(color=SERIES[k % len(SERIES)]), customdata=[int(r["n"]) for _, r in rows],
                                 hovertemplate="%{x}: %{y:$,.0f}<br>Claims: %{customdata:,}<extra>" + labels[d] + "</extra>"))
    style_fig(fig, "Median payout by building age, within each decade of loss", 420, unified=False, center_title=True)
    log_dollar_axis(fig, "y", "Median payout (2025 $, log scale)")
    fig.update_xaxes(title="Building age at time of loss")
    counts = [[labels[d]] + [count_cell(stats.loc[(d, b), "n"]) if (d, b) in stats.index else "0" for b in bands] for d in sorted(labels)]
    return fig, table(["Loss years"] + bands, counts)


def update_tab3(years, counties, occs, floods, factor):
    col, order, factor_note = FACTORS[factor]
    sub = filtered(years, counties, occs, floods)
    note = about([factor_note])
    if sub.empty:
        return empty_fig(NO_CLAIMS, 460), html.Div(), note, html.Div()
    g = sub.groupby(col, observed=True)["amountPaid_real"]
    st = g.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    st.columns = ["q10", "q25", "q50", "q75", "q90"]
    st["n"] = g.size()
    st["mean"] = g.mean()
    st = st[st["n"] > 0].reindex([o for o in order if o in st.index])
    total_n = st["n"].sum()
    big = st[st["n"] >= MIN_CLAIMS]
    if big.empty:
        return empty_fig(f"No group has {MIN_CLAIMS} or more claims", 460), html.Div(), note, html.Div()
    many = len(big) > 8  # with many groups the full "12,345 claims" labels run into each other, so counts are shortened
    labels = [f"{AXIS_SHORT_LABELS.get(str(i), str(i))}<br>" + (compact_count(r["n"]) if many else f"{int(r['n']):,} claims")
              for i, r in big.iterrows()]
    # Plotly's box trace ignores hovertemplate for its auto-generated per-stat labels (always shows every fence/
    # min/max it was given), so the box's own hover is disabled and an invisible full-height bar supplies the
    # tooltip instead, with exactly the fields wanted.
    fig = go.Figure([
        go.Box(x=labels, q1=big["q25"], median=big["q50"], q3=big["q75"], lowerfence=big["q10"], upperfence=big["q90"],
               marker_color=BLUE, line=dict(color=BLUE), fillcolor="rgba(42,120,214,0.18)", hoverinfo="skip", showlegend=False),
        go.Bar(x=labels, y=(big["q90"] - big["q10"]).values, base=big["q10"].values,
               marker=dict(color="rgba(0,0,0,0)", line=dict(width=0)), showlegend=False,
               customdata=np.stack([big["q50"], big["q25"], big["q75"], big["n"]], axis=-1),
               hovertemplate="%{x}<br>Median: %{customdata[0]:$,.2f}<br>25th percentile: %{customdata[1]:$,.2f}<br>"
                             "75th percentile: %{customdata[2]:$,.2f}<extra></extra>"),
    ])
    style_fig(fig, f"Payout by {factor.title()}", 460, unified=False, center_title=True)
    log_dollar_axis(fig, "y", "Payout (2025 $, log scale)")
    fig.update_xaxes(tickangle=0, tickfont=dict(size=12))
    if many:
        fig.update_xaxes(title="Claims in each group (under its label)")
    rows = [[str(i), count_cell(r["n"]), f"{100 * r['n'] / total_n:.1f}%", money(r["q50"]), money(r["mean"]),
             f"{money(r['q25'])} to {money(r['q75'])}"] for i, r in st.iterrows()]
    tbl = table([factor.title(), "Claims", "% of Claims", "Median Payout", "Mean Payout", "Middle 50% of Claims"], rows)
    extra = html.Div()
    if factor == "Building age":
        extra_fig, counts = age_by_decade(sub)
        extra = html.Div([dcc.Graph(figure=extra_fig, config={"displayModeBar": False}),
                          about([f"A point is drawn only where at least {MIN_CLAIMS} claims support it."], tips=LEGEND_CHART_TIPS),
                          html.H3("Claims behind each line", style={"margin": "12px 0 8px"}), counts],
                         className="card")
    return fig, tbl, note, extra


app.callback(Output("t3-chart", "figure"), Output("t3-table", "children"), Output("t3-note", "children"),
             Output("t3-extra", "children"), *FILTERS, Input("t3-factor", "value"))(update_tab3)


_log("app ready")

if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 8058)))
