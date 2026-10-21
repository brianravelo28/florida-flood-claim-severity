"""
Florida Flood Claim Severity - interactive Dash app (real FEMA NFIP claims + NOAA HURDAT2 storm tracks).

A shared filter bar above the tabs (Loss Years, County, Occupancy, Flood Zone, Storm Categories) drives tabs 1-4; the
Claim Estimator swaps it for its own control bar (see toggle_filters).

Tabs:
1. Overview          claims by year and county (payouts in constant 2025 dollars)
2. Storms            which storms drove the most claims and dollars
3. Risk Factors      how payouts differ by storm strength, flood zone, occupancy, elevation, age, distance
4. Model             honest out-of-sample evaluation of the LightGBM severity model, plus two building-age tests
5. Claim Estimator   what-if: typical payout for a hypothetical claim (uses the loss-year-control model)

Run locally: python app.py   (port from $PORT, default 8058).  Hosted: gunicorn wsgi:application
Data: deploy_data/ (built by build_deploy_data.py).
"""
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

from constants import FACTOR_COLUMNS, FLOOD_ORDER, OCC_ORDER, STORM_ORDER

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
SERIES = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4"]

# ---------------------------------------------------------------- data ----
# One compact row per paid claim (all paid claims; categoricals + float32, ~10 MB in memory). Every tab filters and
# aggregates it live, so nothing is a precomputed summary of a subset.
claims = pd.read_parquet(DEPLOY / "claims.parquet")
claims["matched"] = claims["storm_label"].notna()
claims["log_actual"] = np.log1p(claims["amountPaid_real"]).astype("float32")
claims["log_pred"] = np.log1p(claims["prediction_amount"]).astype("float32")
AGG = json.loads((DEPLOY / "aggregates.json").read_text())
METRICS = json.loads((DEPLOY / "model_metrics.json").read_text())
REPORT = json.loads((DEPLOY / "build_report.json").read_text())
_abl = DEPLOY / "ablation_year_control.json"
ABLATION = json.loads(_abl.read_text()) if _abl.exists() else None
_abl2 = DEPLOY / "ablation_postfirm.json"
ABLATION_FIRM = json.loads(_abl2.read_text()) if _abl2.exists() else None
# The claim estimator uses the loss-year-control variant (train_year_model.py) with its own out-of-sample calibration.
booster = lgb.Booster(model_file=str(DEPLOY / "model_year.txt"))
YEAR_MODEL = json.loads((DEPLOY / "year_control_model.json").read_text())
FEATURES = YEAR_MODEL["features"]
_log("data loaded")

YEAR_MIN, YEAR_MAX, N_CLAIMS = AGG["year_min"], AGG["year_max"], AGG["n_claims"]
RESID_Q10, RESID_Q50, RESID_Q90 = (AGG["residual_quantiles"][k] for k in ("q10", "q50", "q90"))
UNDERPREDICT = float(np.exp(RESID_Q50))  # actual median claim / predicted, on years the model never saw
EST_Q10, EST_Q50, EST_Q90 = (YEAR_MODEL["residual_quantiles"][k] for k in ("q10", "q50", "q90"))
EST_UNDER = float(np.exp(EST_Q50))  # the same correction, for the year-control model behind the estimator
CLAIMS_BY_YEAR = {int(k): v for k, v in YEAR_MODEL["claims_by_year"].items()}
YEAR_DEFAULT = max(y for y, n in CLAIMS_BY_YEAR.items() if n >= 1000)  # latest year with a solid sample
LOG_TICKS = [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6]

# Short forms for chart x-axis ticks only; full names stay in the dropdown, table, and hover.
AXIS_SHORT_LABELS = {
    "Tropical Depression": "Trop. Dep.", "Tropical Storm": "Trop. Storm",
    "X / B / C (moderate-minimal)": "X / B / C",
    "A / AH / AO / AR / A99 (no base elevation, shallow, special)": "A / AH / AO...",
    "AE / A1-A30 (base elevation known)": "AE / A1-A30", "V / VE (velocity, coastal)": "V / VE",
    "Condo association": "Condo assoc.", "Unit in multi-unit building": "Multi-unit", "Non-residential": "Non-resid.",
    "No storm match": "No match",
}

FACTOR_NOTES = {
    "Storm category": "Strongest category of the nearest storm (HURDAT2 wind at the closest track point within 150 miles and 7 days). The order is not strictly increasing because a few catastrophic storms dominate individual categories; categories with no claims are omitted.",
    "Flood zone": "FEMA flood zone the property was rated in, grouped by FEMA's zone definitions.",
    "Occupancy type": "FEMA's legacy (1-4, 6) and newer (11-19) occupancy codes describe the same building types, so they are grouped together.",
    "Elevated building": "FEMA's elevatedBuildingIndicator.",
    "Building age": "Age at time of loss. About 4% of buildings have an unusable construction date (98% of them FEMA's 1492 placeholder, and every placeholder is flagged pre-FIRM, so they are probably old), so here they're bucketed using the age of known-date pre-FIRM buildings from the same loss year, rather than one flat age for all of them (this affects only this chart, not the model). Caution: old-building claims are mostly recent claims (median age at loss was about 30 in the 1980s and 50 in the 2020s), and recent claims are larger even after inflation adjustment, so age and loss year are entangled.",
    "Distance from storm track": "Distance from the claim's (blurred) location to the nearest storm track point.",
}
FACTORS = {name: (col, order, FACTOR_NOTES[name]) for name, (col, order) in FACTOR_COLUMNS.items()}

COUNTY_INFO = AGG["counties"]
COUNTIES = sorted(COUNTY_INFO)
OCC_CODE, FLOOD_CODE = AGG["occupancy_code"], AGG["flood_code"]


def _present(col, order=None):
    """Category labels that actually occur in the data (a category with no claims isn't a useful filter option)."""
    have = set(claims[col].dropna().unique())
    return [o for o in (order or sorted(have)) if o in have]


COUNTY_OPTIONS = _present("countyName")
OCC_OPTIONS = _present("occupancy_group", OCC_ORDER)
FLOOD_OPTIONS = _present("flood_zone_group", FLOOD_ORDER)
STORM_OPTIONS = _present("storm_category", STORM_ORDER)


@lru_cache(maxsize=4)
def _subset(years, counties, occs, floods, storms):
    m = claims["yearOfLoss"].between(years[0], years[1]).to_numpy()
    for col, sel in (("countyName", counties), ("occupancy_group", occs), ("flood_zone_group", floods),
                     ("storm_category", storms)):
        if sel:
            m = m & claims[col].isin(sel).to_numpy()
    return claims[m]


def filtered(years, counties, occs, floods, storms):
    """The claims matching the global filter bar (an empty multi-select means 'all')."""
    return _subset(tuple(years or (YEAR_MIN, YEAR_MAX)), tuple(counties or ()), tuple(occs or ()), tuple(floods or ()),
                   tuple(storms or ()))


_log("precompute done")

FEATURE_NAMES = {
    "building_age_years": "Building age", "building_age_missing": "Building age unknown", "is_elevated": "Elevated building",
    "flood_zone_encoded": "Flood zone", "occupancy_group_encoded": "Occupancy type", "distance_bin": "Distance band to storm",
    "storm_category_encoded": "Storm category", "historical_storm_freq": "Storms nearby, prior 40 yrs",
    "latitude": "Latitude", "longitude": "Longitude", "distance_from_track_mi": "Distance to storm track",
    "storm_wind_speed_kt": "Storm wind speed", "days_to_storm": "Days from loss to storm",
}


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


def wind_category(w):
    if w is None or np.isnan(w) or w <= 0:
        return -1
    return int(np.select([w < 39, w < 74, w < 96, w < 111, w < 130, w < 157], [0, 1, 2, 3, 4, 5], default=6))


def distance_bin(dist):
    if dist is None or np.isnan(dist):
        return -1
    return 4 if dist <= 25 else 3 if dist <= 50 else 2 if dist <= 100 else 1


def _feature_row(county, occupancy, flood, age, elevated, wind, dist, year):
    storm = wind is not None and wind > 0
    info = COUNTY_INFO[county]
    return {
        "yearOfLoss": year, "building_age_years": age, "building_age_missing": 0, "is_elevated": int(elevated),
        "flood_zone_encoded": FLOOD_CODE[flood], "occupancy_group_encoded": OCC_CODE[occupancy],
        "distance_bin": distance_bin(dist) if storm else -1, "storm_category_encoded": wind_category(wind),
        "historical_storm_freq": info["freq"], "latitude": info["lat"], "longitude": info["lon"],
        "distance_from_track_mi": dist if storm else np.nan, "storm_wind_speed_kt": wind if storm else np.nan,
        "days_to_storm": 0.0 if storm else np.nan,
    }


def predict_batch(scenarios):
    """Raw model estimates (constant 2025 dollars) for many hypothetical claims in a single model call."""
    x = pd.DataFrame([_feature_row(**s) for s in scenarios])[FEATURES]
    return np.expm1(booster.predict(x, num_threads=1))


def predict_payout(county, occupancy, flood, age, elevated, wind, dist, year):
    """Raw model estimate, in constant 2025 dollars, for one hypothetical claim with a loss in `year`."""
    return float(predict_batch([dict(county=county, occupancy=occupancy, flood=flood, age=age, elevated=elevated,
                                     wind=wind, dist=dist, year=year)])[0])


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
.dash-dropdown, .dash-dropdown *, .dash-datepicker-input, .DateInput_input, .Select-value-label, .Select-input input {{ font-size: 15px !important; }}
.dash-options-list-option-checkbox {{ width: 18px; height: 18px; margin: 0; cursor: pointer; }}
.dash-checklist .dash-options-list-option {{ display: flex !important; flex-direction: column-reverse; align-items: center; gap: 6px; cursor: pointer; font-size: 15px; padding: 0; }}
.dash-options-list:not(.dash-checklist) .dash-options-list-option {{ display: flex !important; align-items: center; gap: 10px; width: 100%; box-sizing: border-box; padding: 8px 12px; margin: 0; cursor: pointer; font-size: 15px; }}
.dash-options-list:not(.dash-checklist) .dash-options-list-option:hover {{ background: {PAGE_BG}; }}
.dash-dropdown-content {{ max-height: 226px !important; }}
table.eval {{ border-collapse: collapse; width: 100%; font-size: 15px; display: block; overflow-x: auto; }}
table.eval th, table.eval td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid {BORDER}; }}
table.eval th {{ color: {INK_2}; font-weight: 600; }}
.chart-hint {{ margin: 4px 2px 0; color: {INK_3}; font-size: 15px; font-style: italic; }}
"""
FONT_STACK = '-apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'

CHART_HINT = (
    "Drag to zoom into part of the chart; double-click the chart to reset the view. "
    "Click a legend entry to hide that series; double-click it to isolate it."
)


MAP_HINT = "Drag to pan the map; scroll to zoom; double-click to reset the view."


def chart_hint(text=CHART_HINT):
    return html.P(text, className="chart-hint")

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
    m = METRICS
    h, base = m["headline_year_grouped_cv"], m["baseline_year_grouped_cv_predict_train_mean"]
    return html.Div(
        [
            html.H1("Florida Flood Claim Severity", style={"margin": "0 0 4px", "fontSize": "28px", "letterSpacing": "-0.02em"}),
            html.Div("How much does a paid NFIP flood claim cost? Real FEMA claims joined to NOAA hurricane tracks, "
                     "with every payout in constant 2025 dollars.", style={"color": INK_2, "marginBottom": "16px"}),
            html.Div(
                [
                    kpi("Paid claims analyzed", f"{N_CLAIMS:,}", f"Florida, {YEAR_MIN}-{YEAR_MAX}"),
                    kpi("Matched to a storm", pct(REPORT["storm_matched_pct"]), "within 150 mi and 7 days of the loss"),
                    kpi("Model R² on unseen years", f"{h['r2_log']:.2f}", f"predicting the average scores {base['r2_log']:.2f}"),
                    kpi("Typical prediction error", f"{h['median_abs_pct_error']:.0f}%", "median absolute % error"),
                ],
                className="kpi-row",
            ),
            html.Details(
                [
                    html.Summary("About this data and model"),
                    html.Ul([
                        html.Li("Claims: OpenFEMA NFIP Redacted Claims (v3), pulled 2026-10-20. Storms: NOAA HURDAT2 Atlantic best-track, interpolated to hourly positions. Prices: CPI-U from FRED, so a 1992 claim and a 2024 claim are comparable."),
                        html.Li("Only claims that were actually paid are included (payout = building + contents + increased-cost-of-compliance). About 30% of claims were closed without payment and are excluded, so this describes severity given a paid claim."),
                        html.Li("A claim is matched to the nearest storm track point within 150 miles and 7 days of the loss date. Checked against FEMA's own storm labels: 95.6% of claims FEMA calls a named hurricane or tropical storm matched the same-named storm."),
                        html.Li(f"The model is LightGBM on log payout. Its score comes from 5-fold cross-validation grouped by loss year, so each claim is predicted by a model that never saw that year: R² {h['r2_log']:.2f}. This is a modest signal, not a precise predictor; storm-to-storm variation is large."),
                        html.Li("Caveat: building age is the model's strongest signal, but it is entangled with loss year (older buildings appear mostly in recent, larger claims), so treat it as a correlation, not a cause. A loss-year test on the Model tab shows the calendar year alone recovers essentially all of its predictive value."),
                        html.Li("Locations are deliberately blurred by FEMA to 0.1 degree (about 7 miles), and 105 paid claims with coordinates outside Florida (many in New York) were dropped."),
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
                field("Storm Categories", multi("f-storm", STORM_OPTIONS, "All categories", "200px", searchable=False)),
            ], className="controls"),
            html.Div(id="f-count", className="note", style={"marginTop": "8px"}),
        ],
        id="filters", className="card", style={"marginBottom": "12px"},
    )


def estimator_bar():
    """The Claim Estimator's controls, in the same position and style as the global filter_bar() above the tabs
    (shown only on that tab; toggle_filters() swaps the two in and out together)."""
    return html.Div(
        html.Div([
            field("County", dcc.Dropdown(id="t5-county", options=COUNTIES, value="Lee", clearable=False, style={"width": "200px"})),
            field("Occupancy", dcc.Dropdown(id="t5-occ", options=[o for o in OCC_ORDER if o != "Unknown"], value="Single-family",
                                            clearable=False, searchable=False, style={"width": "260px"})),
            field("Flood Zone", dcc.Dropdown(id="t5-flood", options=[f for f in FLOOD_ORDER if f in FLOOD_CODE],
                                             value=FLOOD_ORDER[2], clearable=False, searchable=False, style={"width": "330px"})),
            field("Loss Year", html.Div(dcc.Slider(id="t5-year", min=YEAR_MIN, max=YEAR_MAX, step=1, value=YEAR_DEFAULT,
                                                   marks={y: str(y) for y in range(1980, YEAR_MAX, 10)}),
                                        style={"width": "300px", "paddingBottom": "18px"})),
            field("Building Age (years)", html.Div(dcc.Slider(id="t5-age", min=0, max=100, step=1, value=30,
                                                             marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100"}),
                                                   style={"width": "260px", "paddingBottom": "18px"})),
            html.Div(dcc.Checklist(id="t5-elev", options=[{"label": "Elevated?", "value": 1}], value=[]),
                    style={"paddingBottom": "18px"}),
            field("Storm Wind Speed (kt, 0 = no storm)", html.Div(dcc.Slider(id="t5-wind", min=0, max=150, step=5, value=90,
                                                                            marks={0: "0", 39: "TS", 74: "Cat 1", 111: "Cat 3", 150: "150"}),
                                                                  style={"width": "300px", "paddingBottom": "18px"})),
            field("Distance From Storm Track (mi)", html.Div(dcc.Slider(id="t5-dist", min=0, max=150, step=5, value=40,
                                                                        marks={0: "0", 50: "50", 100: "100", 150: "150"}),
                                                              style={"width": "260px", "paddingBottom": "18px"})),
        ], className="controls"),
        id="estimator-bar", className="card", style={"marginBottom": "12px", "display": "none"},
    )


app.layout = html.Div(
    className="wrap",
    children=[
        header(),
        filter_bar(),
        estimator_bar(),
        dcc.Tabs(
            id="tabs", value="tab1", colors={"border": BORDER, "primary": BLUE, "background": PAGE_BG},
            children=[dcc.Tab(label=lbl, value=val, style=TAB_STYLE, selected_style=TAB_SELECTED) for val, lbl in
                      [("tab1", "Overview"), ("tab2", "Storms"), ("tab3", "Risk Factors"), ("tab4", "Model"),
                       ("tab5", "Claim Estimator")]],
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
                                          value="total", clearable=False, style={"width": "240px"})),
        ], className="controls card"),
        html.Div(id="t1-kpis", className="kpi-row", style={"margin": "12px 0"}),
        html.Div([dcc.Graph(id="t1-year-chart", config={"displayModeBar": False}), chart_hint()], className="card"),
        html.Div("Storm years stand out: hover a bar to see the storm behind most of that year's claims.", className="note"),
        html.Div([dcc.Graph(id="t1-map", config={"displayModeBar": False}), chart_hint(MAP_HINT)], className="card", style={"marginTop": "12px"}),
        html.Div("Map bubbles are sized by the selected measure and colored by median claim.", className="note"),
    ])


def tab2_layout():
    return html.Div([
        html.Div([
            field("Rank storms by", dcc.Dropdown(id="t2-metric", options=[{"label": v, "value": k} for k, v in MEASURES.items()],
                                                 value="total", clearable=False, style={"width": "240px"})),
            field("Show Top", dcc.Dropdown(id="t2-n", options=[{"label": str(n), "value": n} for n in (10, 15, 25)],
                                           value=15, clearable=False, searchable=False, style={"width": "110px"})),
        ], className="controls card"),
        html.Div([dcc.Graph(id="t2-chart", config={"displayModeBar": False}), chart_hint()], className="card", style={"marginTop": "12px"}),
        html.Div(id="t2-table", className="card", style={"marginTop": "12px"}),
        html.Div("Each claim is assigned the nearest storm within 150 miles and 7 days; claims with no storm nearby are not counted here. "
                 "Older, unnamed storms appear as 'Unnamed' in the dataset.", className="note"),
    ])


def tab3_layout():
    return html.Div([
        html.Div([field("Compare payouts by", dcc.Dropdown(id="t3-factor", options=[{"label": k, "value": k} for k in FACTORS],
                                                           value="Storm category", clearable=False, searchable=False,
                                                           style={"width": "300px"}))],
                 className="controls card"),
        html.Div([dcc.Graph(id="t3-chart", config={"displayModeBar": False}), chart_hint()], className="card", style={"marginTop": "12px"}),
        html.Div(id="t3-table", className="card", style={"marginTop": "12px"}),
        html.Div(id="t3-note", className="note"),
    ])


def _ablation_table(rows):
    share = lambda v: f"{v:.0f}%" if v else "-"  # noqa: E731
    return html.Table([
        html.Thead(html.Tr([html.Th(c) for c in ["Model version", "R\u00b2 (log)", "Typical % error", "Building age importance",
                                                  "Post-FIRM importance", "Loss year importance"]])),
        html.Tbody([html.Tr([html.Td(name), html.Td(f"{r['r2_log']:.3f}"), html.Td(f"{r['median_abs_pct_error']:.0f}%"),
                             html.Td(share(r["age_share_pct"])), html.Td(share(r.get("post_firm_share_pct", 0))),
                             html.Td(share(r["year_share_pct"]))]) for name, r in rows.items()]),
    ], className="eval")


def ablation_block():
    """Is building age just a stand-in for the loss year? Does FEMA's post-FIRM flag help? (ablation_*.py)"""
    if not ABLATION:
        return html.Div()
    a = ABLATION
    cur, with_year = a["Current model"]["r2_log"], a["Add loss year (year control)"]["r2_log"]
    no_age, year_only = a["Remove building age"]["r2_log"], a["Loss year, no building age"]["r2_log"]
    recovered = 100 * (year_only - no_age) / (cur - no_age) if cur != no_age else float("nan")
    recovered_text = "essentially all" if recovered >= 95 else f"{recovered:.0f}%"
    note = (f"Removing building age costs {cur - no_age:.3f} of R\u00b2 ({cur:.3f} to {no_age:.3f}), so it carries real signal. "
            f"But a loss-year control on its own recovers {recovered_text} of that ({year_only:.3f}), and adding the year on top of age "
            f"barely moves the score ({with_year:.3f}). Most of what age contributes is also available from the calendar year: old "
            "buildings appear mostly in recent, larger claims, so the two can't be cleanly separated. Importance is split between "
            "correlated features, so read those columns loosely. The headline model above keeps age and no year control; the claim "
            "estimator uses the year-control version, and a year feature cannot extrapolate beyond the years in the data.")
    children = [html.H3("Is building age just a stand-in for the year?", style={"margin": "0 0 8px"}),
                _ablation_table(a), html.Div(note, className="note")]

    if ABLATION_FIRM:
        f = ABLATION_FIRM
        add_on = f["Add post-FIRM flag (age kept)"]["r2_log"]
        instead = f["Post-FIRM flag instead of building age"]["r2_log"]
        with_both = f["Loss year + post-FIRM flag (age kept)"]["r2_log"]
        note2 = (f"FEMA's post-FIRM flag says whether construction started after the community's flood map (or after 1974-12-31). It is "
                 f"complete for every claim, so it looked like a cleaner stand-in for age than the construction date. It isn't: used instead "
                 f"of age it scores {instead:.3f}, no better than having no age at all ({no_age:.3f}); added on top of age it doesn't help "
                 f"({add_on:.3f} vs {cur:.3f}); and once the loss year is in the model it adds essentially nothing ({with_both:.3f} vs "
                 f"{with_year:.3f}). It is another coarse era marker: roughly 5% of claims in the 1970s came from post-FIRM buildings, "
                 "versus about 45% in the 2010s. Individual folds can move by more than 0.1 (up to about 0.14 across these tests) between models whose overall scores are nearly equal, so differences of "
                 "a few thousandths are not meaningful.")
        children += [html.H3("Does FEMA's post-FIRM flag help?", style={"margin": "16px 0 8px"}), _ablation_table(f),
                     html.Div(note2, className="note")]
    return html.Div(children, className="card", style={"marginTop": "12px"})


def tab4_layout():
    m = METRICS
    rows = [
        ("Year-grouped cross-validation (headline)", m["headline_year_grouped_cv"], "Every claim predicted by a model that never saw its loss year."),
        ("  baseline: predict the training average", m["baseline_year_grouped_cv_predict_train_mean"], "What 'no model' scores."),
        (f"Time-based stress test ({m['temporal_stress_test']['test_years']})", m["temporal_stress_test"]["model"], f"Train on {m['temporal_stress_test']['train_years']}, test on later years."),
        ("  baseline: predict the training average", m["temporal_stress_test"]["baseline_predict_train_mean"], "Still negative: recent storm years are larger even in constant dollars."),
        ("Random split (for contrast only)", m["random_split_for_contrast"], "Looks better because claims from the same storm land in both train and test."),
    ]
    table = html.Table([
        html.Thead(html.Tr([html.Th(c) for c in ["Evaluation", "R² (log)", "Typical % error", "Claims tested", "What it means"]])),
        html.Tbody([html.Tr([html.Td(n), html.Td(f"{s['r2_log']:.2f}"), html.Td(f"{s['median_abs_pct_error']:.0f}%"),
                             html.Td(f"{s['n']:,}"), html.Td(txt)]) for n, s, txt in rows]),
    ], className="eval")

    imp = pd.Series(m["shap_importance_pct"]).sort_values()
    f3 = go.Figure(go.Bar(x=imp.values, y=[FEATURE_NAMES.get(k, k) for k in imp.index], orientation="h", marker_color=BLUE,
                          hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    style_fig(f3, "What the model relies on (share of mean |SHAP|, all claims)", 420, unified=False)
    f3.update_xaxes(title="% of total attribution")

    folds = ", ".join(f"{r:.2f}" for r in m["fold_r2"])
    return html.Div([
        html.Div([html.H3("Evaluation", style={"margin": "0 0 8px"}), table], className="card"),
        html.Div(f"R² by fold is {folds}: some storm seasons are far harder to predict than others. "
                 f"On years it has never seen, the model under-predicts: the actual median claim runs about {UNDERPREDICT:.1f}x its prediction "
                 "(payouts in a held-out storm year tend to be larger than in the years it learned from). The calibration chart below shows "
                 f"this, and the Claim Estimator corrects for it (by its own model's {EST_UNDER:.2f}x). The table above always covers all claims; everything "
                 "below the next line responds to the filters.", className="note"),
        ablation_block(),
        html.Div(id="t4-dyn", style={"marginTop": "12px"}),
        html.Div([dcc.Graph(figure=f3, config={"displayModeBar": False}), chart_hint()], className="card", style={"marginTop": "12px"}),
        html.Div("R² and error are on the log scale of the payout in 2025 dollars. 'Typical % error' is the median of |predicted - actual| / actual: half of the predictions are within that percentage of the actual payout. "
                 "Damage amounts are not features because they only exist once a claim does. Coverage limits are not features because the question is what storm, place and building characteristics explain (about 4% of building payments sit at the coverage limit).", className="note"),
    ])


def tab5_layout():
    return html.Div([
        html.Div(id="t5-kpis", className="kpi-row", style={"margin": "12px 0"}),
        html.Div([dcc.Graph(id="t5-chart", config={"displayModeBar": False}), chart_hint()], className="card"),
        html.Div([dcc.Graph(id="t5-year-chart", config={"displayModeBar": False}), chart_hint()], className="card", style={"marginTop": "12px"}),
        html.Div(f"An estimate for a paid claim, in constant 2025 dollars. This uses a version of the model that also knows the loss year "
                 f"(R² {YEAR_MODEL['r2_log']:.2f}, vs {METRICS['headline_year_grouped_cv']['r2_log']:.2f} without it), so building age is "
                 f"not standing in for the era. The raw output is scaled up by its median out-of-sample under-prediction ({EST_UNDER:.2f}x), and the "
                 "range covers the middle 80% of its out-of-sample errors, so it is wide by design. Payouts have risen faster than inflation, so "
                 f"the year matters; the default is {YEAR_DEFAULT}, the latest year with a large sample, and it was dominated by Helene and Milton. "
                 "Years with few claims (2019, 2021, 2025-26) are noisy, and the model cannot extrapolate beyond the years in the data. "
                 "These are patterns in past claims, not the effect of changing a storm or a building. Storm inputs are the wind speed at the storm's closest track point and its distance from the claim; location uses the "
                 "county's average claim coordinates.", className="note"),
    ])


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    return {"tab1": tab1_layout, "tab2": tab2_layout, "tab3": tab3_layout, "tab4": tab4_layout, "tab5": tab5_layout}[tab]()


# -------------------------------------------------------------- callbacks ----
def _measure_series(frame_group, measure):
    if measure == "claims":
        return frame_group.size()
    col = frame_group["amountPaid_real"]
    return col.sum() if measure == "total" else col.median()


FILTERS = [Input("f-years", "value"), Input("f-county", "value"), Input("f-occ", "value"), Input("f-flood", "value"),
           Input("f-storm", "value")]


def empty_fig(message, height=300):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16, color=INK_2), xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_fig(fig, height=height)


NO_CLAIMS = "No claims match these filters"


@app.callback(Output("f-count", "children"), *FILTERS)
def update_filter_count(years, counties, occs, floods, storms):
    sub = filtered(years, counties, occs, floods, storms)
    if len(sub) == N_CLAIMS:
        return f"Showing all {N_CLAIMS:,} paid claims. Filters apply to every tab except Claim Estimator, which has its own controls."
    return (f"Showing {len(sub):,} of {N_CLAIMS:,} paid claims ({100 * len(sub) / N_CLAIMS:.1f}%), "
            f"{money(sub['amountPaid_real'].sum())} paid in 2025 dollars.")


@app.callback(Output("filters", "style"), Output("estimator-bar", "style"), Input("tabs", "value"))
def toggle_filters(tab):
    shown, hidden = {"marginBottom": "12px", "display": "block"}, {"marginBottom": "12px", "display": "none"}
    return (hidden, shown) if tab == "tab5" else (shown, hidden)


def _measure_series(frame_group, measure):
    if measure == "claims":
        return frame_group.size()
    col = frame_group["amountPaid_real"]
    return col.sum() if measure == "total" else col.median()


def update_tab1(years, counties, occs, floods, storms, measure):
    sub = filtered(years, counties, occs, floods, storms)
    if sub.empty:
        return [kpi("Claims", "0")], empty_fig(NO_CLAIMS), empty_fig(NO_CLAIMS, 520)
    kpis = [kpi("Claims", f"{len(sub):,}"), kpi("Total paid", money(sub["amountPaid_real"].sum()), "2025 dollars"),
            kpi("Median claim", money(sub["amountPaid_real"].median()), "2025 dollars"),
            kpi("Near a storm", pct(100 * sub["matched"].mean()), "within 150 mi and 7 days")]

    y = _measure_series(sub.groupby("yearOfLoss"), measure)
    top = sub[sub["matched"]].groupby(["yearOfLoss", "storm_label"], observed=True).size().reset_index(name="n")
    top_storm = top.sort_values("n").groupby("yearOfLoss").tail(1).set_index("yearOfLoss")["storm_label"].astype(str).to_dict()
    tops = [top_storm.get(int(i), "none matched") for i in y.index]
    fig = go.Figure(go.Bar(x=y.index, y=y.values, marker_color=BLUE, customdata=tops,
                           hovertemplate="%{x}: %{y:,.0f}<br>Most claims from: %{customdata}<extra></extra>"))
    style_fig(fig, f"{MEASURES[measure]} by year of loss", 380, unified=False)
    fig.update_yaxes(title=MEASURES[measure], tickprefix="" if measure == "claims" else "$")

    g = sub.groupby("countyName", observed=True)
    size = _measure_series(g, measure)
    med = g["amountPaid_real"].median()
    counts = g.size()
    keep = counts[counts > 0].index
    size, med, counts = size[keep], med[keep], counts[keep]
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
    return kpis, fig, m


app.callback(Output("t1-kpis", "children"), Output("t1-year-chart", "figure"), Output("t1-map", "figure"),
             *FILTERS, Input("t1-measure", "value"))(update_tab1)


def update_tab2(years, counties, occs, floods, storms, metric, n):
    sub = filtered(years, counties, occs, floods, storms)
    sub = sub[sub["matched"]]
    if sub.empty:
        return empty_fig("No storm-matched claims match these filters"), html.Div()
    stats = sub.groupby("storm_label", observed=True).agg(
        claims=("amountPaid_real", "size"), total=("amountPaid_real", "sum"), median=("amountPaid_real", "median"))
    top = stats.sort_values(metric, ascending=False).head(n)
    fig = go.Figure(go.Bar(x=top[metric][::-1], y=top.index.astype(str)[::-1], orientation="h", marker_color=ORANGE,
                           customdata=np.c_[top["claims"][::-1], top["total"][::-1], top["median"][::-1]],
                           hovertemplate="%{y}<br>Claims: %{customdata[0]:,.0f}<br>Total paid: %{customdata[1]:$,.0f}"
                                         "<br>Median claim: %{customdata[2]:$,.0f}<extra></extra>"))
    style_fig(fig, f"Top {len(top)} storms by {MEASURES[metric].lower()}", max(360, 26 * len(top) + 120), unified=False)
    fig.update_xaxes(title=MEASURES[metric], tickprefix="" if metric == "claims" else "$")
    tbl = dash_table.DataTable(
        data=[{"Storm": str(i), "Claims": f"{int(r['claims']):,}", "Total paid (2025 $)": money(r["total"]),
               "Median claim (2025 $)": money(r["median"])}
              for i, r in top.iterrows()],
        columns=[{"name": c, "id": c} for c in ["Storm", "Claims", "Total paid (2025 $)", "Median claim (2025 $)"]],
        style_cell={"fontSize": "15px", "fontFamily": FONT_STACK, "padding": "9px 12px", "textAlign": "left",
                    "border": "none", "borderBottom": f"1px solid {BORDER}", "backgroundColor": SURFACE, "color": INK_2},
        style_header={"fontWeight": "650", "fontFamily": FONT_STACK, "color": INK, "backgroundColor": SURFACE,
                      "border": "none", "borderBottom": f"2px solid {BORDER}"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": PAGE_BG}],
    )
    return fig, tbl


app.callback(Output("t2-chart", "figure"), Output("t2-table", "children"), *FILTERS,
             Input("t2-metric", "value"), Input("t2-n", "value"))(update_tab2)


def update_tab3(years, counties, occs, floods, storms, factor):
    col, order, note = FACTORS[factor]
    sub = filtered(years, counties, occs, floods, storms)
    if sub.empty:
        return empty_fig(NO_CLAIMS, 460), html.Div(), note
    g = sub.groupby(col, observed=True)["amountPaid_real"]
    st = g.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    st.columns = ["q10", "q25", "q50", "q75", "q90"]
    st["n"] = g.size()
    st["mean"] = g.mean()
    st = st[st["n"] > 0].reindex([o for o in order if o in st.index])
    total_n = st["n"].sum()
    labels = [AXIS_SHORT_LABELS.get(str(i), str(i)) for i in st.index]
    # Plotly's box trace ignores hovertemplate for its auto-generated per-stat labels (always shows every fence/
    # min/max it was given), so the box's own hover is disabled and an invisible full-height bar supplies the
    # tooltip instead, with exactly the fields wanted.
    fig = go.Figure([
        go.Box(x=labels, q1=st["q25"], median=st["q50"], q3=st["q75"], lowerfence=st["q10"], upperfence=st["q90"],
              marker_color=BLUE, line=dict(color=BLUE), fillcolor="rgba(42,120,214,0.18)", hoverinfo="skip", showlegend=False),
        go.Bar(x=labels, y=(st["q90"] - st["q10"]).values, base=st["q10"].values,
              marker=dict(color="rgba(0,0,0,0)", line=dict(width=0)), showlegend=False,
              customdata=np.stack([st["q50"], st["q25"], st["q75"]], axis=-1),
              hovertemplate="%{x}<br>Median: %{customdata[0]:$,.2f}<br>25th percentile: %{customdata[1]:$,.2f}<br>"
                            "75th percentile: %{customdata[2]:$,.2f}<extra></extra>"),
    ])
    style_fig(fig, f"Payout by {factor.title()}", 460, unified=False, center_title=True)
    log_dollar_axis(fig, "y", "Payout (2025 $, log scale)")
    fig.update_xaxes(tickangle=0, tickfont=dict(size=12))
    tbl = html.Table([
        html.Thead(html.Tr([html.Th(c) for c in [factor.title(), "Claims", "% of Claims", "Median Payout", "Mean Payout",
                                                  "Middle 50% of Claims"]])),
        html.Tbody([html.Tr([html.Td(str(i)), html.Td(f"{int(r['n']):,}"), html.Td(f"{100 * r['n'] / total_n:.1f}%"),
                             html.Td(money(r["q50"])), html.Td(money(r["mean"])),
                             html.Td(f"{money(r['q25'])} to {money(r['q75'])}")]) for i, r in st.iterrows()]),
    ], className="eval")
    return fig, tbl, note


app.callback(Output("t3-chart", "figure"), Output("t3-table", "children"), Output("t3-note", "children"), *FILTERS,
             Input("t3-factor", "value"))(update_tab3)


def update_tab4(years, counties, occs, floods, storms):
    sub = filtered(years, counties, occs, floods, storms)
    if len(sub) < 200:
        return html.Div("Too few claims match these filters to evaluate the model (need at least 200).", className="card")
    y, p = sub["log_actual"].to_numpy(dtype="float64"), sub["log_pred"].to_numpy(dtype="float64")
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - p) ** 2).sum()) / sst if sst > 0 else float("nan")
    actual = sub["amountPaid_real"].to_numpy(dtype="float64")
    ape = float(np.median(np.abs(sub["prediction_amount"].to_numpy(dtype="float64") - actual) / actual) * 100)
    kpis = html.Div([kpi("Claims in this view", f"{len(sub):,}"),
                     kpi("R\u00b2 on this slice", f"{r2:.2f}", "out-of-sample predictions"),
                     kpi("Typical error on this slice", f"{ape:.0f}%", "median absolute % error"),
                     kpi("Actual / predicted (median)", f"{np.exp(np.median(y - p)):.2f}x", "1.00x = unbiased")], className="kpi-row")

    bins = pd.qcut(sub["prediction_amount"], 20, duplicates="drop")
    cal = sub.groupby(bins, observed=True).agg(pred=("prediction_amount", "median"), actual=("amountPaid_real", "median"))
    lo, hi = float(cal.min().min()) * 0.8, float(cal.max().max()) * 1.2
    f1 = go.Figure([go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction", line=dict(color=INK_3, dash="dash")),
                    go.Scatter(x=cal["pred"], y=cal["actual"], mode="lines+markers", name="Median actual claim", line=dict(color=BLUE),
                               hovertemplate="Predicted %{x:$,.0f}<br>Actual (median) %{y:$,.0f}<extra></extra>")])
    style_fig(f1, "Calibration: claims grouped by predicted payout (20 groups)", 420, unified=False)
    log_dollar_axis(f1, "x", "Median predicted payout (2025 $)", lo, hi)
    log_dollar_axis(f1, "y", "Median actual payout (2025 $)", lo, hi)

    by = sub.groupby("yearOfLoss").agg(n=("log_actual", "size"), actual=("log_actual", "mean"), pred=("log_pred", "mean"))
    f2 = go.Figure([
        go.Scatter(x=by.index, y=np.expm1(by["actual"]), name="Actual (typical claim)", line=dict(color=BLUE),
                   customdata=by["n"], hovertemplate="%{x}: actual %{y:$,.0f} (%{customdata:,.0f} claims)<extra></extra>"),
        go.Scatter(x=by.index, y=np.expm1(by["pred"]), name="Predicted (typical claim)", line=dict(color=ORANGE),
                   hovertemplate="%{x}: predicted %{y:$,.0f}<extra></extra>")])
    style_fig(f2, "Typical claim by loss year: actual vs predicted (out of sample)", 380)
    log_dollar_axis(f2, "y", "Typical payout (2025 $)")
    note = html.Div("R\u00b2 within a narrow slice is not comparable to the headline: filtering removes the variation the model "
                    "uses to separate claims, so scores usually fall. Predictions are the year-grouped out-of-sample ones.",
                    className="note")
    return html.Div([kpis, html.Div([dcc.Graph(figure=f1, config={"displayModeBar": False}), chart_hint()], className="card", style={"marginTop": "12px"}),
                     html.Div([dcc.Graph(figure=f2, config={"displayModeBar": False}), chart_hint()], className="card", style={"marginTop": "12px"}), note])


app.callback(Output("t4-dyn", "children"), *FILTERS)(update_tab4)


def update_tab5(county, occ, flood, age, elev, wind, dist, year):
    grid = np.arange(35, 155, 5)
    years = np.arange(YEAR_MIN, YEAR_MAX + 1)
    base = dict(county=county, occupancy=occ, flood=flood, age=age, elevated=bool(elev), dist=dist)
    preds = predict_batch([dict(base, wind=int(w), year=year) for w in grid] + [dict(base, wind=0, year=year)]
                          + [dict(base, wind=wind, year=int(yr)) for yr in years] + [dict(base, wind=wind, year=year)])
    wind_raw, no_storm_raw = preds[:len(grid)], preds[len(grid)]
    raws, raw = preds[len(grid) + 1:len(grid) + 1 + len(years)], float(preds[-1])
    est = raw * EST_UNDER  # correct the model's under-prediction on unseen years (see Model tab)
    lo, hi = raw * np.exp(EST_Q10), raw * np.exp(EST_Q90)
    kpis = [kpi("Typical payout (median)", money(est), f"for a claim like this in {year}, 2025 $"),
            kpi("80% of claims like this", f"{money(lo)} to {money(hi)}", "from out-of-sample errors"),
            kpi("Storm", "None nearby" if wind == 0 else STORM_ORDER[wind_category(wind) + 1], f"{wind} kt at {dist} mi" if wind else "no storm within range")]

    traces = [
        go.Scatter(x=grid, y=wind_raw * EST_UNDER, mode="lines", name="With a storm nearby", line=dict(color=BLUE),
                   hovertemplate="%{x} kt: %{y:$,.0f}<extra></extra>"),
        go.Scatter(x=[grid[0], grid[-1]], y=[no_storm_raw * EST_UNDER] * 2, mode="lines", name="No storm nearby",
                   line=dict(color=INK_3, dash="dash"), hovertemplate="No storm: %{y:$,.0f}<extra></extra>"),
    ]
    if wind > 0:
        traces.append(go.Scatter(x=[wind], y=[est], mode="markers", name="Selected", marker=dict(color=ORANGE, size=12),
                                 hovertemplate="Selected: %{y:$,.0f}<extra></extra>"))
    fig = go.Figure(traces)
    style_fig(fig, f"Typical payout as storm strength changes ({dist} mi from the track, {year} loss)", 380)
    fig.update_xaxes(title="Storm wind speed at closest approach (kt)")
    fig.update_yaxes(title="Typical payout (2025 $)", tickprefix="$")

    counts =[CLAIMS_BY_YEAR.get(int(yr), 0) for yr in years]
    yfig = go.Figure([
        go.Scatter(x=years, y=raws * np.exp(EST_Q90), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
        go.Scatter(x=years, y=raws * np.exp(EST_Q10), mode="lines", line=dict(width=0), fill="tonexty",
                   fillcolor="rgba(42,120,214,0.12)", name="80% range", hoverinfo="skip"),
        go.Scatter(x=years, y=raws * EST_UNDER, mode="lines", name="Typical payout", line=dict(color=BLUE), customdata=counts,
                   hovertemplate="%{x}: %{y:$,.0f} (%{customdata:,.0f} claims that year)<extra></extra>"),
        go.Scatter(x=[year], y=[est], mode="markers", name="Selected year", marker=dict(color=ORANGE, size=12),
                   hovertemplate="Selected: %{y:$,.0f}<extra></extra>"),
    ])
    style_fig(yfig, "The same claim across loss years: how much payouts have grown beyond inflation", 380)
    log_dollar_axis(yfig, "y", "Typical payout (2025 $)")
    return kpis, fig, yfig


app.callback(Output("t5-kpis", "children"), Output("t5-chart", "figure"), Output("t5-year-chart", "figure"),
             Input("t5-county", "value"), Input("t5-occ", "value"), Input("t5-flood", "value"), Input("t5-age", "value"),
             Input("t5-elev", "value"), Input("t5-wind", "value"), Input("t5-dist", "value"),
             Input("t5-year", "value"))(update_tab5)


_log("app ready")

if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 8058)))
