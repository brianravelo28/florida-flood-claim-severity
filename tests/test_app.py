"""Every dashboard callback returns without error on the committed data (default, filtered and empty inputs)."""
import pytest

dash_app = pytest.importorskip("app")

MIN = dash_app.MIN_CLAIMS


def years():
    return (dash_app.YEAR_MIN, dash_app.YEAR_MAX)


def no_filters():
    return years(), [], [], []


def text_of(component):
    """All text inside a Dash component tree."""
    if isinstance(component, (str, int, float)):
        return str(component)
    if isinstance(component, (list, tuple)):
        return " ".join(text_of(c) for c in component)
    return text_of(getattr(component, "children", "") or "")


def test_filter_count_text_with_and_without_filters():
    assert f"{dash_app.N_CLAIMS:,}" in dash_app.update_filter_count(*no_filters())
    narrowed = dash_app.update_filter_count(years(), ["Lee"], [], [])
    assert narrowed.startswith("Showing") and "of" in narrowed


def test_empty_selection_means_all_claims():
    assert len(dash_app.filtered(None, None, None, None)) == dash_app.N_CLAIMS


def test_filtering_by_county_and_years_narrows_the_claims():
    lee = dash_app.filtered((2020, 2024), ["Lee"], [], [])
    assert 0 < len(lee) < dash_app.N_CLAIMS
    assert set(lee["countyName"].astype(str)) == {"Lee"}
    assert lee["yearOfLoss"].between(2020, 2024).all()


@pytest.mark.parametrize("measure", ["claims", "total", "median"])
def test_overview_tab(measure):
    kpis, year_fig, note, decades, map_fig = dash_app.update_tab1(*no_filters(), measure)
    assert kpis and year_fig.data and map_fig.data
    assert "partial" not in text_of(note) and "partial" not in text_of(decades)
    assert "1978-79" in text_of(decades) and "2020-26" in text_of(decades)


def test_overview_year_chart_shows_claims_per_year_and_flags_small_years():
    _, year_fig, note, _, _ = dash_app.update_tab1(*no_filters(), "median")
    bar = year_fig.data[0]
    assert len(bar.customdata) == len(bar.x) and int(bar.customdata[0][0].replace(",", "").rstrip("*")) > 0
    assert any(c[0].endswith("*") for c in bar.customdata), "small years carry an asterisk on their claim count"
    assert not bar.customdata[list(bar.x).index(2024)][0].endswith("*")
    assert "fewer than" in text_of(note), "the small early and late years should be explained"
    assert len(set(bar.marker.color)) == 2


@pytest.mark.parametrize("metric", ["claims", "total", "median"])
def test_storms_tab_ranks_named_storms_with_their_claim_counts(metric):
    coverage, fig, table, other = dash_app.update_tab2(*no_filters(), metric, 15)
    assert "FEMA event label" in coverage and "hurricane or tropical storm" in coverage
    assert fig.data and len(fig.data[0].y) == 15
    assert all("Hurricane" in y or "Tropical Storm" in y for y in fig.data[0].y)
    assert "Claims" in text_of(table)
    assert "Flooding" in text_of(other) and "1979-2000" in text_of(other)


def test_other_events_show_a_median_even_when_they_have_few_claims():
    # 2025 has one other FEMA-labeled event with 26 claims: its median is shown and its claim count carries an asterisk
    _, _, _, other = dash_app.update_tab2((2025, 2025), [], [], [], "total", 10)
    text = text_of(other)
    assert "Nor'easter" in text and "26*" in text
    assert "$" in text.split("26*")[1][:30], "the median should follow the claim count"
    assert "Gray rows" not in text and "fewer than" not in text, "the flag is the asterisk; no note under the table"


def test_the_storms_tab_keeps_its_notes_in_a_collapsible_section():
    layout = text_of(dash_app.tab2_layout())
    assert "About event labels" in layout
    assert "Labels are sparse before the mid-1980s" in layout and "cover only their labeled claims" in layout
    assert "not named hurricanes or tropical storms" not in layout


def test_storms_tab_lists_alberto_with_its_years_and_one_georges():
    _, fig, _, _ = dash_app.update_tab2((1990, 2010), [], [], [], "claims", 25)
    names = list(fig.data[0].y)
    assert "Hurricane Georges" in names
    assert not any("Georges (" in n for n in names)


@pytest.mark.parametrize("factor", list(dash_app.FACTORS))
def test_risk_factors_tab(factor):
    fig, table, note, extra = dash_app.update_tab3(*no_filters(), factor)
    assert fig.data and table is not None
    assert "Drag to zoom into part of the chart" in text_of(note), "every chart's About section carries the zoom tip"
    assert "legend" not in text_of(note), "no legend on these charts, so no legend tip"
    assert ("Occupancy type" == factor) == ("codes describe the same building types" in text_of(note))
    assert "claims" in fig.data[0].x[0], "each group's claim count is on the chart"
    assert ("Building age" == factor) == (extra is not None and bool(getattr(extra, "children", None)))


def test_building_age_shows_each_decade_separately_with_the_entanglement_note():
    fig, _, note, extra = dash_app.update_tab3(*no_filters(), "Building age")
    assert "move together" in text_of(note) and "may reflect when the claim happened" in text_of(note)
    strat_fig = extra.children[0].figure
    assert len(strat_fig.data) >= 4, "one line per decade of loss"
    assert "Claims behind each line" in text_of(extra)


def test_groups_under_the_minimum_are_flagged_not_charted():
    # In 1978-79 the "Unknown" occupancy group has 27 claims, fewer than the minimum
    fig, table, note, _ = dash_app.update_tab3((1978, 1979), [], [], [], "Occupancy type")
    assert "27*" in text_of(table) and "Unknown" in text_of(table)
    assert not any("Unknown" in label for label in fig.data[0].x)
    assert "marked *" in text_of(note) and "30 or more claims" in text_of(note)


def test_a_filter_with_no_matches_does_not_crash_any_tab():
    nothing = ((1978, 1978), ["Lee"], ["Mobile home"], ["V / VE (velocity, coastal)"])
    assert len(dash_app.filtered(*nothing)) == 0
    dash_app.update_tab1(*nothing, "total")
    dash_app.update_tab2(*nothing, "total", 15)
    for factor in dash_app.FACTORS:
        dash_app.update_tab3(*nothing, factor)


def test_a_filter_where_no_group_is_large_enough_does_not_crash():
    few = dash_app.filtered((1984, 1984), ["Alachua"], [], [])
    assert 0 < len(few) < MIN
    args = ((1984, 1984), ["Alachua"], [], [])
    dash_app.update_tab1(*args, "median")
    dash_app.update_tab2(*args, "median", 10)
    dash_app.update_tab3(*args, "Building age")


def test_interaction_tips_live_in_the_about_sections():
    year_note = dash_app.update_tab1(*no_filters(), "total")[2]
    assert "Drag to zoom into part of the chart" in text_of(year_note) and "legend" not in text_of(year_note)
    overview = text_of(dash_app.tab1_layout())
    assert "Drag to pan the map; scroll to zoom; double-click to reset the view." in overview
    assert "About this map" in overview
    storms = text_of(dash_app.tab2_layout())
    assert "Drag to zoom into part of the chart" in storms
    extra = dash_app.update_tab3(*no_filters(), "Building age")[3]
    assert "Click a legend entry to hide that series" in text_of(extra), "the by-decade chart has a legend"


def test_tables_get_no_zoom_tips():
    _, _, _, decades, _ = dash_app.update_tab1(*no_filters(), "total")
    assert "Drag to" not in text_of(decades) and "About" not in text_of(decades), "the decade table has nothing to caution about"
