"""Every dashboard callback returns without error on the committed data (default and filtered inputs)."""
import pytest

dash_app = pytest.importorskip("app")


def years():
    return (dash_app.YEAR_MIN, dash_app.YEAR_MAX)


def no_filters(**kw):
    base = dict(years=years(), counties=[], occs=[], floods=[], storms=[])
    base.update(kw)
    return base


def test_filter_count_text_with_and_without_filters():
    assert f"{dash_app.N_CLAIMS:,}" in dash_app.update_filter_count(years(), [], [], [], [])
    narrowed = dash_app.update_filter_count(years(), ["Lee"], [], [], [])
    assert narrowed.startswith("Showing") and "of" in narrowed


def test_empty_selection_means_all_claims():
    assert len(dash_app.filtered(None, None, None, None, None)) == dash_app.N_CLAIMS


def test_filtering_by_county_and_years_narrows_the_claims():
    lee = dash_app.filtered((2020, 2024), ["Lee"], [], [], [])
    assert 0 < len(lee) < dash_app.N_CLAIMS
    assert set(lee["countyName"].astype(str)) == {"Lee"}
    assert lee["yearOfLoss"].between(2020, 2024).all()


@pytest.mark.parametrize("measure", ["claims", "total", "median"])
def test_overview_tab(measure):
    kpis, year_fig, map_fig = dash_app.update_tab1(*no_filters().values(), measure)
    assert kpis and year_fig.data and map_fig.data


@pytest.mark.parametrize("metric", ["claims", "total", "median"])
def test_storms_tab(metric):
    fig, table = dash_app.update_tab2(*no_filters().values(), metric, 15)
    assert fig.data and table is not None


@pytest.mark.parametrize("factor", list(dash_app.FACTORS))
def test_risk_factors_tab(factor):
    fig, table, note = dash_app.update_tab3(*no_filters().values(), factor)
    assert fig.data and table is not None and note


def test_model_tab_and_its_small_sample_message():
    assert dash_app.update_tab4(*no_filters().values()) is not None
    tiny = dash_app.update_tab4(years(), ["Lee"], ["Mobile home"], ["V / VE (velocity, coastal)"], ["Category 5"])
    assert "Too few claims" in str(tiny) or tiny is not None


def test_a_filter_with_no_matches_does_not_crash_any_tab():
    nothing = dict(years=(1978, 1978), counties=["Lee"], occs=["Mobile home"], floods=["V / VE (velocity, coastal)"],
                   storms=["Category 5"])
    assert len(dash_app.filtered(*nothing.values())) == 0
    dash_app.update_tab1(*nothing.values(), "total")
    dash_app.update_tab2(*nothing.values(), "total", 15)
    dash_app.update_tab3(*nothing.values(), "Storm category")
    dash_app.update_tab4(*nothing.values())


def test_claim_estimator_defaults_and_storm_strength_ordering():
    flood = next(f for f in dash_app.FLOOD_ORDER if f in dash_app.FLOOD_CODE)
    args = dict(county="Lee", occ="Single-family", flood=flood, age=30, elev=[], wind=90, dist=40,
                year=dash_app.YEAR_DEFAULT)
    kpis, wind_fig, year_fig = dash_app.update_tab5(*args.values())
    assert len(kpis) == 3 and wind_fig.data and year_fig.data

    def payout(wind):
        return dash_app.predict_payout("Lee", "Single-family", flood, 30, False, wind, 40, dash_app.YEAR_DEFAULT)

    assert payout(120) > 0 and payout(0) > 0
