"""The sensitivity checks: payments by decade, intervals that resample whole storms, elevated vs not within groups, and other price indexes."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from constants import BIG_THREE_STORMS, MIN_CLAIMS
from pipeline import build_dataset as bd
from pipeline import build_deploy_data as bdd

DEPLOY = Path(__file__).resolve().parent.parent / "deploy_data"


@pytest.fixture(autouse=True)
def fresh_report():
    bd.report.clear()
    yield
    bd.report.clear()


def load(name):
    return json.loads((DEPLOY / name).read_text())


@pytest.fixture(scope="module")
def claims():
    return pd.read_parquet(DEPLOY / "claims.parquet")


# ------------------------------------------------------------ claims that paid nothing, by decade
def raw_with(dates, paid):
    base = dict(amountPaidOnBuildingClaim=np.nan, amountPaidOnContentsClaim=np.nan, amountPaidOnIncreasedCostOfComplianceClaim=np.nan,
                latitude=27.0, longitude=-81.0, nonPaymentReasonBuilding=np.nan, nonPaymentReasonContents=np.nan)
    return pd.DataFrame([dict(base, dateOfLoss=f"{d}T00:00:00.000Z", amountPaidOnBuildingClaim=p) for d, p in zip(dates, paid)])


def test_zero_and_negative_totals_are_counted_apart():
    bd.prepare_claims(raw_with(["2000-01-01"] * 4, [100.0, 0.0, -50.0, np.nan]))
    assert bd.report["total_paid_zero_or_negative"] == {"zero": 2, "negative": 1}


def test_claims_that_paid_nothing_are_counted_by_decade_of_loss():
    raw = raw_with(["1985-05-01", "1986-05-01", "2021-05-01", "2022-05-01", "2023-05-01", "2024-05-01"], [100.0, 0.0, 5.0, 6.0, 7.0, -10.0])
    bd.prepare_claims(raw)
    assert bd.report["closed_without_payment_by_decade"] == {
        1980: {"claims": 2, "without_payment": 1, "pct": 50.0},
        2020: {"claims": 4, "without_payment": 1, "pct": 25.0},
    }


# ------------------------------------------------------------ other price indexes
def index_frame(by_year, name="SERIES", gap=None):
    months = pd.date_range("2014-01-01", "2026-03-01", freq="MS")
    value = pd.Series([by_year.get(m.year, 100.0) for m in months], index=months, dtype="float64")
    if gap is not None:
        value[pd.Timestamp(gap)] = np.nan
    return pd.DataFrame({"observation_date": months, name: value.to_numpy()})


def test_another_price_index_is_applied_the_same_way_as_the_cpi():
    c = pd.DataFrame({"dateOfLoss": pd.to_datetime(["2025-03-15", "2020-06-10"]), "amountPaid": [1000.0, 1000.0]})
    real = bd.deflate_with(bd.monthly_index(index_frame({2020: 80.0, 2025: 100.0})), c)
    assert real.tolist() == pytest.approx([1000.0, 1250.0])  # the same numbers the CPI-U test uses


def test_a_gap_inside_an_index_is_filled_and_a_missing_loss_month_stops_the_build():
    series = bd.monthly_index(index_frame({2025: 100.0}, gap="2025-10-01"))
    assert series[pd.Period("2025-10")] == pytest.approx(100.0)
    late = pd.DataFrame({"dateOfLoss": pd.to_datetime(["2026-09-01"]), "amountPaid": [1.0]})
    with pytest.raises(AssertionError, match="price index missing"):
        bd.deflate_with(series, late)


def sensitivity_claims():
    return pd.DataFrame({
        "dateOfLoss": pd.to_datetime(["2015-06-01", "2022-06-01", "2023-06-01"]),
        "yearOfLoss": [2015, 2022, 2023],
        "amountPaid": [1000.0, 2000.0, 3000.0],
        "amountPaid_real": [1100.0, 2200.0, 3300.0],  # whatever the CPI-U gave
        "event": pd.array([pd.NA, "Hurricane Ian", "Flooding"], dtype="string"),
    })


def test_the_sensitivity_gives_decade_medians_the_blend_and_the_ratio_to_the_2010s():
    indexes = {"materials": index_frame({2015: 50.0}), "wages": index_frame({})}  # materials were half as dear in 2015
    out = bd.deflator_sensitivity(sensitivity_claims(), indexes)
    assert out["cpi"]["median_by_decade"] == {2010: 1100.0, 2020: 2750.0}
    assert out["materials"]["median_by_decade"][2010] == pytest.approx(2000.0)      # 1000 x 100 / 50
    assert out["wages"]["median_by_decade"][2010] == pytest.approx(1000.0)
    assert out["blend"]["median_by_decade"][2010] == pytest.approx(1500.0)           # average of the two adjusted payouts
    for key in ("materials", "wages", "blend"):
        assert out[key]["median_2020s_without_ian_helene_milton"] == pytest.approx(3000.0)  # only the "Flooding" claim remains
    assert out["materials"]["ratio_to_2010s"] == pytest.approx(1.5)
    assert out["cpi"]["ratio_to_2010s"] == pytest.approx(3300.0 / 1100.0)


# ------------------------------------------------------------ intervals that resample whole storms
def test_an_interval_from_identical_storms_has_no_width_and_one_from_different_storms_is_wide():
    values = pd.Series(np.repeat([10.0] * 20, 50))
    assert bdd.median_interval(values, pd.Series(np.repeat(np.arange(20), 50)), reps=200) == (20, 10.0, 10.0)
    values = pd.Series(np.repeat(np.arange(1.0, 21.0), 50))  # 20 storms of 50 claims, each storm at a different level
    k, low, high = bdd.median_interval(values, pd.Series(np.repeat(np.arange(20), 50)), reps=500)
    assert k == 20 and low < values.median() < high and high - low > 3


def test_the_same_seed_gives_the_same_interval():
    values, groups = pd.Series(np.arange(1.0, 101.0)), pd.Series(np.repeat(np.arange(10), 10))
    assert bdd.median_interval(values, groups, reps=100, seed=7) == bdd.median_interval(values, groups, reps=100, seed=7)


def test_a_claim_with_no_event_label_joins_its_loss_year_group():
    table = pd.DataFrame({"event": pd.array(["Hurricane Ian", pd.NA, pd.NA, pd.NA], dtype="string"), "yearOfLoss": [2022, 2022, 2022, 1990]})
    assert bdd.storm_groups(table).tolist() == ["Hurricane Ian", "no label 2022", "no label 2022", "no label 1990"]


# ------------------------------------------------------------ elevated vs not elevated within groups
def test_elevated_vs_not_is_compared_inside_each_group_with_enough_claims_on_both_sides():
    n = MIN_CLAIMS
    rows = ([("Elevated", "0 in", "AE", 100.0)] * n + [("Not elevated", "0 in", "AE", 200.0)] * n          # lower when elevated
            + [("Elevated", "1-6 in", "AE", 300.0)] * n + [("Not elevated", "1-6 in", "AE", 250.0)] * n      # higher when elevated
            + [("Elevated", "7-12 in", "AE", 1.0)] * (n - 1) + [("Not elevated", "7-12 in", "AE", 2.0)] * n)  # too few elevated claims
    table = pd.DataFrame(rows, columns=["elevated", "depth_band", "flood_zone_group", "p"])
    table["yearOfLoss"] = 2021
    pay = table.pop("p")
    out = bdd.elevated_within_groups(table, pay, (table["yearOfLoss"] // 10) * 10)
    assert out["water_depth"]["cells"] == 2 and out["water_depth"]["elevated_lower"] == 1
    assert out["flood_zone"]["cells"] == 1  # all three depth bands sit in one zone, which has enough claims on both sides


# ------------------------------------------------------------ the committed bundle
def test_the_payment_counts_by_decade_add_up_to_the_cleaning_steps(claims):
    report = load("build_report.json")
    by_decade = report["closed_without_payment_by_decade"].values()
    dropped = report["rows_dropped"]
    assert sum(d["without_payment"] for d in by_decade) == dropped["amountPaid <= 0 (closed without payment)"]
    assert sum(report["total_paid_zero_or_negative"].values()) == dropped["amountPaid <= 0 (closed without payment)"]
    paid = sum(d["claims"] - d["without_payment"] for d in by_decade)
    assert paid == len(claims) + dropped["outside Florida bounds"] + dropped["dateOfLoss before 1978"]
    for d in by_decade:
        assert d["pct"] == round(100 * d["without_payment"] / d["claims"], 1)


def test_fewer_claims_closed_without_payment_in_the_2020s_than_in_any_earlier_decade():
    share = {int(k): v["pct"] for k, v in load("build_report.json")["closed_without_payment_by_decade"].items()}
    assert all(share[2020] < v for k, v in share.items() if k < 2020)


def test_the_spread_of_payouts_by_decade_matches_the_claims_table_and_rises_every_decade(claims):
    spread = load("summary.json")["spread_by_decade"]
    decade = (claims["yearOfLoss"] // 10) * 10
    pay = claims["amountPaid_real"].astype("float64")
    for d, row in spread.items():
        sel = pay[decade == int(d)]
        assert row["p25"] == pytest.approx(sel.quantile(0.25)) and row["p75"] == pytest.approx(sel.quantile(0.75))
        assert row["p90"] == pytest.approx(sel.quantile(0.9))
    ordered = [spread[k] for k in sorted(spread, key=int)]
    for q in ("p25", "p75", "p90"):
        values = [row[q] for row in ordered]
        assert values == sorted(values), q


def test_storm_intervals_bracket_each_median_and_count_the_storm_groups(claims):
    intervals = load("summary.json")["median_intervals_by_storm"]
    year = claims["yearOfLoss"].astype(int)
    ev = claims["event"].astype("string")
    group = ev.fillna("no label " + year.astype(str))
    pay = claims["amountPaid_real"].astype("float64")
    big3 = ev.isin(BIG_THREE_STORMS).fillna(False).astype(bool)
    masks = {k: (year // 10 * 10) == int(k) for k in intervals if k.isdigit()}
    masks["without_ian_helene_milton_2020s"] = (year >= 2020) & ~big3
    assert set(masks) == set(intervals)
    for name, row in intervals.items():
        mask = masks[name]
        assert row["claims"] == int(mask.sum()) and row["storm_groups"] == group[mask].nunique()
        assert row["median"] == pytest.approx(pay[mask].median())
        assert row["low"] < row["median"] < row["high"], name


def test_a_stored_storm_interval_can_be_recomputed(claims):
    stored = load("summary.json")["median_intervals_by_storm"]["2010"]
    year = claims["yearOfLoss"].astype(int)
    mask = (year >= 2010) & (year <= 2019)
    pay = claims["amountPaid_real"].astype("float64")
    k, low, high = bdd.median_interval(pay[mask], bdd.storm_groups(claims)[mask])
    assert (k, low, high) == (stored["storm_groups"], pytest.approx(stored["low"]), pytest.approx(stored["high"]))


def test_the_2020s_interval_sits_above_the_2010s_but_the_one_without_the_three_storms_overlaps_it():
    s = load("summary.json")["median_intervals_by_storm"]
    assert s["2020"]["low"] > s["2010"]["high"]
    without = s["without_ian_helene_milton_2020s"]
    assert without["low"] < s["2010"]["high"] and without["high"] > s["2010"]["low"]


def test_elevated_within_groups_matches_the_claims_table_and_holds_in_every_depth_band(claims):
    stored = load("summary.json")["findings"]["elevated_within_groups"]
    decade = (claims["yearOfLoss"] // 10) * 10
    pay = claims["amountPaid_real"].astype("float64")
    for name, col in (("water_depth", "depth_band"), ("flood_zone", "flood_zone_group")):
        cells = lower = 0
        for d in (1990, 2000, 2010, 2020):
            sub = claims[decade == d].assign(p=pay)
            g = sub.groupby([col, "elevated"], observed=True)["p"].agg(["size", "median"])
            for group in sub[col].cat.categories:
                if (group, "Elevated") in g.index and (group, "Not elevated") in g.index:
                    yes, no = g.loc[(group, "Elevated")], g.loc[(group, "Not elevated")]
                    if yes["size"] >= MIN_CLAIMS and no["size"] >= MIN_CLAIMS:
                        cells += 1
                        lower += int(yes["median"] < no["median"])
        assert (stored[name]["cells"], stored[name]["elevated_lower"]) == (cells, lower), name
    assert stored["water_depth"]["elevated_lower"] == stored["water_depth"]["cells"]
    assert all(c["elevated"]["median"] < c["not_elevated"]["median"] for c in stored["flood_zone"]["by_cell"] if c["decade"] >= 2010)


def test_the_cpi_row_of_the_price_index_check_is_the_published_result_and_every_index_keeps_the_2020s_above_the_2010s():
    results = load("build_report.json")["deflator_sensitivity"]["results"]
    published = load("summary.json")
    for d, row in published["by_decade"].items():
        assert results["cpi"]["median_by_decade"][d] == pytest.approx(row["median"], rel=1e-4)
    assert results["cpi"]["median_2020s_without_ian_helene_milton"] == pytest.approx(
        published["sensitivity_2020s"]["without_ian_helene_milton"]["median"], rel=1e-4)
    assert set(results) == {"cpi", "materials", "wages", "blend"}
    for key, row in results.items():
        assert row["ratio_to_2010s"] > 1, key
        assert row["ratio_to_2010s"] == pytest.approx(row["median_2020s_without_ian_helene_milton"] / row["median_by_decade"]["2010"])
