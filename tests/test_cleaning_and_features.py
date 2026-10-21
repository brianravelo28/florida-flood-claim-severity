"""Cleaning, feature engineering and inflation adjustment on small hand-built fixtures."""
import numpy as np
import pandas as pd
import pytest

import build_dataset as bd


@pytest.fixture(autouse=True)
def fresh_report():
    bd.report.clear()
    yield
    bd.report.clear()


def raw_claims():
    base = dict(amountPaidOnBuildingClaim=np.nan, amountPaidOnContentsClaim=np.nan,
                amountPaidOnIncreasedCostOfComplianceClaim=np.nan, dateOfLoss="2000-09-10T00:00:00.000Z",
                latitude=27.0, longitude=-81.0)
    rows = [
        dict(base, amountPaidOnBuildingClaim=1000.0, amountPaidOnContentsClaim=200.0),    # kept: 1,200
        dict(base),                                                                       # nothing paid
        dict(base, amountPaidOnBuildingClaim=-500.0),                                     # negative total
        dict(base, amountPaidOnBuildingClaim=900.0, latitude=np.nan),                     # no location
        dict(base, amountPaidOnBuildingClaim=900.0, latitude=40.7, longitude=-74.0),      # New York, not Florida
        dict(base, amountPaidOnBuildingClaim=900.0, dateOfLoss="1975-06-01T00:00:00.000Z"),  # before 1978
    ]
    return pd.DataFrame(rows)


def test_payout_is_the_sum_of_the_three_paid_fields():
    c = bd.prepare_claims(raw_claims())
    assert c["amountPaid"].tolist() == [1200.0]


def test_every_exclusion_is_counted_with_its_reason():
    bd.prepare_claims(raw_claims())
    assert bd.report["rows_dropped"] == {
        "missing lat/lon/date": 1,
        "amountPaid <= 0 (closed without payment)": 2,
        "dateOfLoss before 1978": 1,
        "outside Florida bounds": 1,
    }


def feature_claims(construction, loss_year=2000, post_firm=False):
    n = len(construction)
    return pd.DataFrame({
        "originalConstructionDate": construction, "yearOfLoss": [loss_year] * n,
        "elevatedBuildingIndicator": [False] * n, "postFIRMConstructionIndicator": [post_firm] * n,
        "ratedFloodZone": ["AE"] * n, "occupancyType": [1] * n, "distance_from_track_mi": [30.0] * n,
        "storm_wind_speed_kt": [80.0] * n, "amountPaid": [1000.0] * n, "countyCode": [12071] * n,
    })


def test_unusable_construction_dates_are_flagged_and_get_the_median_age():
    c = feature_claims([
        "1980-05-01T00:00:00.000Z",   # age 20
        "1492-10-12T00:00:00.000Z",   # FEMA's placeholder
        "2010-01-01T00:00:00.000Z",   # built after the loss
        None,                         # no date
        "1850-01-01T00:00:00.000Z",   # age 150
        "1700-01-01T00:00:00.000Z",   # before 1800
    ])
    out = bd.engineer_features(c)
    assert out["building_age_missing"].tolist() == [0, 1, 1, 1, 0, 1]
    median_known = int(np.median([20, 150]))
    assert out.loc[out["building_age_missing"] == 1, "building_age_years"].eq(median_known).all()
    assert out["building_age_display_years"].notna().all()
    assert bd.report["construction_year_unknown_pct"] == pytest.approx(100 * 4 / 6, abs=0.01)


def test_building_age_is_loss_year_minus_construction_year():
    out = bd.engineer_features(feature_claims(["1990-03-04T00:00:00.000Z", "2000-01-01T00:00:00.000Z"]))
    assert out["building_age_years"].tolist() == [10, 0]


def test_post_firm_and_elevated_flags_read_booleans_and_strings():
    c = feature_claims(["1990-01-01T00:00:00.000Z"] * 3)
    c["postFIRMConstructionIndicator"] = [True, "True", False]
    c["elevatedBuildingIndicator"] = ["true", False, "False"]
    out = bd.engineer_features(c)
    assert out["post_firm"].tolist() == [1, 1, 0]
    assert out["is_elevated"].tolist() == [1, 0, 0]


def test_legacy_and_newer_occupancy_codes_merge_into_one_group():
    g = bd.OCCUPANCY_TO_GROUP
    assert g[1] == g[11] == "Single-family"
    assert g[2] == g[12] == "2-4 units"
    assert g[3] == g[13] == "5+ units"
    assert {g[c] for c in (4, 6, 17, 18, 19)} == {"Non-residential"}
    assert set(bd.OCCUPANCY_LABELS) == set(g), "every labelled occupancy code needs a group"


def test_flood_zone_groups():
    enc, _ = bd.encode_flood_zone(pd.Series(["X", "B", "AE", "A5", "AO", "AH", "A", "VE", "V", "D", None]))
    assert enc.tolist() == [1, 1, 2, 2, 3, 3, 3, 4, 4, 0, 0]


def test_a99_is_grouped_with_the_special_a_zones():
    enc, _ = bd.encode_flood_zone(pd.Series(["A99"]))
    assert enc.tolist() == [3]


def test_wind_speed_in_knots_maps_to_saffir_simpson_categories():
    # HURDAT2 wind is in knots: tropical storm 34-63, Category 1 64-82, 2 83-95, 3 96-112, 4 113-136, 5 137+
    wind = [33.9, 34, 63.9, 64, 82.9, 83, 95.9, 96, 112.9, 113, 136.9, 137, 160, np.nan]
    assert bd.wind_to_category(wind).tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, -1]


@pytest.mark.parametrize("storm, peak_kt, category", [
    ("Andrew 1992 (landfall 145 kt)", 145, "Category 5"),
    ("Michael 2018 (landfall 140 kt)", 140, "Category 5"),
    ("Ian 2022 (landfall 130 kt)", 130, "Category 4"),
    ("Charley 2004 (landfall 130 kt)", 130, "Category 4"),
    ("Irma 2017 (Marco Island landfall, 100 kt)", 100, "Category 3"),
])
def test_known_hurricanes_get_their_published_category(storm, peak_kt, category):
    code = int(bd.wind_to_category([peak_kt])[0])
    assert bd.CATEGORY_LABELS[code] == category, storm


def cpi_fixture():
    months = pd.date_range("2019-01-01", "2026-03-01", freq="MS")
    value = pd.Series(105.0, index=months)
    value[value.index.year == 2020] = 80.0
    value[value.index.year == 2025] = 100.0
    value[pd.Timestamp("2025-10-01")] = np.nan  # FRED has no published value for this month
    return pd.DataFrame({"observation_date": months, "CPIAUCSL": value.to_numpy()})


def test_inflation_factor_is_one_in_the_base_year(monkeypatch):
    monkeypatch.setattr(bd, "fetch_cpi", cpi_fixture)
    c = pd.DataFrame({"dateOfLoss": pd.to_datetime(["2025-03-15", "2020-06-10"]), "amountPaid": [1000.0, 1000.0]})
    out = bd.add_real_dollars(c)
    assert out["cpi_factor"].iloc[0] == pytest.approx(1.0)
    assert out["amountPaid_real"].iloc[1] == pytest.approx(1250.0)  # CPI 80 then, 100 in the base year
    assert bd.report["cpi_interpolated_months"] == ["2025-10"]
    assert np.isfinite(out["log_amountpaid_real"]).all()
