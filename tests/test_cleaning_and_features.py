"""Cleaning, feature engineering, event labels and inflation adjustment on small hand-built fixtures."""
import numpy as np
import pandas as pd
import pytest

from pipeline import build_dataset as bd


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


def test_rows_identical_except_for_id_are_counted_and_kept():
    row = dict(amountPaidOnBuildingClaim=5000.0, amountPaidOnContentsClaim=0.0, amountPaidOnIncreasedCostOfComplianceClaim=0.0,
               dateOfLoss="2005-10-24T00:00:00.000Z", latitude=25.1, longitude=-80.4, occupancyType=15.0)
    raw = pd.DataFrame([dict(row, id=1), dict(row, id=2), dict(row, id=3), dict(row, id=4, amountPaidOnBuildingClaim=5001.0)])
    c = bd.prepare_claims(raw)
    assert len(c) == 4, "repeated rows must not be removed"
    assert bd.report["rows_identical_to_an_earlier_row_except_id"] == 2


def feature_claims(construction, loss_year=2000, post_firm=False, event=None):
    n = len(construction)
    return pd.DataFrame({
        "originalConstructionDate": construction, "yearOfLoss": [loss_year] * n,
        "elevatedBuildingIndicator": [False] * n, "postFIRMConstructionIndicator": [post_firm] * n,
        "ratedFloodZone": ["AE"] * n, "occupancyType": [1] * n, "countyCode": [12071] * n,
        "floodEvent": [event] * n, "numberOfFloorsInTheInsuredBuilding": [1.0] * n, "primaryResidenceIndicator": [True] * n,
        "totalBuildingInsuranceCoverage": [250000.0] * n, "causeOfDamage": ["4"] * n, "waterDepth": [1.0] * n,
        "elevationDifference": [0.0] * n, "buildingDeductibleCode": ["1"] * n, "foundationType": [1] * n,
    })


def test_unusable_construction_dates_are_flagged_and_filled_for_display():
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
    assert out.loc[out["building_age_missing"] == 1, "building_age_years"].isna().all()
    assert out["building_age_display_years"].notna().all()
    assert out["building_age_display_years"].iloc[0] == 20 and out["building_age_display_years"].iloc[4] == 150
    assert bd.report["construction_year_unknown_pct"] == pytest.approx(100 * 4 / 6, abs=0.01)
    assert bd.report["construction_year_unusable"] == 4
    assert bd.report["construction_date_placeholder_1492"] == 1


def test_display_age_for_unusable_dates_follows_the_loss_year():
    # known pre-FIRM buildings are 10 years old in 1980 and 50 in 2020; unusable dates take the same-year median
    rows = []
    for year, age in ((1980, 10), (2020, 50)):
        rows += [dict(c, yearOfLoss=year) for c in feature_claims([f"{year - age}-06-01T00:00:00.000Z"] * 40).to_dict("records")]
        rows += [dict(c, yearOfLoss=year) for c in feature_claims(["1492-10-12T00:00:00.000Z"] * 3).to_dict("records")]
    out = bd.engineer_features(pd.DataFrame(rows))
    missing = out[out["building_age_missing"] == 1]
    assert missing.groupby("yearOfLoss")["building_age_display_years"].first().to_dict() == {1980: 10, 2020: 50}


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
    assert set(bd.OCCUPANCY_LABELS) == set(g), "every labeled occupancy code needs a group"


def test_flood_zone_groups():
    enc, _ = bd.encode_flood_zone(pd.Series(["X", "B", "AE", "A5", "AO", "AH", "A", "VE", "V", "D", None]))
    assert enc.tolist() == [1, 1, 2, 2, 3, 3, 3, 4, 4, 0, 0]


def test_a99_is_grouped_with_the_special_a_zones():
    enc, _ = bd.encode_flood_zone(pd.Series(["A99"]))
    assert enc.tolist() == [3]


def labeled(events, years):
    c = pd.DataFrame({"floodEvent": events, "yearOfLoss": years})
    return bd.label_events(c)


def test_event_labels_keep_fema_names_and_group_them():
    out = labeled(["Hurricane Ian", "Tropical Storm Fay", "Flooding", "Blizzard of 1993", None, "  ", "April Florida Flooding"],
                   [2022, 2008, 1990, 1993, 2015, 2015, 2023])
    assert out["event"].tolist()[:4] == ["Hurricane Ian", "Tropical Storm Fay", "Flooding", "Blizzard of 1993"]
    assert out["event"].isna().tolist() == [False, False, False, False, True, True, False]
    assert out["event_group"].tolist() == ["Named storm", "Named storm", "Other FEMA-labeled event", "Other FEMA-labeled event",
                                           "No event label", "No event label", "Other FEMA-labeled event"]


def test_georges_is_merged_and_alberto_gets_its_year():
    out = labeled(["Hurricane Georges (Keys)", "Hurricane Georges (Panhandle)", "Tropical Storm Alberto", "Tropical Storm Alberto"],
                   [1998, 1998, 1994, 2006])
    assert out["event"].tolist() == ["Hurricane Georges", "Hurricane Georges", "Tropical Storm Alberto (1994)", "Tropical Storm Alberto (2006)"]
    assert set(out["event_group"]) == {"Named storm"}


def test_event_label_shares_are_recorded_for_the_report():
    labeled(["Hurricane Ian", "Flooding", None, None], [2022, 1990, 2015, 2015])
    assert bd.report["event_label_pct"] == 50.0
    assert bd.report["named_storm_label_pct"] == 25.0


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


def test_floors_are_fema_categories_not_counts():
    c = feature_claims(["1990-01-01T00:00:00.000Z"] * 8)
    c["numberOfFloorsInTheInsuredBuilding"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan, 2.0]
    out = bd.engineer_features(c)
    assert out["floors_group"].tolist() == ["One floor", "Two floors", "Three or more floors", "Split-level",
                                            "Manufactured home or trailer on foundation", "Townhouse, three or more floors",
                                            "Not recorded", "Two floors"]


def test_primary_residence_and_first_flood_map_groups():
    c = feature_claims(["1990-01-01T00:00:00.000Z"] * 3)
    c["primaryResidenceIndicator"] = [True, "True", False]
    c["postFIRMConstructionIndicator"] = [True, False, "True"]
    out = bd.engineer_features(c)
    assert out["primary_residence"].tolist() == ["Primary residence", "Primary residence", "Not a primary residence"]
    assert out["firm_group"].tolist() == ["Built after the first flood map", "Built before the first flood map", "Built after the first flood map"]


def test_coverage_bands_give_the_250k_limit_and_no_coverage_their_own_groups():
    c = feature_claims(["1990-01-01T00:00:00.000Z"] * 8)
    c["totalBuildingInsuranceCoverage"] = [0, 1, 50000, 50001, 150000, 249999, 250000, 500000]
    out = bd.engineer_features(c)
    assert out["coverage_band"].tolist() == ["No building coverage", "Up to $50k", "Up to $50k", "$50k-$100k", "$100k-$150k",
                                             "$150k-$250k", "$250k", "Over $250k"]


def grouped(**columns):
    n = len(next(iter(columns.values())))
    c = feature_claims(["1990-01-01T00:00:00.000Z"] * n)
    for name, values in columns.items():
        c[name] = values
    return bd.engineer_features(c)


def test_cause_of_flooding_uses_fema_codes_and_groups_the_rest():
    out = grouped(causeOfDamage=["4", "1", 2.0, "3", "0", "9", "B", "D", None, "@"])
    assert out["cause_group"].tolist() == ["Accumulation of rainfall or snowmelt", "Tidal water overflow", "Stream, river or lake overflow",
                                           "Alluvial fan overflow", "Other causes"] + ["Other or not recorded"] * 5


def test_flood_water_depth_bands_and_negative_values_count_as_not_recorded():
    out = grouped(waterDepth=[0, 1, 6, 7, 12, 13, 36, 37, 376, -1, np.nan])
    assert out["depth_band"].tolist() == ["0 in", "1-6 in", "1-6 in", "7-12 in", "7-12 in", "13-36 in", "13-36 in", "Over 36 in",
                                          "Over 36 in", "Not recorded", "Not recorded"]


def test_elevation_bands_compare_the_lowest_floor_with_the_base_flood_level():
    out = grouped(elevationDifference=[-5, -1, 0, 1, 2, 3, 40, np.nan])
    assert out["elevation_band"].tolist() == ["Below the base flood level", "Below the base flood level", "At the base flood level",
                                              "1 ft above", "2 ft above", "3 ft or more above", "3 ft or more above", "Not recorded"]


def test_deductible_codes_become_dollar_amounts():
    out = grouped(buildingDeductibleCode=["0", "1", "9", "F", "5", "A", "E", "H", None, "Z"])
    assert out["deductible_group"].tolist() == ["$500", "$1,000", "$750", "$1,250", "$5,000", "$10,000", "$50,000", "$200",
                                                "Not recorded", "Not recorded"]


def test_foundation_type_reads_fema_codes_and_treats_zero_as_not_recorded():
    out = grouped(foundationType=[1, "2", 3.0, 4, 5, 6, 0, None])
    assert out["foundation_group"].tolist() == ["Slab", "Basement", "Crawlspace", "Elevated, no enclosure (post, pile or pier)",
                                                "Elevated with enclosure (post, pile or pier)",
                                                "Elevated with enclosure (not post, pile or pier)", "Not recorded", "Not recorded"]
