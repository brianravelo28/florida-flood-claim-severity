"""FEMA's non-payment-reason fields: how the claims that paid nothing are coded."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from constants import NON_PAYMENT_GROUP_ORDER
from pipeline import build_dataset as bd

DEPLOY = Path(__file__).resolve().parent.parent / "deploy_data"


@pytest.fixture(autouse=True)
def fresh_report():
    bd.report.clear()
    yield
    bd.report.clear()


def raw_with(rows):
    """rows: (loss date, building payment, building reason, contents reason)."""
    base = dict(amountPaidOnContentsClaim=np.nan, amountPaidOnIncreasedCostOfComplianceClaim=np.nan, latitude=27.0, longitude=-81.0)
    return pd.DataFrame([dict(base, dateOfLoss=f"{d}T00:00:00.000Z", amountPaidOnBuildingClaim=p, nonPaymentReasonBuilding=b,
                              nonPaymentReasonContents=ct) for d, p, b, ct in rows])


def test_a_zero_total_is_grouped_by_its_building_reason_else_its_contents_reason():
    raw = raw_with([
        ("2000-05-01", 0.0, "01", np.nan),    # building reason: below the deductible
        ("2000-05-02", 0.0, np.nan, "20"),    # no building reason, contents reason: no damage
        ("2000-05-03", 0.0, "06", "01"),      # both: the building reason wins (not a flood)
        ("2000-05-04", 0.0, "98", np.nan),    # administrative error
        ("2000-05-05", 0.0, "17", np.nan),    # erosion: other reason
        ("2000-05-06", 0.0, np.nan, np.nan),  # nothing recorded
        ("2000-05-07", 500.0, "01", np.nan),  # paid: not part of the zero-total rows
    ])
    bd.prepare_claims(raw)
    out = bd.report["non_payment_reasons"]
    assert out["zero_total"] == {"rows": 6, "with_reason": 5, "pct_with_reason": 83.3}
    assert out["zero_total_by_decade"][2000] == {"Below the deductible": 1, "No damage or not a flood": 2, "Administrative error": 1,
                                                 "Other reason": 1, "No reason recorded": 1}


def test_codes_that_arrive_as_numbers_are_read_as_two_digit_codes():
    raw = raw_with([("2005-05-01", 0.0, 1.0, np.nan), ("2005-05-02", 0.0, 6, np.nan), ("2005-05-03", 0.0, 98.0, np.nan)])
    bd.prepare_claims(raw)
    assert bd.report["non_payment_reasons"]["zero_total_by_decade"][2000] == {
        "Below the deductible": 1, "No damage or not a flood": 1, "Administrative error": 1, "Other reason": 0, "No reason recorded": 0}


def test_negative_totals_are_counted_apart_and_the_newest_years_show_zero_totals_without_a_reason():
    raw = raw_with([("2021-05-01", -20.0, "97", np.nan), ("2021-06-01", -5.0, np.nan, np.nan), ("2026-02-01", 0.0, np.nan, np.nan),
                    ("2026-03-01", 0.0, "01", np.nan), ("2026-04-01", 900.0, np.nan, np.nan), ("1999-05-01", 0.0, np.nan, np.nan)])
    bd.prepare_claims(raw)
    out = bd.report["non_payment_reasons"]
    assert out["negative_total"] == {"rows": 2, "with_reason": 1}
    assert out["since_2020"] == {2021: {"claims": 2, "zero_total_without_reason": 0}, 2026: {"claims": 3, "zero_total_without_reason": 1}}


def test_the_reason_fields_do_not_change_which_rows_count_as_repeats():
    row = dict(amountPaidOnBuildingClaim=5000.0, amountPaidOnContentsClaim=0.0, amountPaidOnIncreasedCostOfComplianceClaim=0.0,
               dateOfLoss="2005-10-24T00:00:00.000Z", latitude=25.1, longitude=-80.4)
    raw = pd.DataFrame([dict(row, id=1, nonPaymentReasonBuilding=np.nan, nonPaymentReasonContents=np.nan),
                        dict(row, id=2, nonPaymentReasonBuilding="97", nonPaymentReasonContents=np.nan)])
    bd.prepare_claims(raw)
    assert bd.report["rows_identical_to_an_earlier_row_except_id"] == 1


# ------------------------------------------------------------ the committed report
def report():
    return json.loads((DEPLOY / "build_report.json").read_text())["non_payment_reasons"]


def test_the_reason_counts_agree_with_the_zero_and_negative_totals():
    full = json.loads((DEPLOY / "build_report.json").read_text())
    out = full["non_payment_reasons"]
    assert out["zero_total"]["rows"] == full["total_paid_zero_or_negative"]["zero"]
    assert out["negative_total"]["rows"] == full["total_paid_zero_or_negative"]["negative"]
    by_decade = out["zero_total_by_decade"]
    assert sum(sum(g.values()) for g in by_decade.values()) == out["zero_total"]["rows"]
    assert sum(g["No reason recorded"] for g in by_decade.values()) == out["zero_total"]["rows"] - out["zero_total"]["with_reason"]
    assert all(list(g) == NON_PAYMENT_GROUP_ORDER for g in by_decade.values())
    assert out["zero_total"]["pct_with_reason"] == round(100 * out["zero_total"]["with_reason"] / out["zero_total"]["rows"], 1)


def test_the_reason_groups_by_decade_fit_inside_the_claims_that_paid_nothing():
    full = json.loads((DEPLOY / "build_report.json").read_text())
    for decade, groups in full["non_payment_reasons"]["zero_total_by_decade"].items():
        assert sum(groups.values()) <= full["closed_without_payment_by_decade"][decade]["without_payment"]


def test_nearly_every_zero_total_carries_a_coded_reason_and_the_unlabelled_ones_cluster_in_the_newest_years():
    out = report()
    assert out["zero_total"]["pct_with_reason"] >= 97
    share = {int(y): v["zero_total_without_reason"] / v["claims"] for y, v in out["since_2020"].items()}
    assert share[2026] > share[2025] > max(share[y] for y in share if y <= 2024)
