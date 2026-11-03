"""The committed deploy_data/ bundle is internally consistent and agrees with the numbers the docs quote."""
import json
from pathlib import Path

import pandas as pd
import pytest

DEPLOY = Path(__file__).resolve().parent.parent / "deploy_data"


def load(name):
    return json.loads((DEPLOY / name).read_text())


@pytest.fixture(scope="module")
def claims():
    return pd.read_parquet(DEPLOY / "claims.parquet")


def test_the_bundle_holds_only_the_files_the_app_reads():
    assert {f.name for f in DEPLOY.iterdir()} == {"claims.parquet", "aggregates.json", "summary.json", "build_report.json"}


def test_claim_counts_agree_across_files(claims):
    report, agg, summary = load("build_report.json"), load("aggregates.json"), load("summary.json")
    assert len(claims) == report["final_rows"] == agg["n_claims"] == summary["n_claims"]
    assert report["raw_rows"] - sum(report["rows_dropped"].values()) == report["final_rows"]


def test_claims_table_has_no_gaps_where_the_app_needs_values(claims):
    for col in ("occupancy_group", "flood_zone_group", "elevated", "age_band", "event_group", "yearOfLoss", "amountPaid_real",
                "floors_group", "primary_residence", "firm_group", "coverage_band"):
        assert claims[col].notna().all(), col
    assert (claims["amountPaid_real"] > 0).all()
    agg = load("aggregates.json")
    assert claims["yearOfLoss"].min() == agg["year_min"] and claims["yearOfLoss"].max() == agg["year_max"]


def test_data_validation_checks_all_passed():
    checks = load("build_report.json")["checks"]
    assert checks and all(checks.values()), [k for k, v in checks.items() if not v]


def test_event_groups_partition_the_claims_and_match_the_report(claims):
    summary, report = load("summary.json"), load("build_report.json")
    groups = summary["event_groups"]
    assert sum(g["claims"] for g in groups.values()) == len(claims)
    assert claims["event_group"].astype(str).value_counts().to_dict() == {k: v["claims"] for k, v in groups.items()}
    assert summary["event_label_pct"] == report["event_label_pct"]
    assert summary["named_storm_label_pct"] == report["named_storm_label_pct"]
    assert (claims["event"].notna() == (claims["event_group"].astype(str) != "No event label")).all()


def test_decade_and_year_summaries_match_the_claims_table(claims):
    summary = load("summary.json")
    assert sum(d["claims"] for d in summary["by_decade"].values()) == len(claims)
    assert sum(y["claims"] for y in summary["by_year"].values()) == len(claims)
    decade = (claims["yearOfLoss"] // 10) * 10
    for d, row in summary["by_decade"].items():
        sel = claims.loc[decade == int(d), "amountPaid_real"].astype("float64")
        assert len(sel) == row["claims"]
        assert sel.median() == pytest.approx(row["median"])
    assert summary["median_claim"] == pytest.approx(claims["amountPaid_real"].astype("float64").median())


def test_the_big_seasons_sensitivity_uses_subsets_of_the_2020s(claims):
    s = load("summary.json")["sensitivity_2020s"]
    assert s["all"]["claims"] == int((claims["yearOfLoss"] >= 2020).sum())
    assert all(v["claims"] < s["all"]["claims"] for k, v in s.items() if k != "all")


def test_top_storms_are_ranked_and_named_storms(claims):
    top = load("summary.json")["top_named_storms_by_claims"]
    assert [t["claims"] for t in top] == sorted((t["claims"] for t in top), reverse=True)
    named = claims.loc[claims["event_group"].astype(str) == "Named storm", "event"].astype(str)
    assert all(t["event"] in set(named) for t in top)
    counts = named.value_counts()
    assert all(counts[t["event"]] == t["claims"] for t in top)


def test_event_labels_follow_the_tidying_rules(claims):
    events = set(claims["event"].dropna().astype(str))
    assert "Hurricane Georges" in events and not any("Georges (" in e for e in events)
    assert "Tropical Storm Alberto (1994)" in events and "Tropical Storm Alberto (2006)" in events
    assert "Tropical Storm Alberto" not in events


def test_the_risk_factor_columns_cover_every_claim(claims):
    from constants import FACTOR_COLUMNS
    for name, (col, order) in FACTOR_COLUMNS.items():
        assert col in claims.columns, name
        assert set(claims[col].astype(str)) <= set(order), name
    assert claims["coverage_band"].astype(str).value_counts().idxmax() == "$250k"
