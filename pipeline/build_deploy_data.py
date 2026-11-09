"""Build the compact deploy_data/ bundle the hosted dashboard loads.

Run after build_dataset.py:   python -m pipeline.build_deploy_data

Render's instance has 512 MB of RAM, so every static aggregate is computed here, once, instead of at app startup:

  claims.parquet     ALL paid claims, one compact row each (categoricals + float32); every tab filters and aggregates it live
  aggregates.json    county centroids and the year range
  summary.json       the headline numbers quoted in the README and Methodology (medians by decade and year, sensitivity to the
                     biggest seasons, event-label coverage, top events, building age by decade)
  build_report.json  data-cleaning counts and checks
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from constants import (AGE_BINS, AGE_LABELS, CAUSE_ORDER, COVERAGE_LABELS, DEDUCTIBLE_ORDER, DEPTH_ORDER, ELEVATED_LABELS,
                       ELEVATION_ORDER, EVENT_GROUPS, FIRM_LABELS, FLOOD_ORDER, FLOOR_ORDER, FOUNDATION_ORDER, MIN_CLAIMS, OCC_ORDER,
                       PRIMARY_LABELS)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "deploy_data"

USECOLS = [
    "claimId", "yearOfLoss", "countyName", "latitude", "longitude", "occupancy_group", "flood_zone_group",
    "building_age_years", "building_age_missing", "building_age_display_years", "is_elevated", "amountPaid_real",
    "event", "event_group", "floors_group", "primary_residence", "firm_group", "coverage_band",
    "cause_group", "depth_band", "elevation_band", "deductible_group", "foundation_group",
]


def decade_label(decade, years):
    first, last = int(years.min()), int(years.max())
    return f"{first}-{str(last)[-2:]}" if first // 10 == last // 10 else f"{first}-{last}"


def summarize(df, table, report):
    """Headline numbers, all computed from the same float32 payouts the dashboard uses."""
    pay = table["amountPaid_real"].astype("float64")
    year = table["yearOfLoss"]
    decade = (year // 10) * 10
    out = {"n_claims": int(len(table)), "total_paid": float(pay.sum()), "median_claim": float(pay.median()),
           "year_min": int(year.min()), "year_max": int(year.max())}

    out["by_decade"] = {}
    for d, g in pay.groupby(decade):
        yrs = year[decade == d]
        out["by_decade"][int(d)] = {"label": decade_label(d, yrs), "claims": int(len(g)), "median": float(g.median()),
                                    "total_paid": float(g.sum())}
    out["by_year"] = {int(y): {"claims": int(len(g)), "median": float(g.median())} for y, g in pay.groupby(year)}

    d20 = (year >= 2020)
    ev = table["event"].astype("string")
    big3 = ev.isin(["Hurricane Ian", "Hurricane Helene", "Hurricane Milton"])
    out["sensitivity_2020s"] = {
        "all": {"claims": int(d20.sum()), "median": float(pay[d20].median())},
        "without_2022": {"claims": int((d20 & (year != 2022)).sum()), "median": float(pay[d20 & (year != 2022)].median())},
        "without_2024": {"claims": int((d20 & (year != 2024)).sum()), "median": float(pay[d20 & (year != 2024)].median())},
        "without_2022_and_2024": {"claims": int((d20 & ~year.isin([2022, 2024])).sum()),
                                  "median": float(pay[d20 & ~year.isin([2022, 2024])].median())},
        "without_ian_helene_milton": {"claims": int((d20 & ~big3).sum()), "median": float(pay[d20 & ~big3].median())},
    }
    d10 = (year >= 2010) & (year <= 2019)
    out["median_2010s"] = {"claims": int(d10.sum()), "median": float(pay[d10].median())}
    out["without_big3_vs_2010s_ratio"] = out["sensitivity_2020s"]["without_ian_helene_milton"]["median"] / out["median_2010s"]["median"]

    group = table["event_group"].astype(str)
    out["event_groups"] = {}
    for gname in EVENT_GROUPS:
        m = group == gname
        out["event_groups"][gname] = {"claims": int(m.sum()), "pct_claims": round(100 * float(m.mean()), 1),
                                      "paid": float(pay[m].sum()), "pct_paid": round(100 * float(pay[m].sum() / pay.sum()), 1),
                                      "median": float(pay[m].median())}
    out["event_label_pct"] = round(100 * float(ev.notna().mean()), 1)
    out["named_storm_label_pct"] = out["event_groups"]["Named storm"]["pct_claims"]
    out["named_storm_pct_by_decade"] = {int(d): round(100 * float((group[decade == d] == "Named storm").mean()), 1)
                                        for d in sorted(decade.unique())}
    out["years_without_a_named_storm_claim"] = int(sum(1 for y, g in group.groupby(year) if not (g == "Named storm").any()))

    named = table[group == "Named storm"].assign(pay=pay)
    stats = named.groupby("event", observed=True).agg(claims=("pay", "size"), total=("pay", "sum"), median=("pay", "median"),
                                                      year=("yearOfLoss", "min"))
    ranked = stats.sort_values("claims", ascending=False)
    out["named_storms"] = {"count": int(len(stats)), "under_min_claims": int((stats["claims"] < MIN_CLAIMS).sum())}
    out["top_named_storms_by_claims"] = [
        {"event": str(k), "year": int(r["year"]), "claims": int(r["claims"]), "total_paid": float(r["total"]), "median": float(r["median"])}
        for k, r in ranked.head(10).iterrows()]
    for k in (10, 25):
        top = ranked.head(k)
        out[f"top_{k}_named_storms"] = {"pct_claims": round(100 * float(top["claims"].sum() / len(table)), 1),
                                        "pct_paid": round(100 * float(top["total"].sum() / pay.sum()), 1)}

    known = df[df["building_age_missing"] == 0]
    kdec = (known["yearOfLoss"] // 10) * 10
    out["median_building_age_at_loss_by_decade"] = {int(d): float(g.median()) for d, g in known["building_age_years"].groupby(kdec)}
    out["duplicates_kept"] = report["rows_identical_to_an_earlier_row_except_id"]
    return out


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "insurance_clean.csv.gz", usecols=USECOLS, low_memory=False)
    report = json.loads((DATA / "build_report.json").read_text())

    df["elevated"] = np.where(df["is_elevated"] == 1, "Elevated", "Not elevated")
    # The Building age chart uses the year-aware age for buildings with no usable construction date (build_dataset.py).
    df["age_band"] = pd.cut(df["building_age_display_years"], AGE_BINS, labels=AGE_LABELS).astype(str)

    cty = df.groupby("countyName").agg(lat=("latitude", "mean"), lon=("longitude", "mean"))
    agg = {"n_claims": int(len(df)), "year_min": int(df["yearOfLoss"].min()), "year_max": int(df["yearOfLoss"].max()),
           "counties": {k: {c: float(v) for c, v in row.items()} for k, row in cty.iterrows()}}
    (OUT / "aggregates.json").write_text(json.dumps(agg))

    # One compact row per paid claim: every categorical is a category, every float is float32.
    table = pd.DataFrame({
        "yearOfLoss": df["yearOfLoss"].astype("int16"),
        "countyName": df["countyName"].astype("category"),
        "occupancy_group": pd.Categorical(df["occupancy_group"], categories=OCC_ORDER),
        "flood_zone_group": pd.Categorical(df["flood_zone_group"], categories=FLOOD_ORDER),
        "elevated": pd.Categorical(df["elevated"], categories=ELEVATED_LABELS),
        "age_band": pd.Categorical(df["age_band"], categories=AGE_LABELS),
        "floors_group": pd.Categorical(df["floors_group"], categories=FLOOR_ORDER),
        "primary_residence": pd.Categorical(df["primary_residence"], categories=PRIMARY_LABELS),
        "firm_group": pd.Categorical(df["firm_group"], categories=FIRM_LABELS),
        "coverage_band": pd.Categorical(df["coverage_band"], categories=COVERAGE_LABELS),
        "cause_group": pd.Categorical(df["cause_group"], categories=CAUSE_ORDER),
        "depth_band": pd.Categorical(df["depth_band"], categories=DEPTH_ORDER),
        "elevation_band": pd.Categorical(df["elevation_band"], categories=ELEVATION_ORDER),
        "deductible_group": pd.Categorical(df["deductible_group"], categories=DEDUCTIBLE_ORDER),
        "foundation_group": pd.Categorical(df["foundation_group"], categories=FOUNDATION_ORDER),
        "event": df["event"].astype("category"),
        "event_group": pd.Categorical(df["event_group"], categories=EVENT_GROUPS),
        "amountPaid_real": df["amountPaid_real"].astype("float32"),
    })
    for col in ("occupancy_group", "flood_zone_group", "elevated", "age_band", "event_group", "floors_group", "primary_residence",
                "firm_group", "coverage_band", "cause_group", "depth_band", "elevation_band", "deductible_group", "foundation_group"):
        assert table[col].notna().all(), f"unmapped label in {col}"
    table.to_parquet(OUT / "claims.parquet", compression="snappy", index=False)

    (OUT / "summary.json").write_text(json.dumps(summarize(df, table, report), indent=2))
    shutil.copy(DATA / "build_report.json", OUT / "build_report.json")

    # Stale files from earlier bundles would otherwise be committed and served.
    expected = {"claims.parquet", "aggregates.json", "summary.json", "build_report.json"}
    for f in OUT.iterdir():
        if f.name not in expected:
            f.unlink()

    total = 0.0
    for f in sorted(OUT.iterdir()):
        mb = f.stat().st_size / 1e6
        total += mb
        print(f"{f.name:22s} {mb:6.2f} MB")
    print(f"{'TOTAL':22s} {total:6.2f} MB | claims table {len(table):,} rows x {table.shape[1]} cols, "
          f"~{table.memory_usage(deep=True).sum() / 1e6:.0f} MB in memory")


if __name__ == "__main__":
    main()
