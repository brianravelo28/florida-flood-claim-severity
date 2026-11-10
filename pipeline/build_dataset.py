"""Build the analysis dataset from real OpenFEMA NFIP claims: clean, adjust for inflation, group and label events."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from constants import (CAUSE_LABELS, COVERAGE_BINS, COVERAGE_LABELS, DEDUCTIBLE_LABELS, DEPTH_BINS, DEPTH_LABELS, ELEVATION_LABELS,
                       BIG_THREE_STORMS, EVENT_MERGES, EVENT_NEEDS_YEAR, FIRM_LABELS, FLOOR_LABELS, FOUNDATION_LABELS, NAMED_STORM_PATTERN,
                       NO_REASON, NON_PAYMENT_FIELDS, NON_PAYMENT_GROUP_ORDER, NON_PAYMENT_REASON_GROUPS, OTHER_REASON, PRIMARY_LABELS)
from pipeline.fetch_data import fetch_cpi, fetch_fema, fetch_price_indexes

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "data" / "insurance_clean.csv.gz"
OUT_REPORT = ROOT / "data" / "build_report.json"

# Florida incl. the panhandle (a 30.5N / -87.5W box would cut off Pensacola and the northern border)
FL_LAT, FL_LON = (24.3, 31.1), (-87.7, -79.8)

FL_COUNTIES = {
    12001: "Alachua", 12003: "Baker", 12005: "Bay", 12007: "Bradford", 12009: "Brevard",
    12011: "Broward", 12013: "Calhoun", 12015: "Charlotte", 12017: "Citrus", 12019: "Clay",
    12021: "Collier", 12023: "Columbia", 12027: "DeSoto", 12029: "Dixie", 12031: "Duval",
    12033: "Escambia", 12035: "Flagler", 12037: "Franklin", 12039: "Gadsden", 12041: "Gilchrist",
    12043: "Glades", 12045: "Gulf", 12047: "Hamilton", 12049: "Hardee", 12051: "Hendry",
    12053: "Hernando", 12055: "Highlands", 12057: "Hillsborough", 12059: "Holmes",
    12061: "Indian River", 12063: "Jackson", 12065: "Jefferson", 12067: "Lafayette", 12069: "Lake",
    12071: "Lee", 12073: "Leon", 12075: "Levy", 12077: "Liberty", 12079: "Madison", 12081: "Manatee",
    12083: "Marion", 12085: "Martin", 12086: "Miami-Dade", 12087: "Monroe", 12089: "Nassau",
    12091: "Okaloosa", 12093: "Okeechobee", 12095: "Orange", 12097: "Osceola", 12099: "Palm Beach",
    12101: "Pasco", 12103: "Pinellas", 12105: "Polk", 12107: "Putnam", 12109: "St. Johns",
    12111: "St. Lucie", 12113: "Santa Rosa", 12115: "Sarasota", 12117: "Seminole", 12119: "Sumter",
    12121: "Suwannee", 12123: "Taylor", 12125: "Union", 12127: "Volusia", 12129: "Wakulla",
    12131: "Walton", 12133: "Washington",
}

CPI_BASE_YEAR = 2025

# Verified against FEMA's official data dictionary (OpenFemaDataSetFields, NfipClaims, occupancyType).
# FEMA notes that 2-digit codes are for Risk Rating 2.0 policies, yet they appear in claims from 1985 on.
OCCUPANCY_LABELS = {
    0: "Unknown", 1: "Single-family residence", 2: "2-4 unit residential building",
    3: "Residential building, more than 4 units", 4: "Non-residential building", 6: "Non-residential - business",
    11: "Single-family residential building", 12: "Residential non-condo building, 2-4 units (all units insured)",
    13: "Residential non-condo building, 5+ units (all units insured)", 14: "Residential mobile/manufactured home",
    15: "Residential condo association (building coverage)", 16: "Single residential unit in a multi-unit building",
    17: "Non-residential mobile/manufactured home", 18: "Non-residential building",
    19: "Non-residential unit in a multi-unit building",
}
# Legacy (1-4, 6) and newer (11-19) codes describe the same kinds of buildings; grouping them removes the era effect.
OCCUPANCY_GROUPS = {
    "Unknown": (0,), "Single-family": (1, 11), "2-4 units": (2, 12), "5+ units": (3, 13), "Mobile home": (14,),
    "Condo association": (15,), "Unit in multi-unit building": (16,), "Non-residential": (4, 6, 17, 18, 19),
}
OCCUPANCY_GROUP_CODE = {name: i for i, name in enumerate(OCCUPANCY_GROUPS)}
OCCUPANCY_TO_GROUP = {code: name for name, codes in OCCUPANCY_GROUPS.items() for code in codes}

report = {}


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- claims
def without_payment_by_decade(c):
    """Claims with a loss date and location, and how many of them paid nothing (zero or negative total), by decade of loss."""
    decade = (c["dateOfLoss"].dt.year // 10) * 10
    unpaid = c["amountPaid"] <= 0
    claims, none_paid = unpaid.groupby(decade).size(), unpaid.groupby(decade).sum()
    return {int(d): {"claims": int(claims[d]), "without_payment": int(none_paid[d]), "pct": round(100 * float(none_paid[d] / claims[d]), 1)}
            for d in claims.index}


def non_payment_reasons(c):
    """How FEMA codes the claims that paid nothing: its non-payment-reason field for the building, else for the contents.

    Counts, for claims with a zero total and for claims with a negative total, how many carry a coded reason; the zero-total
    claims by decade of loss and reason group; and, for each loss year since 2020, the claims with a zero total and no reason.
    """
    code = _code(c["nonPaymentReasonBuilding"]).fillna(_code(c["nonPaymentReasonContents"])).str.zfill(2)
    lookup = {k: g for g, codes in NON_PAYMENT_REASON_GROUPS.items() for k in codes}
    group = code.map(lookup).fillna(OTHER_REASON).where(code.notna(), NO_REASON)
    zero, negative = c["amountPaid"] == 0, c["amountPaid"] < 0
    decade = (c["dateOfLoss"].dt.year // 10) * 10
    year = c["dateOfLoss"].dt.year
    out = {
        "zero_total": {"rows": int(zero.sum()), "with_reason": int((zero & code.notna()).sum())},
        "negative_total": {"rows": int(negative.sum()), "with_reason": int((negative & code.notna()).sum())},
        "zero_total_by_decade": {int(d): {g: int(((decade == d) & zero & (group == g)).sum()) for g in NON_PAYMENT_GROUP_ORDER}
                                 for d in sorted(decade.unique())},
        "since_2020": {int(y): {"claims": int((year == y).sum()), "zero_total_without_reason": int(((year == y) & zero & code.isna()).sum())}
                       for y in sorted(year[year >= 2020].unique())},
    }
    rows = out["zero_total"]["rows"]
    out["zero_total"]["pct_with_reason"] = round(100 * out["zero_total"]["with_reason"] / rows, 1) if rows else None
    return out


def prepare_claims(raw):
    c = raw.copy()
    log(f"Raw FEMA claims: {len(c):,}")
    report["raw_rows"] = len(c)

    parts = ["amountPaidOnBuildingClaim", "amountPaidOnContentsClaim", "amountPaidOnIncreasedCostOfComplianceClaim"]
    c["amountPaid"] = c[parts].fillna(0).sum(axis=1)
    c["dateOfLoss"] = pd.to_datetime(c["dateOfLoss"].str[:10], errors="coerce")

    steps = {}
    n = len(c)
    c = c.dropna(subset=["latitude", "longitude", "dateOfLoss"])
    steps["missing lat/lon/date"] = n - len(c)
    report["missing_latlon_pct"] = round(100 * steps["missing lat/lon/date"] / n, 3)

    n = len(c)
    report["closed_without_payment_by_decade"] = without_payment_by_decade(c)
    report["total_paid_zero_or_negative"] = {"zero": int((c["amountPaid"] == 0).sum()), "negative": int((c["amountPaid"] < 0).sum())}
    report["non_payment_reasons"] = non_payment_reasons(c)
    c = c[c["amountPaid"] > 0]
    steps["amountPaid <= 0 (closed without payment)"] = n - len(c)

    n = len(c)
    c = c[c["dateOfLoss"] >= "1978-01-01"]
    steps["dateOfLoss before 1978"] = n - len(c)

    n = len(c)
    inside = c["latitude"].between(*FL_LAT) & c["longitude"].between(*FL_LON)
    c = c[inside]
    steps["outside Florida bounds"] = n - len(c)

    for k, v in steps.items():
        log(f"  dropped {v:>8,}  {k}")
    report["rows_dropped"] = steps
    log(f"Claims after cleaning: {len(c):,}")

    # Rows identical to an earlier row on every field except id. They are kept: they cluster in multi-unit buildings and
    # condo associations, where several units of one building file claims with the same values, and the data have no
    # building or policy identifier to tell a repeat from a neighbour.
    fields = [col for col in raw.columns if col != "id" and col not in NON_PAYMENT_FIELDS]  # the 996 figure predates the reason fields
    report["rows_identical_to_an_earlier_row_except_id"] = int(c.duplicated(subset=fields).sum())
    return c.reset_index(drop=True)


# ---------------------------------------------------------------- features
def display_building_age(c, min_n=30):
    """Building age in years, with a year-aware value for buildings that have no usable construction date.

    Rows with no usable construction date (building_age_missing == 1) get the median age of known-date pre-FIRM claims
    from the SAME loss year: FEMA's placeholder date marks buildings that are all flagged pre-FIRM (built before the
    community's flood map), and "pre-FIRM" is not a fixed age (the median pre-FIRM building was about 9 years old at
    loss in the 1970s and about 54 in the 2020s, as the same construction era gets older). Loss years with fewer than
    `min_n` known-date pre-FIRM claims borrow from the surrounding +/-2 years, then from all pre-FIRM claims.
    """
    known_prefirm = c[(c["building_age_missing"] == 0) & (c["post_firm"] == 0)]
    by_year = known_prefirm.groupby("yearOfLoss")["building_age_years"].agg(["median", "size"])
    overall_median = known_prefirm["building_age_years"].median()

    def lookup(yr):
        if yr in by_year.index and by_year.loc[yr, "size"] >= min_n:
            return by_year.loc[yr, "median"]
        window = by_year.loc[by_year.index.to_series().between(yr - 2, yr + 2)]
        return np.average(window["median"], weights=window["size"]) if window["size"].sum() >= min_n else overall_median

    year_value = {yr: lookup(yr) for yr in c["yearOfLoss"].unique()}
    age = c["building_age_years"].copy()
    ph = c["building_age_missing"] == 1
    age[ph] = c.loc[ph, "yearOfLoss"].map(year_value).round()
    return age.astype(int)


def encode_flood_zone(z):
    """Groups follow FEMA's ratedFloodZone definitions; later rules override earlier (overlapping) ones."""
    z = z.fillna("UNKNOWN").str.upper().str.strip()
    enc = np.zeros(len(z), dtype=int)
    label = np.full(len(z), "D / unknown", dtype=object)
    for mask, code, name in [
        (z.isin(["X", "B", "C"]), 1, "X / B / C (moderate-minimal)"),
        (z.str.startswith("A"), 3, "A / AH / AO / AR / A99 (no base elevation, shallow, special)"),
        (z.str.match(r"^AE$|^A(?!99$)\d{1,2}$"), 2, "AE / A1-A30 (base elevation known)"),
        (z.str.startswith("V"), 4, "V / VE (velocity, coastal)"),
    ]:
        enc[mask.to_numpy()] = code
        label[mask.to_numpy()] = name
    return enc, label


def _code(series):
    """A coded field as clean text ('1', 'A'): numeric codes can arrive as 1, 1.0 or '1' depending on the column's other values."""
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def label_factors(c):
    """Readable groups for the fields behind the Factors tab, using FEMA's own code definitions."""
    c["cause_group"] = _code(c["causeOfDamage"]).map(CAUSE_LABELS).fillna("Other or not recorded")
    depth = pd.to_numeric(c["waterDepth"], errors="coerce")
    c["depth_band"] = pd.cut(depth.where(depth >= 0), DEPTH_BINS, labels=DEPTH_LABELS).astype("string").fillna("Not recorded")
    elev = pd.to_numeric(c["elevationDifference"], errors="coerce")
    c["elevation_band"] = np.select([elev.isna(), elev < 0, elev == 0, elev == 1, elev == 2],
                                    ["Not recorded"] + ELEVATION_LABELS[:4], default=ELEVATION_LABELS[4])
    c["deductible_group"] = _code(c["buildingDeductibleCode"]).map(DEDUCTIBLE_LABELS).fillna("Not recorded")
    c["foundation_group"] = _code(c["foundationType"]).map(FOUNDATION_LABELS).fillna("Not recorded")
    return c


def label_events(c):
    """FEMA's own event label (floodEvent), tidied: Georges is one storm that FEMA split by region, and Alberto is two
    different storms (1994 and 2006) with one name, so it gets its year. Every other label is kept as FEMA wrote it."""
    event = c["floodEvent"].astype("string").str.strip()
    event = event.where(event != "", pd.NA).replace(EVENT_MERGES)
    with_year = event + " (" + c["yearOfLoss"].astype(str) + ")"
    event = event.where(~event.isin(EVENT_NEEDS_YEAR), with_year)
    named = event.str.contains(NAMED_STORM_PATTERN, case=False, na=False)
    c["event"] = event
    c["event_group"] = np.where(event.isna(), "No event label", np.where(named, "Named storm", "Other FEMA-labeled event"))
    report["event_label_pct"] = round(100 * float(event.notna().mean()), 1)
    report["named_storm_label_pct"] = round(100 * float(named.mean()), 1)
    return c


def engineer_features(c):
    yr = pd.to_numeric(c["originalConstructionDate"].str[:4], errors="coerce")  # pandas dates overflow before 1677
    bad = ~((yr >= 1800) & (yr <= c["yearOfLoss"]))
    report["construction_year_unknown_pct"] = round(100 * bad.mean(), 2)
    report["construction_year_unusable"] = int(bad.sum())
    report["construction_date_placeholder_1492"] = int(c["originalConstructionDate"].astype(str).str.startswith("1492").sum())
    yr = yr.where(~bad)
    c["yearOfConstruction"] = yr
    c["building_age_missing"] = bad.astype(int)
    c["building_age_years"] = (c["yearOfLoss"] - yr).clip(0, 200)  # NaN where the construction date is unusable

    c["is_elevated"] = c["elevatedBuildingIndicator"].astype(str).str.lower().eq("true").astype(int)
    # Construction started after the community's initial FIRM (or after 1974-12-31, whichever is later). Complete for
    # every claim, unlike the construction date.
    c["post_firm"] = c["postFIRMConstructionIndicator"].astype(str).str.lower().eq("true").astype(int)
    c["building_age_display_years"] = display_building_age(c)
    c["firm_group"] = np.where(c["post_firm"] == 1, FIRM_LABELS[1], FIRM_LABELS[0])
    c["floors_group"] = c["numberOfFloorsInTheInsuredBuilding"].map(FLOOR_LABELS).fillna("Not recorded")
    c["primary_residence"] = np.where(c["primaryResidenceIndicator"].astype(str).str.lower().eq("true"), PRIMARY_LABELS[0], PRIMARY_LABELS[1])
    c["coverage_band"] = pd.cut(c["totalBuildingInsuranceCoverage"], COVERAGE_BINS, labels=COVERAGE_LABELS).astype(str)
    c["flood_zone_encoded"], c["flood_zone_group"] = encode_flood_zone(c["ratedFloodZone"])
    c["occupancy_encoded"] = c["occupancyType"].fillna(0).astype(int)
    c["occupancy_label"] = c["occupancy_encoded"].map(OCCUPANCY_LABELS)
    c["occupancy_group"] = c["occupancy_encoded"].map(OCCUPANCY_TO_GROUP)
    c["countyName"] = c["countyCode"].astype("Int64").map(FL_COUNTIES).fillna("Unknown")
    return label_events(label_factors(c))


def add_real_dollars(c):
    """Deflate payouts to constant CPI_BASE_YEAR dollars using the CPI-U value for each claim's month of loss."""
    cpi = fetch_cpi()
    gaps = cpi.loc[cpi["CPIAUCSL"].isna(), "observation_date"].dt.strftime("%Y-%m").tolist()
    if gaps:  # FRED has no value for 2025-10; interpolate between the neighbouring months
        cpi["CPIAUCSL"] = cpi["CPIAUCSL"].interpolate(limit_area="inside")
        log(f"CPI: no published value for {gaps}; filled by linear interpolation")
        report["cpi_interpolated_months"] = gaps
    base_rows = cpi[cpi["observation_date"].dt.year == CPI_BASE_YEAR]["CPIAUCSL"]
    assert len(base_rows) == 12, f"need 12 months of {CPI_BASE_YEAR} CPI, have {len(base_rows)}"
    monthly = dict(zip(cpi["observation_date"].dt.to_period("M"), cpi["CPIAUCSL"]))
    at_loss = c["dateOfLoss"].dt.to_period("M").map(monthly)
    assert at_loss.notna().all(), "CPI missing for some loss months"
    c["cpi_factor"] = base_rows.mean() / at_loss
    c["amountPaid_real"] = c["amountPaid"] * c["cpi_factor"]
    report["cpi_base"] = f"{CPI_BASE_YEAR} average CPI-U (FRED CPIAUCSL) = {base_rows.mean():.3f}"
    log(f"Inflation-adjusted to {CPI_BASE_YEAR} dollars: factor ranges {c['cpi_factor'].min():.2f} to {c['cpi_factor'].max():.2f}")
    return c


def monthly_index(frame):
    """A FRED monthly series as a Series indexed by month; a gap inside the series is filled by linear interpolation."""
    value = pd.to_numeric(frame[frame.columns[1]], errors="coerce").interpolate(limit_area="inside")
    return pd.Series(value.to_numpy(), index=frame["observation_date"].dt.to_period("M"))


def deflate_with(series, c):
    """Payouts in CPI_BASE_YEAR dollars using another monthly price index (same method as add_real_dollars)."""
    base = series[series.index.year == CPI_BASE_YEAR]
    assert len(base) == 12 and base.notna().all(), f"need 12 months of {CPI_BASE_YEAR} values"
    at_loss = c["dateOfLoss"].dt.to_period("M").map(series)
    assert at_loss.notna().all(), "price index missing for some loss months"
    return c["amountPaid"] * base.mean() / at_loss


def deflator_sensitivity(c, indexes):
    """Median payout by decade when the same payments are adjusted with construction costs instead of CPI-U.

    The blend is the average of the payout adjusted with materials prices and the payout adjusted with construction wages.
    Also gives the 2020s median without the three largest storms and its ratio to the 2010s median.
    """
    pay = {"cpi": c["amountPaid_real"]}
    for key in ("materials", "wages"):
        pay[key] = deflate_with(monthly_index(indexes[key]), c)
    pay["blend"] = (pay["materials"] + pay["wages"]) / 2
    labels = {"cpi": "CPI-U (as published)", "materials": "Construction materials prices", "wages": "Construction wages",
              "blend": "Average of materials prices and wages"}
    decade = (c["yearOfLoss"] // 10) * 10
    in_2020s = c["yearOfLoss"] >= 2020
    big3 = c["event"].astype("string").isin(BIG_THREE_STORMS).fillna(False).astype(bool)
    out = {}
    for key, p in pay.items():
        by_decade = {int(d): float(g.median()) for d, g in p.groupby(decade)}
        without = float(p[in_2020s & ~big3].median())
        out[key] = {"label": labels[key], "median_by_decade": by_decade, "median_2020s_without_ian_helene_milton": without,
                    "ratio_to_2010s": without / by_decade[2010]}
    return out


# ---------------------------------------------------------------- checks
def validate(c):
    log("\nValidation")
    checks = {
        "dateOfLoss >= 1978": bool((c["dateOfLoss"] >= "1978-01-01").all()),
        "building age in [0, 200]": bool(c["building_age_display_years"].between(0, 200).all()),
        "lat/lon inside Florida bounds": bool(c["latitude"].between(*FL_LAT).all() and c["longitude"].between(*FL_LON).all()),
        "amountPaid > 0": bool((c["amountPaid"] > 0).all()),
        "amountPaid_real > 0": bool((c["amountPaid_real"] > 0).all()),
        "every claim has a flood zone group and an occupancy group": bool(c[["flood_zone_group", "occupancy_group"]].notna().all().all()),
        "every claim has a coverage band": bool(c["coverage_band"].isin(COVERAGE_LABELS).all()),
        "all present county codes are valid FL FIPS": bool(c.loc[c["countyCode"].notna(), "countyName"].ne("Unknown").all()),
        "missing lat/lon < 2%": report["missing_latlon_pct"] < 2,
        "every claim has an event group": bool(c["event_group"].notna().all()),
    }
    for k, v in checks.items():
        log(f"  [{'PASS' if v else 'FAIL'}] {k}")
    report["checks"] = checks


def main():
    raw = fetch_fema()
    claims = prepare_claims(raw)
    claims = engineer_features(claims)
    claims = add_real_dollars(claims)
    prices = fetch_price_indexes()
    report["deflator_sensitivity"] = {
        "series": {"materials": "FRED WPUSI012011 (PPI, construction materials)",
                   "wages": "FRED CES2000000008 (average hourly earnings, construction)",
                   "last_month": {k: str(monthly_index(v).index.max()) for k, v in prices.items()}},
        "results": deflator_sensitivity(claims, prices)}
    validate(claims)

    cols = ["id", "dateOfLoss", "yearOfLoss", "latitude", "longitude", "countyCode", "countyName", "ratedFloodZone",
            "flood_zone_group", "occupancyType", "occupancy_label", "occupancy_group", "yearOfConstruction",
            "elevatedBuildingIndicator", "is_elevated", "post_firm", "amountPaidOnBuildingClaim", "amountPaidOnContentsClaim",
            "amountPaidOnIncreasedCostOfComplianceClaim", "amountPaid", "cpi_factor", "amountPaid_real",
            "building_age_years", "building_age_missing", "building_age_display_years", "floors_group", "primary_residence",
            "firm_group", "totalBuildingInsuranceCoverage", "coverage_band", "causeOfDamage", "cause_group", "waterDepth", "depth_band",
            "elevationDifference", "elevation_band", "buildingDeductibleCode", "deductible_group", "foundationType", "foundation_group",
            "floodEvent", "event", "event_group"]
    out = claims[[k for k in cols if k in claims]].rename(columns={"id": "claimId"})
    out["dateOfLoss"] = out["dateOfLoss"].dt.strftime("%Y-%m-%d")
    OUT_CSV.parent.mkdir(exist_ok=True)
    out.round(2).to_csv(OUT_CSV, index=False)
    report["final_rows"], report["final_columns"] = len(out), out.shape[1]
    OUT_REPORT.write_text(json.dumps(report, indent=2, default=str))
    log(f"\nSaved {OUT_CSV} ({OUT_CSV.stat().st_size / 1e6:.1f} MB): {len(out):,} rows x {out.shape[1]} columns")


if __name__ == "__main__":
    main()
