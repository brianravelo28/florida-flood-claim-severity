"""Build the analysis dataset from real OpenFEMA NFIP claims and NOAA HURDAT2 storm tracks."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from constants import WIND_BREAKS_KT
from fetch_data import fetch_cpi, fetch_fema, fetch_hurdat2

ROOT = Path(__file__).parent
OUT_CSV = ROOT / "data" / "insurance_clean.csv.gz"
OUT_REPORT = ROOT / "data" / "build_report.json"

EARTH_MI = 3959.0
# Tuned against FEMA's own floodEvent storm labels: 50 mi / +/-30 d gave 67% correct
# storms and reached only ~half of named-storm claims; 150 mi / +/-7 d gives ~99.8% correct and ~96% reach.
RADIUS_MI = 150.0
WINDOW_H = 7 * 24
FREQ_RADIUS_MI = 50.0
FREQ_YEARS = 40
TRACK_START_YEAR = 1978 - FREQ_YEARS
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

CATEGORY_LABELS = {-1: "No storm", 0: "Tropical Depression", 1: "Tropical Storm",
                   2: "Category 1", 3: "Category 2", 4: "Category 3", 5: "Category 4", 6: "Category 5"}

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


# ---------------------------------------------------------------- HURDAT2
_HEADER = re.compile(r"^[A-Z]{2}\d{6}$")


def parse_hurdat2(text):
    recs, cur, repaired, skipped = [], None, 0, 0
    for line in text.splitlines():
        fixed = re.sub(r"([NS])\s+(\d)", r"\1,\2", line)  # a few records are missing the lat/lon comma
        repaired += fixed != line
        p = [x.strip() for x in fixed.split(",")]
        if _HEADER.match(p[0]):
            cur = (p[0], p[1].title(), int(p[0][4:8]))
        elif cur is not None and len(p) >= 8:
            if p[4][-1:] not in ("N", "S") or p[5][-1:] not in ("E", "W"):  # e.g. "38.83" with no hemisphere letter
                skipped += 1
                continue
            lat =float(p[4][:-1]) * (1 if p[4][-1] == "N" else -1)
            lon = float(p[5][:-1]) * (-1 if p[5][-1] == "W" else 1)
            recs.append((cur[0], cur[1], cur[2], p[0] + p[1], lat, lon, int(p[6]), int(p[7])))
    log(f"HURDAT2: repaired {repaired} and skipped {skipped} malformed track line(s)")
    df = pd.DataFrame(recs, columns=["storm_id", "storm_name", "storm_year", "stamp", "lat", "lon", "wind", "pressure"])
    df["time"] = pd.to_datetime(df["stamp"], format="%Y%m%d%H%M")
    df["wind"] = df["wind"].where(df["wind"] >= 0)
    df["pressure"] = df["pressure"].where(df["pressure"] > 0)
    return df.drop(columns="stamp")


def interpolate_tracks(df):
    """Linearly interpolate each storm to hourly positions so 6-hourly gaps (~100 mi) don't hide near-misses."""
    epoch = pd.Timestamp("1970-01-01")
    parts = []
    for sid, g in df.groupby("storm_id", sort=False):
        g = g.sort_values("time")
        h = ((g["time"] - epoch).dt.total_seconds() / 3600).to_numpy()
        grid = np.arange(np.ceil(h[0]), np.floor(h[-1]) + 1)
        if len(grid) == 0:
            continue
        ok = g["wind"].notna().to_numpy()
        wind = np.interp(grid, h[ok], g["wind"].to_numpy()[ok]) if ok.any() else np.full(len(grid), np.nan)
        parts.append(pd.DataFrame({
            "storm_id": sid, "storm_name": g["storm_name"].iloc[0], "storm_year": g["storm_year"].iloc[0],
            "hour": grid, "lat": np.interp(grid, h, g["lat"].to_numpy()),
            "lon": np.interp(grid, h, g["lon"].to_numpy()), "wind": wind,
        }))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- claims
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
    return c.reset_index(drop=True)


# ---------------------------------------------------------------- spatial-temporal join
def join_storms(claims, tracks):
    pts = tracks[tracks["lat"].between(21.5, 34.0) & tracks["lon"].between(-91.0, -76.5)].reset_index(drop=True)
    log(f"Hourly track points near Florida: {len(pts):,} from {pts['storm_id'].nunique()} storms")
    report["track_points"] = len(pts)
    tree = BallTree(np.radians(pts[["lat", "lon"]].to_numpy()), metric="haversine")

    claims["loc_id"] = claims.groupby(["latitude", "longitude"]).ngroup()
    locs = claims.drop_duplicates("loc_id").sort_values("loc_id")[["latitude", "longitude"]].to_numpy()
    log(f"Unique claim locations: {len(locs):,}")
    ind, dist = tree.query_radius(np.radians(locs), r=RADIUS_MI / EARTH_MI, return_distance=True)

    t_h = pts["hour"].to_numpy()
    sid = pts["storm_id"].astype("category").cat.codes.to_numpy()
    syear = pts["storm_year"].to_numpy()
    claim_h = ((claims["dateOfLoss"] - pd.Timestamp("1970-01-01")).dt.total_seconds() / 3600).to_numpy()
    claim_year = claims["yearOfLoss"].to_numpy()

    match_idx = np.full(len(claims), -1, dtype=np.int64)
    match_dist = np.full(len(claims), np.nan)
    freq = np.zeros(len(claims), dtype=np.int64)

    for loc, rows in claims.groupby("loc_id").indices.items():
        ii = ind[loc]
        if len(ii) == 0:
            continue
        o = np.argsort(t_h[ii])
        ii, dd = ii[o], dist[loc][o] * EARTH_MI
        tt = t_h[ii]

        uh, inv = np.unique(claim_h[rows], return_inverse=True)
        lo = np.searchsorted(tt, uh - WINDOW_H, "left")
        hi = np.searchsorted(tt, uh + WINDOW_H, "right")
        best = np.full(len(uh), -1, dtype=np.int64)
        bestd = np.full(len(uh), np.nan)
        for k in range(len(uh)):
            if hi[k] > lo[k]:
                j = lo[k] + dd[lo[k]:hi[k]].argmin()
                best[k], bestd[k] = ii[j], dd[j]
        match_idx[rows] = best[inv]
        match_dist[rows] = bestd[inv]

        near = dd <= FREQ_RADIUS_MI
        _, first = np.unique(sid[ii][near], return_index=True)
        years = np.sort(syear[ii][near][first])
        y = claim_year[rows]
        freq[rows] = np.searchsorted(years, y, "left") - np.searchsorted(years, y - FREQ_YEARS, "left")

    hit = match_idx >= 0
    claims["distance_from_track_mi"] = match_dist
    claims["storm_id"] = np.where(hit, pts["storm_id"].to_numpy()[match_idx], None)
    claims["nearest_storm_name"] = np.where(hit, pts["storm_name"].to_numpy()[match_idx], None)
    claims["storm_wind_speed_kt"] = np.where(hit, pts["wind"].to_numpy()[match_idx], np.nan)
    claims["days_to_storm"] = np.where(hit, np.round((t_h[match_idx] - claim_h) / 24), np.nan)
    claims["historical_storm_freq"] = freq
    log(f"Claims matched to a storm (<= {RADIUS_MI:.0f} mi, +/-{WINDOW_H // 24} days): {hit.sum():,} ({100 * hit.mean():.1f}%)")
    report["storm_matched_pct"] = round(100 * hit.mean(), 2)
    return claims.drop(columns="loc_id")


# ---------------------------------------------------------------- features
def wind_to_category(w):
    w = np.asarray(w, dtype=float)
    return np.select([np.isnan(w)] + [w < b for b in WIND_BREAKS_KT], [-1, 0, 1, 2, 3, 4, 5], default=6).astype(int)


def display_building_age(c, min_n=30):
    """Building age for DISPLAY/aggregation only (e.g. the Risk factors 'Building age' chart) -- NOT a model feature.

    Rows with no usable construction date (building_age_missing==1: 11,466 are FEMA's 1492 sentinel, all flagged
    pre-FIRM, plus 253 other bad dates) get the median age of known-date pre-FIRM
    claims from the SAME loss year, instead of the flat overall median: "pre-FIRM" is not a fixed age (it ages from
    ~9 years old in the 1970s to ~55 by the 2020s as the same construction era gets older), and 92% of these rows are
    themselves 1970s-80s losses. Tested in ablation_placeholder_age.py: this makes no difference to the model (R2
    0.200 vs 0.201, within fold-to-fold noise), so the model keeps the flat-median building_age_years column; this
    display-only version just avoids lumping ~11,700 claims from every era into one misleading age bucket. (It also
    applies the pre-FIRM age to the 253 non-sentinel rows, 0.08% of claims, some of which are really new buildings.)
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
    age[ph] = c.loc[ph, "yearOfLoss"].map(year_value).round().astype(int)
    return age


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


def engineer_features(c):
    yr = pd.to_numeric(c["originalConstructionDate"].str[:4], errors="coerce")  # pandas dates overflow before 1677
    bad = ~((yr >= 1800) & (yr <= c["yearOfLoss"]))
    report["construction_year_unknown_pct"] = round(100 * bad.mean(), 2)
    yr = yr.where(~bad)
    c["yearOfConstruction"] = yr
    age = (c["yearOfLoss"] - yr).clip(0, 200)
    c["building_age_missing"] = age.isna().astype(int)
    c["building_age_years"] = age.fillna(age.median()).astype(int)

    c["is_elevated"] = c["elevatedBuildingIndicator"].astype(str).str.lower().eq("true").astype(int)
    # Construction started after the community's initial FIRM (or after 1974-12-31, whichever is later). Complete for
    # every claim, unlike the construction date.
    c["post_firm"] = c["postFIRMConstructionIndicator"].astype(str).str.lower().eq("true").astype(int)
    c["building_age_display_years"] = display_building_age(c)
    c["flood_zone_encoded"], c["flood_zone_group"] = encode_flood_zone(c["ratedFloodZone"])
    c["occupancy_encoded"] = c["occupancyType"].fillna(0).astype(int)
    c["occupancy_label"] = c["occupancy_encoded"].map(OCCUPANCY_LABELS)
    c["occupancy_group"] = c["occupancy_encoded"].map(OCCUPANCY_TO_GROUP)
    c["occupancy_group_encoded"] = c["occupancy_group"].map(OCCUPANCY_GROUP_CODE)

    d = c["distance_from_track_mi"]
    c["distance_bin"] = np.select([d.isna(), d <= 25, d <= 50, d <= 100], [-1, 4, 3, 2], default=1)
    c["storm_category_encoded"] = wind_to_category(c["storm_wind_speed_kt"])
    c["storm_category"] = c["storm_category_encoded"].map(CATEGORY_LABELS)
    c["log_amountpaid"] = np.log1p(c["amountPaid"])
    c["countyName"] = c["countyCode"].astype("Int64").map(FL_COUNTIES).fillna("Unknown")
    return c


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
    c["log_amountpaid_real"] = np.log1p(c["amountPaid_real"])
    report["cpi_base"] = f"{CPI_BASE_YEAR} average CPI-U (FRED CPIAUCSL) = {base_rows.mean():.3f}"
    log(f"Inflation-adjusted to {CPI_BASE_YEAR} dollars: factor ranges {c['cpi_factor'].min():.2f} to {c['cpi_factor'].max():.2f}")
    return c


# ---------------------------------------------------------------- checks
def validate(c):
    log("\nValidation")
    checks = {
        "dateOfLoss >= 1978": bool((c["dateOfLoss"] >= "1978-01-01").all()),
        "building_age in [0, 200]": bool(c["building_age_years"].between(0, 200).all()),
        "lat/lon inside Florida bounds": bool(c["latitude"].between(*FL_LAT).all() and c["longitude"].between(*FL_LON).all()),
        "amountPaid > 0": bool((c["amountPaid"] > 0).all()),
        "log_amountpaid complete": bool(c["log_amountpaid"].notna().all()),
        "encoded categoricals complete": bool(c[["flood_zone_encoded", "occupancy_encoded", "distance_bin",
                                                 "storm_category_encoded"]].notna().all().all()),
        "all present county codes are valid FL FIPS": bool(c.loc[c["countyCode"].notna(), "countyName"].ne("Unknown").all()),
        "missing lat/lon < 2%": report["missing_latlon_pct"] < 2,
        "historical_storm_freq is non-negative": bool((c["historical_storm_freq"] >= 0).all()),
    }
    log(f"  historical_storm_freq range: {c['historical_storm_freq'].min()}-{c['historical_storm_freq'].max()}")
    for k, v in checks.items():
        log(f"  [{'PASS' if v else 'FAIL'}] {k}")
    raw_skew, log_skew = c["amountPaid"].skew(), c["log_amountpaid"].skew()
    log(f"  target skewness: raw {raw_skew:.2f}, log {log_skew:.2f}, "
        f"log of inflation-adjusted {c['log_amountpaid_real'].skew():.2f}")
    report["checks"] = checks
    report["skew_raw"], report["skew_log"] = round(float(raw_skew), 3), round(float(log_skew), 3)


def check_against_flood_event(c, events):
    """Compare our HURDAT2 match with FEMA's own storm label (floodEvent). Not used as a feature."""
    e = events.rename("floodEvent")
    named = e.str.contains(r"^(?:Hurricane|Tropical Storm)\b", case=False, na=False)
    if named.sum() == 0:
        return
    got_storm = c.loc[named, "storm_id"].notna()
    want = e[named].str.replace(r"^(?:Hurricane|Tropical Storm)\s+", "", regex=True).str.lower().str.strip()
    same = [(isinstance(n, str) and w.startswith(n.lower())) for n, w in zip(c.loc[named, "nearest_storm_name"], want)]
    report["flood_event_check"] = {
        "claims_labelled_hurricane_or_TS_by_FEMA": int(named.sum()),
        "matched_to_any_storm_pct": round(100 * float(got_storm.mean()), 1),
        "matched_to_same_named_storm_pct": round(100 * float(np.mean(same)), 1),
    }
    log(f"\nAgainst FEMA floodEvent labels ({named.sum():,} claims FEMA calls a named hurricane/tropical storm):")
    log(f"  we matched some storm: {100 * got_storm.mean():.1f}% | the same-named storm: {100 * np.mean(same):.1f}%")


def main():
    raw = fetch_fema()
    tracks = interpolate_tracks(parse_hurdat2(fetch_hurdat2()).query("storm_year >= @TRACK_START_YEAR"))

    claims = prepare_claims(raw)
    claims = join_storms(claims, tracks)
    claims = engineer_features(claims)
    claims = add_real_dollars(claims)
    validate(claims)
    if "floodEvent" in claims:
        check_against_flood_event(claims, claims["floodEvent"])

    cols = ["id", "dateOfLoss", "yearOfLoss", "latitude", "longitude", "countyCode", "countyName", "nfipCommunityName",
            "ratedFloodZone", "flood_zone_group", "occupancyType", "yearOfConstruction", "elevatedBuildingIndicator",
            "amountPaidOnBuildingClaim", "amountPaidOnContentsClaim", "amountPaidOnIncreasedCostOfComplianceClaim",
            "amountPaid", "distance_from_track_mi", "storm_id", "nearest_storm_name", "storm_wind_speed_kt",
            "storm_category", "days_to_storm", "building_age_years", "building_age_missing", "is_elevated",
            "flood_zone_encoded", "occupancy_encoded", "distance_bin", "storm_category_encoded", "post_firm",
            "historical_storm_freq", "log_amountpaid", "building_age_display_years", "occupancy_label", "occupancy_group",
            "occupancy_group_encoded", "cpi_factor", "amountPaid_real", "log_amountpaid_real", "floodEvent"]
    out = claims[[k for k in cols if k in claims]].rename(columns={"id": "claimId"})
    out["dateOfLoss"] = out["dateOfLoss"].dt.strftime("%Y-%m-%d")
    out.round(2).to_csv(OUT_CSV, index=False)
    report["final_rows"], report["final_columns"] = len(out), out.shape[1]
    OUT_REPORT.write_text(json.dumps(report, indent=2, default=str))
    log(f"\nSaved {OUT_CSV} ({OUT_CSV.stat().st_size / 1e6:.1f} MB): {len(out):,} rows x {out.shape[1]} columns")


if __name__ == "__main__":
    main()
