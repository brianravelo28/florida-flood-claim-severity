"""Download and cache the real source data: OpenFEMA NFIP claims (Florida), NOAA HURDAT2 and FRED CPI-U.

Everything lands in data/raw/ (gitignored); a re-run reuses the cache unless called with --refresh.
Claims come from OpenFEMA's "NFIP Redacted Claims" v3 (the web service is named NfipClaims). FEMA retired version 2 of the
dataset on 2026-10-15.
"""
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).parent / "data" / "raw"
FEMA_CACHE = RAW_DIR / "fema_nfip_claims_v3_fl.csv.gz"
HURDAT_CACHE = RAW_DIR / "hurdat2.txt"
CPI_CACHE = RAW_DIR / "cpi_u.csv"
CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"

FEMA_URL = "https://www.fema.gov/api/open/v3/NfipClaims"
NHC_DATA_PAGE = "https://www.nhc.noaa.gov/data/"
HEADERS = {"User-Agent": "florida-flood-claim-severity/1.0 (research project; https://github.com/brianravelo28/florida-flood-claim-severity)"}

FEMA_FIELDS = [
    "id", "dateOfLoss", "yearOfLoss", "floodEvent", "latitude", "longitude", "countyCode",
    "nfipCommunityName", "reportedCity", "ratedFloodZone", "floodZoneCurrent",
    "occupancyType", "originalConstructionDate", "elevatedBuildingIndicator",
    "postFIRMConstructionIndicator", "primaryResidenceIndicator",
    "numberOfFloorsInTheInsuredBuilding", "totalBuildingInsuranceCoverage",
    "totalContentsInsuranceCoverage", "buildingPropertyValue",
    "amountPaidOnBuildingClaim", "amountPaidOnContentsClaim",
    "amountPaidOnIncreasedCostOfComplianceClaim",
]
PAGE_SIZE = 10000


def _get(url, params=None, tries=4):
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=120)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            if attempt == tries - 1:
                raise
            print(f"  retry {attempt + 1} after error: {exc}", flush=True)
            time.sleep(3 * (attempt + 1))


def fetch_fema(refresh=False):
    if FEMA_CACHE.exists() and not refresh:
        print(f"FEMA: using cache {FEMA_CACHE}", flush=True)
        return pd.read_csv(FEMA_CACHE, low_memory=False)

    frames, skip = [], 0
    while True:
        params = {
            "$filter": "state eq 'FL'",
            "$select": ",".join(FEMA_FIELDS),
            "$orderby": "id",
            "$top": PAGE_SIZE,
            "$skip": skip,
        }
        payload = _get(FEMA_URL, params).json()
        rows = payload["NfipClaims"]
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        skip += len(rows)
        print(f"FEMA: fetched {skip:,} rows", flush=True)
        if len(rows) < PAGE_SIZE:
            break
        time.sleep(0.3)

    df = pd.concat(frames, ignore_index=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEMA_CACHE, index=False)
    print(f"FEMA: saved {len(df):,} rows to {FEMA_CACHE}", flush=True)
    return df


def fetch_hurdat2(refresh=False):
    if HURDAT_CACHE.exists() and not refresh:
        print(f"HURDAT2: using cache {HURDAT_CACHE}", flush=True)
        return HURDAT_CACHE.read_text()

    page = _get(NHC_DATA_PAGE).text
    matches = re.findall(r"hurdat/hurdat2-1851-\d{4}-\d+\.txt", page)
    if not matches:
        raise RuntimeError("Could not find the Atlantic HURDAT2 file link on " + NHC_DATA_PAGE)
    url = NHC_DATA_PAGE + sorted(set(matches))[-1]
    print(f"HURDAT2: downloading {url}", flush=True)
    text = _get(url).text
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    HURDAT_CACHE.write_text(text)
    print(f"HURDAT2: saved {len(text) // 1024:,} KB", flush=True)
    return text


def fetch_cpi(refresh=False):
    """CPI-U, all items, seasonally adjusted (FRED series CPIAUCSL), monthly."""
    if not CPI_CACHE.exists() or refresh:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        CPI_CACHE.write_text(_get(CPI_URL).text)
        print(f"CPI: saved {CPI_CACHE}", flush=True)
    return pd.read_csv(CPI_CACHE, parse_dates=["observation_date"])


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    fetch_hurdat2(refresh)
    fetch_cpi(refresh)
    fetch_fema(refresh)
