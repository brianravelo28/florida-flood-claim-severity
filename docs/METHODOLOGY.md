# Methodology

Contents: [Terms](#terms) · [Data sources](#data-sources) · [Data and cleaning](#data-and-cleaning) · [Event labels](#event-labels) · [Payouts over time](#payouts-over-time) · [Factors](#factors) · [Reproducibility](#reproducibility) · [Deployment notes](#deployment-notes)

## Terms

| Term | Meaning |
|---|---|
| NFIP | The National Flood Insurance Program, run by FEMA (the Federal Emergency Management Agency). |
| OpenFEMA | FEMA's open data service, where it publishes NFIP claims. |
| CPI-U | The Consumer Price Index for All Urban Consumers, used to put payouts in constant dollars. |
| FIRM | A community's Flood Insurance Rate Map. |
| Post-FIRM | FEMA's label for a building whose construction started after December 31, 1974 or after the community's first FIRM, whichever is later. |
| Base flood elevation | The flood level with a 1% chance of being reached in any year, used to rate a building. |
| Median | The middle value: half of the claims are larger and half are smaller. |
| Percentile | The 25th percentile is the value a quarter of the claims fall below; the 90th is the value nine tenths fall below. |
| PPI | The Producer Price Index, which tracks the prices businesses receive; here, for construction materials. |

## Data sources

| Source | What | Version used |
|---|---|---|
| [OpenFEMA NFIP Redacted Claims](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3) | Florida claims, 1978 to 2026 | v3 (web service `NfipClaims`): 448,618 Florida rows and 28 fields, pulled 2026-11-03. FEMA last refreshed the dataset on 2026-09-09. |
| [FRED CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL) | CPI-U, all items, seasonally adjusted, monthly | Complete back to 1947 except October 2025, which has no published value. |
| [FRED WPUSI012011](https://fred.stlouisfed.org/series/WPUSI012011) | Producer Price Index by Commodity: Construction Materials, monthly | Pulled 2026-11-10; last month 2026-09. Used for the check under Payouts over time. |
| [FRED CES2000000008](https://fred.stlouisfed.org/series/CES2000000008) | Average Hourly Earnings of Production and Nonsupervisory Employees, Construction, monthly | Pulled 2026-11-10; last month 2026-10. Used for the same check. |

OpenFEMA data are published under OpenFEMA's terms of use. This project is not endorsed by FEMA.

**Data versions.** FEMA retired version 2 of the claims dataset on 2026-10-15. A pull of v2 (448,425 Florida rows, frozen 2026-06-01) and the v3 pull differ little: v3 has 193 more rows (mostly recent losses), the same count of placeholder construction dates (19,370), slightly different values in about 0.3% of building payments and 0.1% of contents payments (the largest change is about +$38 million in 2024 payments), and a filled-in community name (blank for 51% of v2 rows). The claim `id` is an integer in v3 and a UUID in v2. Two v3 pulls, 2026-10-20 and 2026-11-03, agree on every id and on every field they share.

## Data and cleaning

**Payout.** A claim's payout is its building, contents and increased-cost-of-compliance payments added together; FEMA has no single total field.

**Exclusions.** `pipeline/build_dataset.py` logs every row it drops and why (`build_report.json`):

| Step | Rows dropped | Remaining |
|---|---|---|
| Raw Florida claims | | 448,618 |
| Missing latitude, longitude or loss date | 2,591 | 446,027 |
| Total paid is zero or negative | 134,415 | 311,612 |
| Loss date before 1978 | 0 | 311,612 |
| Coordinates outside the Florida bounding box | 105 | **311,507** |

About 30% of claims were closed without payment, so the dashboard describes the cost of a claim given that it was paid. 134,378 of the excluded rows have a total of zero and 37 a negative total; FEMA's dictionary notes that a negative amount appears when a check is not cashed and has to be reissued. The 105 claims outside the Florida box have coordinates in other states (many near New York City); FEMA's location blurring explains a move of a few miles, and the cause of these is unknown.

**Location.** FEMA blurs coordinates to 0.1 degree (about 7 miles) and warns that a point can land in the wrong county or state; its county field is the reliable location. The county map uses the average coordinates of each county's claims.

**Repeated rows.** 996 rows are identical to an earlier row on every pulled field except `id`. They are kept: they cluster in multi-unit buildings and condo associations, where several units of one building file claims with the same values, and the data have no building or policy identifier to tell a repeat from a neighbor. Comparing only the original 22 fields (before cause, depth, elevation, deductible and foundation type were added) finds 1,270, so 274 of those apparent repeats are different claims. Removing all 996 moves every decade median by 0.5% at most, and the 2020-26 median by 0.1%.

**Inflation.** Each payout is multiplied by (2025 average CPI-U) / (CPI-U in the claim's loss month), with the 2025 average equal to 322.186. The factors run from 0.96 to 5.14. October 2025 has no published CPI value; it is filled by linear interpolation between September and November, and `build_report.json` records it.

**Construction dates.** 11,719 claims (3.76%) have no usable construction year. 11,466 of them carry FEMA's `1492-10-12` placeholder date, and every one of those buildings is flagged pre-FIRM, so they are probably old buildings whose date is unknown. The other 253 have a date after the loss year, no date, or a year before 1800. Buildings with no usable date take the median age of known-date pre-FIRM buildings from the same loss year (that median was 9 years in the 1970s and 54 in the 2020s as the same construction era aged), so they land in a sensible age band; loss years with fewer than 30 such buildings borrow from the surrounding two years, then from all pre-FIRM claims.

## Event labels

FEMA labels most claims with the event behind them (`floodEvent`). The dashboard uses those labels as FEMA wrote them, with two tidy-ups: Hurricane Georges (1998) was split by FEMA into a Keys and a Panhandle label and is one event here, and the name Tropical Storm Alberto covers two storms (1994 and 2006), so it carries its year. Each claim falls in one of three groups:

| Group | Claims | Share of claims | Total paid | Share of dollars | Median claim |
|---|---|---|---|---|---|
| Named storm | 236,318 | 75.9% | $21.19B | 89.1% | $40,002 |
| Other FEMA-labeled event | 38,792 | 12.5% | $1.49B | 6.3% | $16,310 |
| No event label | 36,397 | 11.7% | $1.11B | 4.7% | $11,128 |

"Named storm" means the label starts with Hurricane or Tropical Storm: 45 storms. The 11 other labels include "Flooding" (17,538 claims with loss years from 1979 to 2000, so a label FEMA used for many floods), the Blizzard of 1993 and several rain and nor'easter events. Storm names carry their loss year on the Storms tab.

The share of claims with a label rises from 55.2% in 1978-79 to 97.5% in 2020-26, and the share naming a hurricane or tropical storm from 0.0% to 93.9%:

| Decade of loss | 1978-79 | 1980-89 | 1990-99 | 2000-09 | 2010-19 | 2020-26 |
|---|---|---|---|---|---|---|
| Claims with an event label | 55.2% | 78.8% | 81.4% | 86.6% | 80.5% | 97.5% |
| Claims naming a hurricane or tropical storm | 0.0% | 44.8% | 62.7% | 67.3% | 75.2% | 93.9% |

26 of the 49 loss years have no named-storm claim, and a claim with no label can still be storm-related. The ten largest named storms by claims account for 57.3% of all claims and 76.0% of all dollars, and the largest 25 for 74.1% and 88.1%.

## Payouts over time

Median paid claim in constant 2025 dollars, by decade of loss:

| Loss years | Claims | Median claim | Total paid |
|---|---|---|---|
| 1978-79 | 4,077 | $4,792 | $79.4M |
| 1980-89 | 17,625 | $8,381 | $367.1M |
| 1990-99 | 61,324 | $13,988 | $2,462.7M |
| 2000-09 | 63,944 | $19,187 | $3,836.9M |
| 2010-19 | 39,944 | $28,084 | $2,496.8M |
| 2020-26 | 124,593 | $77,944 | $14,547.5M |

By single year, the median was $15,180 in 2015 (1,180 claims), $76,699 in 2022 (39,736 claims) and $89,057 in 2024 (68,076 claims). The first decade covers two loss years, and the last covers seven. The 2020s median ($77,944) reflects a few very large seasons, which the table below separates:

| 2020-26 claims included | Claims | Median claim |
|---|---|---|
| All | 124,593 | $77,944 |
| Without 2022 (Ian) | 84,857 | $78,506 |
| Without 2024 (Helene, Milton) | 56,517 | $64,592 |
| Without 2022 and 2024 | 16,781 | $45,693 |
| Without Ian, Helene and Milton claims | 21,262 | $42,529 |

The 2010s median is $28,084 (39,944 claims) and the 2020s median without the three largest storms is $42,529 (21,262 claims). Earlier decades shift the same way when their biggest years are set aside: the 1990s median is $13,988 and $18,880 without 1995 and 1999, and the 2000s median is $19,187 and $12,849 without 2004 and 2005.

**Claims that paid nothing.** Among claims with a loss date and location, the share that paid nothing was 42.5% in 1978-79, between 35.9% and 36.7% in each decade from the 1980s to the 2010s, and 17.8% in 2020-26. The paid claims in each decade, with the 25th, 75th and 90th percentiles beside the median:

| Loss years | Share that paid nothing | 25th percentile | Median | 75th percentile | 90th percentile |
|---|---|---|---|---|---|
| 1978-79 | 42.5% | $1,780 | $4,792 | $15,226 | $45,540 |
| 1980-89 | 36.2% | $2,858 | $8,381 | $24,055 | $54,269 |
| 1990-99 | 36.6% | $4,402 | $13,988 | $43,793 | $95,266 |
| 2000-09 | 35.9% | $6,502 | $19,187 | $60,288 | $148,845 |
| 2010-19 | 36.7% | $8,667 | $28,084 | $77,426 | $154,795 |
| 2020-26 | 17.8% | $19,503 | $77,944 | $152,269 | $255,192 |

Each of the three percentiles rises in every decade, so the rise in the median is not confined to one end of the distribution.

Claims from one storm share a flood, so the intervals below treat each storm as one unit: a claim's FEMA event label, with the claims that have no label forming one group per loss year, resampled 1,000 times, and the range holding the middle 95% of the resampled medians.

| Loss years | Claims | Storm groups | Median claim | 95% interval |
|---|---|---|---|---|
| 1978-79 | 4,077 | 4 | $4,792 | $3,204 to $8,629 |
| 1980-89 | 17,625 | 14 | $8,381 | $4,472 to $13,439 |
| 1990-99 | 61,324 | 21 | $13,988 | $8,203 to $25,247 |
| 2000-09 | 63,944 | 27 | $19,187 | $13,837 to $30,006 |
| 2010-19 | 39,944 | 17 | $28,084 | $20,191 to $32,607 |
| 2020-26 | 124,593 | 25 | $77,944 | $38,630 to $95,685 |
| 2020-26 without Ian, Helene and Milton | 21,262 | 22 | $42,529 | $26,053 to $59,803 |

The 2020-26 interval ($38,630 to $95,685) lies above the 2010-19 interval ($20,191 to $32,607). The interval without Ian, Helene and Milton ($26,053 to $59,803) overlaps the 2010-19 interval.

The ten named storms with the most claims:

| Storm | Claims | Total paid | Median claim |
|---|---|---|---|
| Hurricane Helene (2024) | 49,037 | $6.60B | $106,695 |
| Hurricane Ian (2022) | 38,111 | $5.29B | $80,157 |
| Hurricane Irma (2017) | 18,544 | $1.29B | $30,822 |
| Hurricane Milton (2024) | 16,183 | $1.21B | $46,251 |
| Hurricane Irene (1999) | 13,680 | $0.23B | $6,277 |
| Hurricane Ivan (2004) | 10,556 | $1.64B | $75,625 |
| Hurricane Wilma (2005) | 9,434 | $0.58B | $28,521 |
| Hurricane Opal (1995) | 9,022 | $0.81B | $32,076 |
| Hurricane Elena (1985) | 7,438 | $0.23B | $16,441 |
| Hurricane Josephine (1996) | 6,495 | $0.21B | $21,030 |

CPI-U puts claims from different years on one price level. Payouts have grown much faster than that: the data show the growth, and possible reasons include storm severity, larger coverage limits, changes in claims handling and a different mix of insured buildings.

**Other price indexes.** CPI-U measures the prices consumers pay. The same payments were also adjusted to 2025 dollars with two construction-cost series, by loss month and with the 2025 average as the base, as for CPI-U: the producer price index for construction materials and average hourly earnings of production and nonsupervisory employees in construction. The last row averages the two adjusted payouts.

| Price index | 1978-79 median | 2010-19 median | 2020-26 median | 2020-26 relative to 1978-79 | 2020-26 without Ian, Helene and Milton, relative to 2010-19 |
|---|---|---|---|---|---|
| CPI-U (used throughout) | $4,792 | $28,084 | $77,944 | 16.3 times | +51% |
| Construction materials prices | $4,138 | $32,629 | $76,854 | 18.6 times | +33% |
| Construction wages | $4,592 | $30,004 | $79,727 | 17.4 times | +46% |
| Average of the two | $4,362 | $31,343 | $78,503 | 18.0 times | +39% |

Under every index the 2020-26 median is 16.3 to 18.6 times the 1978-79 median, and the 2020s median without the three storms is 33% to 51% above the 2010s median.

## Factors

The Factors tab compares payouts across 13 factors. Each shows a box for the middle 50% of claims with whiskers at the 10th and 90th percentiles, and a table of claims, share of claims, median, mean and middle 50%. Codes are labeled from the OpenFEMA data dictionary for `NfipClaims`.

| Factor | FEMA field | Groups |
|---|---|---|
| **Building age** | `originalConstructionDate and the loss year` | 0-10 yrs; 11-20; 21-30; 31-40; 41-50; 51-60; 60+ |
| **Building coverage** | `totalBuildingInsuranceCoverage` | No building coverage; Up to $50k; $50k-$100k; $100k-$150k; $150k-$250k; $250k; Over $250k |
| **Building deductible** | `buildingDeductibleCode` | $200; $500; $750; $1,000; $1,250; $1,500; $2,000; $3,000; $4,000; $5,000; $10,000; $15,000; $20,000; $25,000; $50,000; Not recorded |
| **Building floors** | `numberOfFloorsInTheInsuredBuilding` | One floor; Two floors; Three or more floors; Split-level; Manufactured home or trailer on foundation; Townhouse, three or more floors; Not recorded |
| **Built after first flood map** | `postFIRMConstructionIndicator` | Built before the first flood map; Built after the first flood map |
| **Cause of flooding** | `causeOfDamage` | Accumulation of rainfall or snowmelt; Tidal water overflow; Stream, river or lake overflow; Alluvial fan overflow; Other causes; Other or not recorded |
| **Elevated building** | `elevatedBuildingIndicator` | Not elevated; Elevated |
| **Elevation vs base flood level** | `elevationDifference` | Below the base flood level; At the base flood level; 1 ft above; 2 ft above; 3 ft or more above; Not recorded |
| **Flood water depth** | `waterDepth` | 0 in; 1-6 in; 7-12 in; 13-36 in; Over 36 in; Not recorded |
| **Flood zone** | `ratedFloodZone` | X / B / C (moderate-minimal); A / AH / AO / AR / A99 (no base elevation, shallow, special); AE / A1-A30 (base elevation known); V / VE (velocity, coastal); D / unknown |
| **Foundation type** | `foundationType` | Slab; Basement; Crawlspace; Elevated, no enclosure (post, pile or pier); Elevated with enclosure (post, pile or pier); Elevated with enclosure (not post, pile or pier); Not recorded |
| **Occupancy type** | `occupancyType` | Single-family; 2-4 units; 5+ units; Mobile home; Condo association; Unit in multi-unit building; Non-residential; Unknown |
| **Primary residence** | `primaryResidenceIndicator` | Primary residence; Not a primary residence |

- **Flood zone.** Five groups follow FEMA's `ratedFloodZone` definitions: moderate-minimal (X, B, C), A-type zones without a base elevation (A, AH, AO, AR, A99), zones with a base elevation (AE, A1-A30), coastal high-velocity (V, VE), and D or unknown.
- **Occupancy type.** FEMA's legacy codes (1-4, 6) and newer codes (11-19) describe the same kinds of buildings and are merged into eight groups.
- **Building floors.** FEMA's field is a code: 1 one floor, 2 two floors, 3 three or more, 4 split-level, 5 manufactured home or trailer on a foundation, 6 townhouse with three or more floors.
- **Foundation type.** Codes 1 to 6 are FEMA's six types. Code 0 is not in FEMA's list; it covers 95-100% of claims before 2010, so it is shown as Not recorded.
- **Elevation vs base flood level.** Feet between the building's lowest floor, as used for rating, and the base flood elevation: below, at, or 1, 2 or 3 or more feet above.
- **Building age.** Age at the time of loss, in bands of ten years, with the construction-date handling described above.
- **Built after first flood map.** FEMA's post-FIRM indicator.
- **Building coverage.** The policy's building limit. 39.9% of claims sit at exactly $250,000, the most common limit, and 2.6% have no building coverage, so each has a band of its own.
- **Building deductible.** FEMA's deductible codes, shown as dollar amounts ($500 is code 0, $1,000 code 1, $5,000 code 5, and so on).
- **Cause of flooding.** Codes 1 to 4 and 0 (tidal water overflow, stream, river or lake overflow, alluvial fan overflow, accumulation of rainfall or snowmelt, other causes). The other codes FEMA lists (erosion, earth movement, closed-basin lake, claim handling) and values outside its list are grouped as Other or not recorded.
- **Flood water depth.** Inches, in bands. FEMA notes that some measurements were recorded in feet, and negative values count as Not recorded.

**Fields recorded for part of the claims.** The share of claims in the Not recorded group varies by decade:

| Field | Overall | 1978-79 | 1980-89 | 1990-99 | 2000-09 | 2010-19 | 2020-26 |
|---|---|---|---|---|---|---|---|
| Foundation type | 54.8% | 100% | 100% | 99% | 95% | 61% | 3% |
| Elevation vs base flood level | 60.9% | 85% | 91% | 62% | 51% | 54% | 63% |
| Flood water depth | 8.0% | 11% | 4% | 9% | 7% | 11% | 7% |
| Building deductible | 4.2% | 95% | 25% | 0% | 1% | 1% | 3% |

The Loss Years filter narrows any of these to the years where the field is filled in; for example, foundation type is recorded for 97% of 2020-26 claims.

**Building age and loss year.** The median building age at loss (buildings with known dates) was 15 years in the 1980s, 22 in the 1990s, 28 in the 2000s, 38 in the 2010s and 50 in the 2020s, and payouts rose over the same period. The Building age view therefore also draws one line per decade of loss, so a building's age is compared with others of the same era. The first-flood-map factor overlaps with building age for the same reason.

**Elevated buildings within groups.** The elevated median is below the not-elevated median in all 24 combinations of decade (1990s to 2020s) and water-depth band that have 30 or more claims on both sides. Within flood zones it is lower in 13 of 16 combinations, including every zone in the 2010s and 2020s; it is higher in X / B / C and A / AH / AO / AR / A99 zones in the 1990s and in X / B / C zones in the 2000s. For 2020-26:

| Flood zone | Elevated claims | Elevated median | Not elevated claims | Not elevated median |
|---|---|---|---|---|
| X / B / C (moderate-minimal) | 467 | $20,461 | 6,835 | $40,036 |
| A / AH / AO / AR / A99 (no base elevation, shallow, special) | 337 | $17,690 | 4,120 | $55,445 |
| AE / A1-A30 (base elevation known) | 18,975 | $15,355 | 88,788 | $98,228 |
| V / VE (velocity, coastal) | 1,707 | $27,177 | 3,357 | $100,141 |

**Small groups.** A claim count under 30 carries an asterisk in every table and in the year chart's hover text. The charts draw groups with 30 or more claims, and the by-decade building-age lines use points with 30 or more claims.

## Reproducibility

Run order (details in the README): `pipeline/fetch_data.py`, `pipeline/build_dataset.py`, `pipeline/build_deploy_data.py`. The pipeline is deterministic: the same pull gives the same cleaned dataset and the same `deploy_data/`.

**Data version.** The raw pull (about 16 MB) is kept outside the repository, and a new pull returns FEMA's current data. The SHA-256 of the cached pull behind these numbers, `data/raw/fema_nfip_claims_v3_fl.csv.gz` (pulled 2026-11-03), is `e0a913f3bc8d0ae41e59a34122c2701d2897312dd43bdfbccb7bd0dd4e941bc8`. The earlier 2026-10-20 pull, whose 23 fields all agree with it, is not the source of record.

**Software.** Python 3.14 with pandas 3.1.0, NumPy 2.5.4, PyArrow 26.0.0, Requests 2.34.2, Plotly 7.1.0 and Dash 4.5.0; CI runs the tests on Python 3.13 and 3.14. The cleaned dataset is identical on the earlier pinned versions (pandas 2.0.3, NumPy 1.24.3) and on these.

**Tests.** `tests/` holds more than 80 tests on small hand-built examples and on the committed bundle: the exclusion counts and payout sum, the inflation adjustment and its construction-cost alternatives, the storm-level intervals, the event labels, the grouping of every factor, the consistency of `deploy_data/`, and every dashboard callback, including empty filters.

## Deployment notes

The hosted dashboard runs on Render's Starter plan (512 MB of RAM, kept awake). The service installs `requirements-render.txt` and serves `wsgi.py` with gunicorn 26.2.0 (one worker, four threads); `docs/render.yaml` records these settings, and Render's dashboard controls the live service, with its settings taking precedence over the file.

The app holds all 311,507 paid claims as one compact table (4.0 MB on disk) and filters it live; the headline numbers are precomputed in `summary.json`. On Render the process uses about 160 MB after loading the app and about 205 MB after the three tabs have been opened (`/debugz` reports the live figure).

`wsgi.py` answers `/healthz` immediately and imports the dashboard in a background thread started on the first request in each worker process (a thread started when the module is imported exists only in gunicorn's master process, so the worker would never finish loading). `/debugz` reports whether the app has loaded, its memory use and any startup error.
