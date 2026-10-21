# Methodology

How the numbers in the README and the dashboard are produced, what they support, and what they do not. Every figure here comes from a file the pipeline writes (`data/build_report.json`, `data/model_metrics.json`, `data/year_control_model.json`, `data/ablation_*.json`) or from the cleaned dataset it builds.

**Terms used throughout.** NFIP is the National Flood Insurance Program; FEMA (Federal Emergency Management Agency) runs it and publishes its claims through OpenFEMA. NOAA (National Oceanic and Atmospheric Administration) publishes HURDAT2, the Atlantic hurricane database of storm positions and wind speeds every six hours. CPI-U is the Consumer Price Index for All Urban Consumers, used to put payouts in constant dollars. FIRM is a community's Flood Insurance Rate Map; a building is "post-FIRM" if construction started after the community's first FIRM (or after 1974-12-31, whichever is later). LightGBM is a gradient-boosted decision-tree library. R² is the share of variance a model explains (0 is no better than predicting the average, 1 is perfect, below 0 is worse than the average).

Contents: [Data sources](#data-sources) · [Data and cleaning](#data-and-cleaning) · [Storm join](#storm-join) · [Model](#model) · [Evaluation](#evaluation) · [Payouts over time](#payouts-over-time) · [Is building age a stand-in for the loss year?](#is-building-age-a-stand-in-for-the-loss-year) · [Claim Estimator](#claim-estimator) · [Limitations](#limitations) · [Reproducibility](#reproducibility) · [Deployment notes](#deployment-notes)

## Data sources

| Source | What | Version used |
|---|---|---|
| [OpenFEMA NFIP Redacted Claims](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3) | Florida claims, 1978 to 2026 | v3 (web service `NfipClaims`), pulled 2026-10-20: 448,618 Florida rows. FEMA last refreshed the dataset on 2026-09-09. |
| [NOAA HURDAT2](https://www.nhc.noaa.gov/data/) | Atlantic storm tracks | The 1851-2025 release current at the time of the pull. The filename changes with each release, so the fetcher looks it up on the NHC data page. |
| [FRED CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL) | CPI-U, all items, seasonally adjusted, monthly | Series complete back to 1947 except October 2025 (see below). |

Claims data are published by FEMA under OpenFEMA's terms of use. Version 2 of the claims dataset was retired by FEMA on 2026-10-15; see [Reproducibility](#reproducibility) for how the two versions compare.

## Data and cleaning

**Payout.** Each claim's payout is the sum of the three paid fields: building, contents and increased cost of compliance. FEMA has no single total field.

**Exclusions.** `build_dataset.py` logs every row it drops and why (`data/build_report.json`):

| Step | Rows dropped | Remaining |
|---|---|---|
| Raw Florida claims | | 448,618 |
| Missing latitude, longitude or loss date | 2,591 | 446,027 |
| Total paid is zero or negative (closed without payment) | 134,415 | 311,612 |
| Loss date before 1978 | 0 | 311,612 |
| Coordinates outside the Florida bounding box | 105 | **311,507** |

About 30% of claims were closed without payment, so everything here describes severity given a paid claim. It says nothing about whether a claim is paid or how many claims a storm produces. The 105 out-of-box claims have coordinates in other states (many near New York City); FEMA's location blurring cannot explain a move that large, and the cause is unknown.

**Location.** FEMA blurs coordinates to 0.1 degree (about 7 miles) and warns that a point can land in the wrong county. Distances to a storm track are therefore approximate.

**Flood zones and occupancy.** Flood zones are grouped into five classes following FEMA's `ratedFloodZone` definitions (moderate-minimal, A-type without a base elevation, AE/A1-A30, V/VE coastal, and D/unknown). Two occupancy code schemes coexist in the data (1-4, 6 and 11-19) and describe the same kinds of buildings, so they are merged into eight groups; otherwise the code would act as a proxy for the era.

**Inflation.** Each payout is multiplied by (2025 average CPI-U) / (CPI-U in the claim's loss month), with the 2025 average equal to 322.186. Factors run from 0.96 to 5.14. The CPI series has no published value for October 2025; that month is filled by linear interpolation between September and November, and `build_report.json` records it.

**Unusable construction dates.** 11,719 claims (3.76%) have no usable construction year. Of these, 11,466 carry FEMA's `1492-10-12` placeholder date; every one of them is flagged pre-FIRM, so they are probably old buildings whose date is unknown, not random gaps. The other 253 have a date after the loss year, no date, or a year before 1800; only 21% of these are pre-FIRM. For the model, the building age of these claims is the overall median age, plus a `building_age_missing` flag. For the dashboard's Building age chart only, they take the median age of known-date pre-FIRM claims from the same loss year, so that 11.7k claims from every era do not land in one age bucket. Using that year-aware fill as a model feature made no difference (R² 0.198 vs 0.199).

## Storm join

1. NOAA's six-hourly positions are linearly interpolated to hourly positions, because six hours at hurricane speed is about 100 miles, enough to hide a near miss.
2. For every claim location, a haversine `BallTree` finds all track points within **150 miles**; the claim takes the closest of those within **±7 days** of its loss date. This gives the storm's name and id, its wind speed at that point (`storm_wind_speed_kt`), the distance (`distance_from_track_mi`) and the days between loss and closest approach (`days_to_storm`).
3. 77.8% of claims (242k) match a storm; the other 22.2% have no track point within 150 miles and 7 days.
4. `historical_storm_freq` counts distinct storms that passed within 50 miles of the claim's location in the 40 years strictly before the loss year, so a claim's own storm season is never counted.

**How the radius and window were chosen.** FEMA labels some claims with a storm name (`floodEvent`). Using those labels as a reference, a 50-mile, ±30-day join assigned the correct storm to 67% of the claims it matched and reached only about half of the claims FEMA labels as a named storm. At 150 miles and ±7 days, 95.8% of the 236,318 claims FEMA labels as a hurricane or tropical storm match some storm, and 95.6% match the same-named storm (about 99.8% of the matches).

**What that check supports.** The radius and window were chosen using the same labels the 95.6% is computed against, so the figure shows the join recovers FEMA's labelled storm, not that the join is right for every claim. It says nothing about the 22.2% unmatched claims or about unlabelled claims; what those claims are (rain floods, unnamed systems, other causes) is not established here.

## Model

- **Target:** natural log of (1 + payout in constant 2025 dollars).
- **Algorithm:** LightGBM, **L1 (median) objective**, 250 rounds, learning rate 0.05, 31 leaves, minimum 200 claims per leaf, L2 regularization 10, 80% feature and row sampling, seed 42. The L1 objective scored slightly better than L2 on the same folds and predicts a typical (median) payout, not one pulled up by outliers. The output is a median, so it cannot be multiplied by claim counts to estimate a total loss.
- **13 features:** building age, a flag for unusable construction date, elevated building, flood-zone group, occupancy group, storm category, storm wind speed, distance from the storm track (as miles and as a five-level band), days to the storm, storms nearby in the prior 40 years, latitude and longitude. The distance band repeats the distance in miles, and the storm category repeats the wind speed; they are kept as they are, and dropping them is untested.
- **Not features:** damage amounts only exist once a claim does. Coverage limits (`totalBuildingInsuranceCoverage` and the contents equivalent) are complete for every paid claim and are set before a loss, but they cap what NFIP can pay, and the model is built to ask what storm, location and building characteristics explain. About 3.7% of building payments equal the building coverage limit exactly, and 11.3% of contents payments equal the contents limit. A model with the limits as inputs is untested.
- **Importance** (mean absolute SHAP value, share of total): building age 27.5%, storm wind speed 19.0%, elevated 11.0%, latitude 10.7%, occupancy 7.6%, longitude 5.1%, storms nearby 4.6%, distance 4.6%, storm category 4.2%, flood zone 2.3%, missing-age flag 2.1%, days to storm 0.8%, distance band 0.4%. Importance is split among correlated features and shows what the model uses, not what causes payouts.

## Evaluation

**Cross-validation by loss year.** The headline score uses 5-fold cross-validation with all claims from a loss year kept in the same fold, so every claim is predicted by a model that never saw its year. Claims from one storm are close to duplicates (same weather, same year, same neighborhoods), so a random split lets the model memorize storms: it scores R² 0.41, against 0.20 by year. The random split is reported only for contrast.

| Measure | Result |
|---|---|
| R² on the log payout, year-grouped | **0.199** |
| R² of predicting the training average, same folds | −0.049 |
| Median absolute percentage error | 73.9% (half of the predictions are within 74% of the actual payout) |
| Random 80/20 split R² (leaks storms) | 0.413 |
| Train 1978-2019, test 2020-2026: model / average | −0.233 / −0.621 |

R² and the error are measured on the log payout. R² 0.20 means about a fifth of the variation in log payout is explained; it is not a statement about dollars.

**Fold results.** The folds are not equal in difficulty:

| Fold | Loss years | Claims | R² |
|---|---|---|---|
| 0 | 2024 only | 68,076 | 0.301 |
| 1 | 10 years, 1978-2022 | 60,849 | 0.061 |
| 2 | 13 years, 1981-2023 | 60,916 | 0.154 |
| 3 | 13 years, 1979-2026 | 60,833 | −0.034 |
| 4 | 12 years, 1984-2025 | 60,833 | 0.153 |

The pooled score is dominated by a few seasons: 2024 holds 21.9% of the claims, 2022 holds 12.8%, 2004 holds 7.1% and 2005 holds 6.5%, so four years hold 48%. 2024 falls in a fold of its own, and its R² of 0.30 is the best of the five. Which years share a fold changes individual fold scores a lot (see the ablations below).

**Bias by year.** Mean error (actual minus predicted, in log payout): 2022 +0.95 (actual about 2.6 times predicted), 1993 +1.21 (3.4 times), 2024 +0.20, 2005 +0.33, 1999 −0.64 (actual about half of predicted), 1987 −0.78. The model is least reliable for seasons that look unlike the rest.

**Systematic under-prediction.** Across all held-out years the median actual claim is **1.44 times** the prediction (median error +0.367 in log payout, from `predictions.csv.gz` and the cleaned dataset). Years the model has not seen tend to be larger than the years it learned from. The L1 objective does not remove this. The Claim Estimator's own model has a 1.50x factor (see below). Both are the exponential of the median out-of-fold error, i.e. the ratio of the actual to the predicted median.

**Forward in time.** Training on 1978-2019 and testing on 2020-2026 gives R² −0.233, which is better than the −0.621 of predicting the average but still negative. Payouts have grown faster than inflation (next section), and the model has no way to know the future level.

## Payouts over time

Median paid claim in constant 2025 dollars, by decade of loss: $4,792 (1970s), $8,381 (1980s), $13,988 (1990s), $19,187 (2000s), $28,084 (2010s), $77,944 (2020s). Single years: $15,180 in 2015 (1,180 claims), $76,699 in 2022 (39,736 claims) and $89,057 in 2024 (68,076 claims). Payouts have grown far faster than CPI-U.

The data do not say why. Candidate explanations are the severity of recent storms (Ian, Helene, Milton), rebuilding costs rising faster than CPI-U, larger coverage limits, changes in how claims are handled, and changes in which buildings are insured. None is tested here. CPI-U was chosen because it is standard and public; a construction-cost index would give a different (probably smaller) drift, and that comparison is not made.

## Is building age a stand-in for the loss year?

Building age is the model's strongest feature, but old buildings appear mostly in recent claims: the median building age at loss (known construction dates) is 15 years in the 1980s, 38 in the 2010s and 50 in the 2020s. Recent claims are also larger, so age can pick up the calendar. `ablation_year_control.py` scores variants with the same folds as the headline model:

| Variant | R² | Median % error | Fold R² |
|---|---|---|---|
| Current model (age, no year) | 0.199 | 73.9 | 0.30, 0.06, 0.15, −0.03, 0.15 |
| Add loss year | 0.208 | 72.2 | 0.36, 0.09, 0.13, −0.04, 0.16 |
| Remove building age | 0.143 | 76.8 | 0.14, 0.06, 0.09, 0.00, 0.04 |
| Loss year, no building age | 0.213 | 72.1 | 0.24, 0.23, 0.12, −0.06, 0.17 |
| Loss year and construction year, no age | 0.230 | 71.4 | 0.37, 0.13, 0.16, −0.04, 0.19 |

Removing age costs 0.056 of R², so it carries signal. Replacing it with the loss year alone more than recovers that (0.213 vs 0.199), and adding the year on top of age barely moves the score. Most of what age contributes is therefore also available from the calendar year, and the two cannot be cleanly separated. Age should not be read as "older buildings cause larger payouts".

`ablation_postfirm.py` tests FEMA's post-FIRM flag, which is complete for every claim, as a cleaner stand-in:

| Variant | R² | Median % error |
|---|---|---|
| Add post-FIRM flag (age kept) | 0.191 | 74.1 |
| Post-FIRM flag instead of building age | 0.137 | 75.9 |
| Loss year and post-FIRM flag (age kept) | 0.203 | 72.0 |
| Loss year and post-FIRM flag, no building age | 0.226 | 71.2 |

The flag does not improve on age and is another coarse era marker (4.6% of claims from the 1970s are post-FIRM buildings, 45.0% of those from the 2010s). With the loss year in the model and age removed, it adds 0.014.

**How much these differences mean.** Ablation scores differ by a few hundredths, and individual folds move by 0.1 or more between models whose overall scores are within 0.01 of each other (up to 0.14). Fold 1 (ten mixed years) is the least stable. Between the two data pulls described under [Reproducibility](#reproducibility), the same "loss year and construction year" variant scored 0.211 and then 0.230, almost entirely from fold 1 (0.020 to 0.132). Differences of about 0.02 or less between variants should not be treated as findings. The conclusion that holds in both pulls is that the loss year recovers all of age's signal.

## Claim Estimator

The Claim Estimator tab uses a second LightGBM model with the loss year added to the 13 features (`train_year_model.py`; R² 0.208, median error 72%), so building age is not standing in for the era. The headline model, and the Model tab, keep the original features so that the headline stays conservative.

- **Typical payout:** the model's prediction multiplied by 1.50, the exponential of the median out-of-fold error of this model.
- **80% range:** the prediction multiplied by the exponential of the 10th and 90th percentiles of the same out-of-fold errors (−1.60 and +2.09 in log payout). It is an empirical error band pooled over all claims, so it has the same relative width for every scenario. It is probably too narrow for rare combinations of inputs and too wide for common ones. It is not a statistical prediction interval.
- **Inputs not set by the user:** the county's average claim coordinates and median storm frequency; the building-age-missing flag is 0 and days to storm is 0.
- **Interpretation:** the "what if the storm were stronger" control moves one input while the rest are held fixed. That shows a pattern in past claims, not the effect of changing a storm or a building. The year-aware model cannot extrapolate beyond the years in the data (1978 to 2026).

## Limitations

- **Modest signal.** R² 0.20 on log payout with a 74% median error on unseen years. The model is useful for seeing which factors matter, not for pricing a specific claim.
- **Severity given payment only.** Claims closed without payment (about 30%) are excluded.
- **Fragile folds.** 2024 is a fold on its own and carries 21.9% of the claims; fold scores range from −0.03 to 0.30.
- **Does not extrapolate in time.** The 2020-2026 test is negative, and the model under-predicts unseen years by about 1.4 times. The Claim Estimator corrects the median but not the cause.
- **Payout growth unexplained.** See [Payouts over time](#payouts-over-time).
- **Storm assignment is a heuristic.** The nearest track point within 150 miles and 7 days is not necessarily the storm that flooded a given property; 22.2% of claims match no storm.
- **Blurred locations.** About 7 miles.
- **Within narrow filter slices** (one county, one storm category) R² is much lower than the headline; the dashboard shows it for the filtered slice.
- **Coverage limits are not used,** although they cap payouts (about 4% of building payments sit at the limit).
- **No simple baseline beyond the average.** The model is compared with predicting the training average, not with a table of median payouts by county and year.
- **Pre-FIRM dates.** The 1492 placeholder is treated as unknown; whether that building is old is an inference from the pre-FIRM flag.
- **Flood zone A99.** Four claims in zone A99 are coded with the AE/A1-A30 group, although the group label lists A99 with the zones that have no base elevation. It does not change any result; `tests/` records it as an expected failure.
- **Tests.** `tests/` holds about 40 tests on small hand-built examples (cleaning, storm join, inflation, year-grouped folds, the committed bundle and every dashboard callback). They do not cover page layout or the live deployment, and they do not judge model quality.

## Reproducibility

Run order (details in the README): `fetch_data.py`, `build_dataset.py`, `train_model.py`, `train_year_model.py`, the three `ablation_*.py` scripts, then `build_deploy_data.py`. Training is seeded, and the results above are what that sequence produced from the 2026-10-20 pull. Rerunning it on Python 3.14 with pandas 3.0.6, NumPy 2.5.4, scikit-learn 1.9.1 and LightGBM 4.7.0 gave the same cleaned dataset, the same predictions and metrics to within 1e-13, as it had on the previous pinned versions (pandas 2.0.3, NumPy 1.24.3, scikit-learn 1.3.0, LightGBM 4.6.0).

**Data version.** The raw pull is not in the repository (about 14 MB). A new pull returns FEMA's current data, which may differ from the pull used here. The SHA-256 of the cached v3 pull, `data/raw/fema_nfip_claims_v3_fl.csv.gz`, is `7201a19294bfb37dd5abdb9085ef27acede67a43a6b0accab43851e95809f6fa`.

**Sensitivity to the data version.** The pipeline was first run on a pull of v2 of the same dataset (448,425 Florida rows, frozen 2026-06-01). Compared with v3, v2 had 193 fewer rows (mostly recent losses), the same placeholder dates, and slightly different values in about 0.3% of building payments and 0.1% of contents payments (the largest change is about +$38 million in 2024 payments). The results moved a little:

| | v2 pull | v3 pull |
|---|---|---|
| Paid claims | 311,400 | 311,507 |
| Headline R² | 0.204 | 0.199 |
| Fold R² | 0.32, 0.06, 0.16, −0.04, 0.16 | 0.30, 0.06, 0.15, −0.03, 0.15 |
| Train through 2019, test 2020+ | −0.271 | −0.233 |
| Claim Estimator model R² | 0.211 | 0.208 |
| Claim Estimator under-prediction factor | 1.53x | 1.50x |
| Loss year and construction year variant | 0.211 | 0.230 |

v3 also fills in `nfipCommunityName`, which was blank for 51% of v2 rows; the pipeline does not use it as a feature. The claim `id` changed from a UUID to an integer.

## Deployment notes

The hosted dashboard runs on Render's Starter plan (512 MB of RAM, kept awake). `render.yaml` installs `requirements-render.txt` and serves `wsgi.py` with gunicorn (one worker, four threads). The app holds all 311,507 paid claims as one compact table (about 4 MB on disk) and uses roughly 190 MB at startup and about 250 MB in use; every aggregate that can be precomputed is stored in `deploy_data/` by `build_deploy_data.py`.

`wsgi.py` answers `/healthz` immediately and imports the dashboard in a background thread started on the first request in each worker process. Starting the thread at import time does not work under gunicorn: the master process imports the module and then forks the worker, so the thread exists only in the master and the worker never finishes loading (forking while a thread is mid-import could also deadlock the worker; that possibility was not tested). `/debugz` reports whether the app has loaded, its memory use and any startup error.
