# Florida Flood Claim Severity

![Florida Flood Claim Severity dashboard](docs/screenshot.png)

An interactive dashboard of what Florida's federal flood-insurance claims cost, and what drives the cost. It joins every paid claim to the hurricane or tropical storm that most likely caused it, and puts all payouts in constant 2025 dollars so a 1992 claim and a 2024 claim can be compared.

**Live dashboard: https://florida-flood-claim-severity.onrender.com/**

## What you get

- **311,507 paid claims, 1978 to 2026,** from the National Flood Insurance Program (NFIP), with each payout adjusted for inflation (CPI-U, the Consumer Price Index for All Urban Consumers).
- **Storm matching:** 77.8% of claims are matched to a storm that passed within 150 miles and 7 days of the loss. Of the claims FEMA itself labels as a named hurricane or tropical storm, 95.6% match the same-named storm.
- **A model, with its limits stated.** On storm seasons it has not seen, the model explains about a fifth of the variation in what a paid claim costs (R² 0.20), and a typical prediction is off by about 74%. That is a modest signal, not a precise predictor. It is most useful for seeing which factors matter: building age, storm wind speed, whether the building is elevated, and location.

## The dashboard

A shared filter bar (**Loss Years, County, Occupancy, Flood Zone, Storm Categories**; empty means all) controls the first four tabs. The Claim Estimator has its own controls.

| Tab | What it shows |
|---|---|
| **Overview** | Claims, total paid and median claim by year of loss (hover a bar for the storm behind that year), plus a county map. |
| **Storms** | The top storms by claims, total paid or median claim, with a table. |
| **Risk Factors** | Payout distributions by storm category, flood zone, occupancy type, elevated building, building age and distance from the storm track, with a table of claims, share of claims, median, mean and middle 50%. |
| **Model** | How well the model does on unseen years (with the average-prediction baseline and a leaky random split for contrast), calibration charts for the filtered slice, feature importance, and tests of whether building age is just a stand-in for the loss year. |
| **Claim Estimator** | A what-if tool: the typical payout and an 80% range for a hypothetical claim, as storm strength changes and across loss years. It shows patterns in past claims, not the effect of changing a storm or a building. |

## Data

| Source | What |
|---|---|
| [OpenFEMA NFIP Redacted Claims, v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3) (FEMA, Federal Emergency Management Agency) | 448,618 Florida claims, pulled 2026-10-20. |
| [NOAA HURDAT2](https://www.nhc.noaa.gov/data/) (Hurricane Database 2) | Atlantic storm positions and wind speeds, every six hours since 1851. |
| [FRED CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL) | CPI-U, for the inflation adjustment. |

Payout is building plus contents plus increased-cost-of-compliance payments. About 30% of claims were closed without payment and are excluded, so the numbers describe the cost of a claim given that it was paid. The data use FEMA's published, location-blurred coordinates (about 7 miles). This project is not endorsed by FEMA.

## How it works, and how far to trust it

Storm tracks are interpolated to hourly positions and each claim takes the closest track point within 150 miles and 7 days. A LightGBM model (gradient-boosted decision trees) predicts the log of the 2025-dollar payout from 13 building, storm and location features. It is scored with cross-validation that keeps each loss year entirely in either training or testing, because claims from one storm are near-duplicates and a random split overstates skill (R² 0.41 instead of 0.20). Trained through 2019 and tested on 2020 onward, the model scores below zero (−0.24, against −0.62 for predicting the average), because payouts have grown faster than inflation.

[docs/METHODOLOGY.md](docs/METHODOLOGY.md) covers every cleaning step with its counts, how the storm join was tuned and what that check does and does not show, the evaluation by fold and by year, the tests of building age and the loss year, how the Claim Estimator is calibrated, and the limitations.

## Run it

Requires Python 3.13 or newer (tested on 3.13 and 3.14; the hosted dashboard runs on 3.14).

```bash
pip install -r requirements.txt

python fetch_data.py             # download and cache raw data into data/raw/ (gitignored)
python build_dataset.py          # clean, storm-join, inflation-adjust, engineer features
python train_model.py            # year-grouped cross-validation, metrics, out-of-fold predictions
python train_year_model.py       # the loss-year model behind the Claim Estimator (required)
python ablation_year_control.py  # optional, about 10 min: is building age a stand-in for the loss year?
python ablation_postfirm.py      # optional, about 10 min: does FEMA's post-FIRM flag replace or add to age?
python ablation_placeholder_age.py  # optional, a few min: alternative fill for unusable construction dates
python build_deploy_data.py      # compact bundle for the app -> deploy_data/
python app.py                    # dashboard at http://localhost:8058
```

`deploy_data/` is committed, so the dashboard runs without any of the steps above. A fresh pull from OpenFEMA returns FEMA's current data, which can differ slightly from the 2026-10-20 pull behind the published numbers (see Reproducibility in the Methodology).

The hosted dashboard runs on Render using `render.yaml` (a Render Blueprint) and `wsgi.py`; deployment notes are in the Methodology.

## Tests

The test suite (about 40 tests) checks that the pipeline does what the documentation says, using small hand-built examples, so they need no downloaded data and no network. They run in about 15 seconds, and on every push through GitHub Actions. Examples of what they check:

- **Cleaning:** each exclusion removes the right rows and is counted; the three paid fields are summed; the 1492 placeholder and other unusable construction dates are flagged.
- **Storm join:** a claim matches the nearest storm inside 150 miles and 7 days and not outside either limit; the storm-frequency feature counts only earlier years.
- **Inflation:** the adjustment factor is 1 in the base year, and the missing October 2025 CPI month is interpolated.
- **Evaluation:** in the year-grouped cross-validation, no loss year appears in both the training and test side of a fold.
- **Committed bundle and dashboard:** `deploy_data/` is internally consistent (counts, metrics, the estimator model), and every dashboard callback returns without error, including when a filter matches no claims.

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

They do not check page layout, the live Render deployment, or whether the model is any good; model quality is what the evaluation in the Methodology measures.

## Files

| File | Purpose |
|---|---|
| `fetch_data.py` | Download and cache FEMA claims, HURDAT2 and CPI |
| `build_dataset.py` | Cleaning, storm join, features, inflation adjustment, validation |
| `train_model.py` | LightGBM training and evaluation (year-grouped cross-validation) |
| `train_year_model.py` | The loss-year model the Claim Estimator uses |
| `ablation_*.py` | Tests of whether building age is a stand-in for the loss year, whether the post-FIRM flag helps, and how to fill unusable construction dates |
| `build_deploy_data.py` | Packages the claims table, models and metrics into `deploy_data/` |
| `constants.py` | Category labels shared by the build script and the app |
| `app.py`, `wsgi.py` | The Dash dashboard and its hosting entry point |
| `render.yaml`, `requirements-render.txt` | Render deployment and the hosted app's dependencies |
| `requirements.txt` | Dependencies for the full pipeline and the app |
| `deploy_data/` | What the hosted app loads |
| `tests/`, `pytest.ini`, `requirements-dev.txt` | The test suite and its dependency |
| `.github/workflows/tests.yml` | Runs the tests on every push |
| `docs/METHODOLOGY.md` | Methods, evaluation, limitations and reproducibility |

Released under the MIT License.
