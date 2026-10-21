"""Does a better guess for the 11,719 buildings with no usable construction date improve the model?

98% of them (11,466) carry FEMA's '1492' sentinel date; the rest are 253 other bad dates.
Today those buildings (3.8% of claims) get the overall median age. But every sentinel row is flagged pre-FIRM (see
ablation_postfirm.py), and "pre-FIRM" is not a fixed age: the known-date pre-FIRM population's median age at loss grows
from 9 years in the 1970s to 54 in the 2020s (it's a fixed construction era getting older). 92% of placeholder rows are
themselves 1970s-80s losses. A single flat "old" constant would misfit most of them; a year-aware value should not.

Imputation tested: for each placeholder row, the median age of KNOWN-date, pre-FIRM claims from the SAME loss year
(falling back to a +/-2-year window, then the pre-FIRM population overall, for years too thin to have their own median).
Scored with the same year-grouped CV and folds as ablation_year_control.py / ablation_postfirm.py.

Run after build_dataset.py:   python ablation_placeholder_age.py   ->  data/ablation_placeholder_age.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import train_model as tm
from ablation_year_control import BASE, fit

ROOT = Path(__file__).parent


def main():
    # building_age_display_years is computed by build_dataset.py (display_building_age); it's used there for the
    # dashboard's Building age chart, and tested here as a candidate MODEL feature too.
    df = pd.read_csv(tm.DATA, low_memory=False)
    y, year = df[tm.TARGET], df["yearOfLoss"]
    folds = list(GroupKFold(n_splits=5).split(df, y, groups=year))

    variants = {
        "Current model (global median impute)": ("building_age_years", BASE),
        "Year-aware pre-FIRM median impute": ("building_age_display_years", BASE),
    }
    results = {}
    for name, (age_col, base_feats) in variants.items():
        feats = [age_col if f == "building_age_years" else f for f in base_feats]
        oof = np.zeros(len(df))
        for tr, te in folds:
            oof[te] = fit(df.iloc[tr], y.iloc[tr], feats).predict(df.iloc[te][feats])
        actual = np.expm1(y)
        ape = float(np.median(np.abs(np.expm1(oof) - actual) / actual) * 100)
        ph = df["building_age_missing"] == 1
        ape_ph = float(np.median(np.abs(np.expm1(oof[ph]) - actual[ph]) / actual[ph]) * 100)
        r2_ph = float(r2_score(y[ph], oof[ph])) if ph.sum() > 50 else None
        results[name] = {
            "r2_log": round(float(r2_score(y, oof)), 4), "median_abs_pct_error": round(ape, 1),
            "fold_r2": [round(float(r2_score(y.iloc[te], oof[te])), 3) for _, te in folds],
            "r2_log_placeholder_rows_only": round(r2_ph, 4) if r2_ph is not None else None,
            "median_abs_pct_error_placeholder_rows_only": round(ape_ph, 1),
        }
        r = results[name]
        print(f"{name:42s} R2 {r['r2_log']:.3f} | err {r['median_abs_pct_error']:.0f}% | "
              f"placeholder-only R2 {r['r2_log_placeholder_rows_only']} | placeholder-only err {r['median_abs_pct_error_placeholder_rows_only']:.0f}%",
              flush=True)

    (ROOT / "data" / "ablation_placeholder_age.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
