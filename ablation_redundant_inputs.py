"""Do the two repeated inputs earn their place?

`distance_bin` is a five-level band of `distance_from_track_mi`, and `storm_category_encoded` is a category cut from
`storm_wind_speed_kt`. A tree model can recover either band from the continuous column, so both look redundant. This scores
the model with each removed, and with both removed, using the same year-grouped folds as the headline model, for the
headline feature set and for the feature set plus the loss year (the Claim Estimator's model).

Run after build_dataset.py:   python ablation_redundant_inputs.py   ->  data/ablation_redundant_inputs.json
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
BAND, CATEGORY = "distance_bin", "storm_category_encoded"


def without(features, *drop):
    return [f for f in features if f not in drop]


VARIANTS = {
    "Current model": BASE,
    "Without distance band": without(BASE, BAND),
    "Without storm category": without(BASE, CATEGORY),
    "Without both": without(BASE, BAND, CATEGORY),
    "Loss year added (Claim Estimator model)": BASE + ["yearOfLoss"],
    "Loss year, without both": without(BASE, BAND, CATEGORY) + ["yearOfLoss"],
}


def main():
    df = pd.read_csv(tm.DATA, low_memory=False)
    y, year = df[tm.TARGET], df["yearOfLoss"]
    folds = list(GroupKFold(n_splits=5).split(df, y, groups=year))
    results = {}
    for name, feats in VARIANTS.items():
        oof = np.zeros(len(df))
        for tr, te in folds:
            oof[te] = fit(df.iloc[tr], y.iloc[tr], feats).predict(df.iloc[te][feats])
        actual = np.expm1(y)
        ape = float(np.median(np.abs(np.expm1(oof) - actual) / actual) * 100)
        results[name] = {
            "n_features": len(feats),
            "r2_log": round(float(r2_score(y, oof)), 4), "median_abs_pct_error": round(ape, 1),
            "fold_r2": [round(float(r2_score(y.iloc[te], oof[te])), 3) for _, te in folds],
            "median_residual": round(float(np.median(y - oof)), 4),
        }
        r = results[name]
        print(f"{name:42s} {r['n_features']:2d} features | R2 {r['r2_log']:.4f} | err {r['median_abs_pct_error']:.1f}% | "
              f"folds {r['fold_r2']}", flush=True)
    (ROOT / "data" / "ablation_redundant_inputs.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
