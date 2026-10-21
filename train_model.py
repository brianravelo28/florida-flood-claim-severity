"""Train the LightGBM claim-severity model on the real dataset; save honest out-of-sample metrics + predictions.

Headline metric: 5-fold cross-validation grouped by loss year, so every claim is predicted by a model that never saw
any claim from that year (i.e. never saw that storm season). A random split leaks storm-level information and is
reported only for contrast.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "insurance_clean.csv.gz"

FEATURES = [
    "building_age_years", "building_age_missing", "is_elevated", "flood_zone_encoded", "occupancy_group_encoded",
    "distance_bin", "storm_category_encoded", "historical_storm_freq", "latitude", "longitude",
    "distance_from_track_mi", "storm_wind_speed_kt", "days_to_storm",
]
CATEGORICAL = ["occupancy_group_encoded", "flood_zone_encoded"]
TARGET = "log_amountpaid_real"  # log of payout in constant 2025 dollars (CPI-U adjusted)
STRESS_TRAIN_END = 2019  # temporal stress test: train through 2019, test 2020+
N_ROUNDS = 250
# L1 in log space predicts the *median* payout (a "typical claim"); it scored slightly better than L2 on the same CV.
PARAMS = dict(objective="regression_l1", learning_rate=0.05, num_leaves=31, min_child_samples=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, lambda_l2=10.0, verbose=-1, seed=42)


def scores(y_log, p_log):
    y, p = np.expm1(y_log), np.expm1(p_log)
    return {
        "rmse_log": float(np.sqrt(mean_squared_error(y_log, p_log))),
        "mae_log": float(mean_absolute_error(y_log, p_log)),
        "r2_log": float(r2_score(y_log, p_log)),
        "median_abs_pct_error": float(np.median(np.abs(p - y) / y) * 100),
        "n": int(len(y_log)),
    }


def fit(X, y):
    return lgb.train(PARAMS, lgb.Dataset(X, y, categorical_feature=CATEGORICAL), num_boost_round=N_ROUNDS)


def main():
    df = pd.read_csv(DATA, low_memory=False)
    X, y, year = df[FEATURES], df[TARGET], df["yearOfLoss"]

    oof = np.zeros(len(df))
    base = np.zeros(len(df))
    fold = np.zeros(len(df), dtype=int)
    fold_r2 = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=year)):
        oof[te] = fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        base[te] = y.iloc[tr].mean()
        fold[te] = k
        fold_r2.append(round(float(r2_score(y.iloc[te], oof[te])), 3))
        print(f"fold {k}: years {[int(v) for v in sorted(year.iloc[te].unique())[:4]]}... n={len(te):,} R2={fold_r2[-1]}", flush=True)

    metrics = {
        "headline_year_grouped_cv": scores(y, oof),
        "fold_r2": fold_r2,
        "baseline_year_grouped_cv_predict_train_mean": scores(y, base),
        "features": FEATURES,
        "params": {**PARAMS, "num_boost_round": N_ROUNDS},
    }

    tr, te = (year <= STRESS_TRAIN_END).to_numpy(), (year > STRESS_TRAIN_END).to_numpy()
    p = fit(X[tr], y[tr]).predict(X[te])
    metrics["temporal_stress_test"] = {
        "train_years": f"1978-{STRESS_TRAIN_END}", "test_years": f"{STRESS_TRAIN_END + 1}-{int(year.max())}",
        "model": scores(y[te], p), "baseline_predict_train_mean": scores(y[te], np.full(te.sum(), y[tr].mean())),
    }

    r = np.random.RandomState(42).rand(len(df))
    rtr, rte = r < 0.8, r >= 0.8
    metrics["random_split_for_contrast"] = scores(y[rte], fit(X[rtr], y[rtr]).predict(X[rte]))

    by_year = pd.DataFrame({"year": year, "actual": y, "pred": oof}).groupby("year").agg(
        n=("actual", "size"), actual_mean_log=("actual", "mean"), predicted_mean_log=("pred", "mean"))
    metrics["by_year"] = {int(k): {c: round(float(v), 3) for c, v in row.items()} for k, row in by_year.iterrows()}

    final = fit(X, y)
    sample = X.sample(20000, random_state=42)
    imp = np.abs(final.predict(sample, pred_contrib=True)[:, :-1]).mean(axis=0)
    metrics["shap_importance_pct"] = dict(sorted(zip(FEATURES, (100 * imp / imp.sum()).round(2).tolist()),
                                                 key=lambda kv: -kv[1]))

    (ROOT / "data" / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    final.save_model(str(ROOT / "data" / "model.txt"))
    pd.DataFrame({"claimId": df["claimId"], "fold": fold, "prediction_log": oof.round(4),
                  "prediction_amount": np.expm1(oof).round(0)}).to_csv(ROOT / "data" / "predictions.csv.gz", index=False)

    def line(name, s):
        print(f"{name:44s} R2 {s['r2_log']:6.3f} | RMSE(log) {s['rmse_log']:.3f} | median abs % err {s['median_abs_pct_error']:.0f}% | n={s['n']:,}")

    print()
    line("YEAR-GROUPED CV (headline)", metrics["headline_year_grouped_cv"])
    line("  baseline: predict training mean", metrics["baseline_year_grouped_cv_predict_train_mean"])
    line(f"temporal stress test {metrics['temporal_stress_test']['test_years']}", metrics["temporal_stress_test"]["model"])
    line("  baseline: predict training mean", metrics["temporal_stress_test"]["baseline_predict_train_mean"])
    line("random split (leaky, contrast only)", metrics["random_split_for_contrast"])
    print("SHAP importance %:", metrics["shap_importance_pct"])


if __name__ == "__main__":
    main()
