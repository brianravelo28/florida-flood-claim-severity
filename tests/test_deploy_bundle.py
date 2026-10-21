"""The committed deploy_data/ bundle is internally consistent and agrees with the numbers the docs quote."""
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

DEPLOY = Path(__file__).resolve().parent.parent / "deploy_data"


def load(name):
    return json.loads((DEPLOY / name).read_text())


@pytest.fixture(scope="module")
def claims():
    return pd.read_parquet(DEPLOY / "claims.parquet")


def test_claim_counts_agree_across_files(claims):
    report, agg = load("build_report.json"), load("aggregates.json")
    assert len(claims) == report["final_rows"] == agg["n_claims"]
    assert report["raw_rows"] - sum(report["rows_dropped"].values()) == report["final_rows"]


def test_claims_table_has_no_gaps_where_the_app_needs_values(claims):
    for col in ("occupancy_group", "flood_zone_group", "storm_category", "elevated", "age_band", "distance_band",
                "yearOfLoss", "amountPaid_real", "prediction_amount"):
        assert claims[col].notna().all(), col
    assert (claims["amountPaid_real"] > 0).all()
    agg = load("aggregates.json")
    assert claims["yearOfLoss"].min() == agg["year_min"] and claims["yearOfLoss"].max() == agg["year_max"]


def test_data_validation_checks_all_passed():
    checks = load("build_report.json")["checks"]
    assert checks and all(checks.values()), [k for k, v in checks.items() if not v]


def test_headline_metrics_are_consistent():
    m = load("model_metrics.json")
    head, base = m["headline_year_grouped_cv"], m["baseline_year_grouped_cv_predict_train_mean"]
    assert len(m["fold_r2"]) == 5
    assert head["r2_log"] > base["r2_log"], "the model should beat predicting the average"
    assert m["random_split_for_contrast"]["r2_log"] > head["r2_log"], "a random split leaks storms and should score higher"
    assert abs(sum(m["shap_importance_pct"].values()) - 100) < 0.5
    assert abs(head["r2_log"] - load("ablation_year_control.json")["Current model"]["r2_log"]) < 1e-3, \
        "the ablations must use the same folds as the headline model"


def test_residual_quantiles_are_ordered_and_give_the_documented_correction():
    for q in (load("aggregates.json")["residual_quantiles"], load("year_control_model.json")["residual_quantiles"]):
        assert q["q10"] < q["q50"] < q["q90"]
    assert math.exp(load("aggregates.json")["residual_quantiles"]["q50"]) > 1, "unseen years are under-predicted"


def test_estimator_model_loads_and_predicts_a_plausible_payout():
    meta = load("year_control_model.json")
    booster = lgb.Booster(model_file=str(DEPLOY / "model_year.txt"))
    assert booster.feature_name() == meta["features"]
    assert "yearOfLoss" in meta["features"]
    row = {f: 0.0 for f in meta["features"]}
    row.update(yearOfLoss=2020, building_age_years=30, storm_wind_speed_kt=90, distance_from_track_mi=40,
               storm_category_encoded=3, distance_bin=3, days_to_storm=0, latitude=27.0, longitude=-81.0,
               historical_storm_freq=15)
    payout = float(np.expm1(booster.predict(pd.DataFrame([row])[meta["features"]])[0]))
    assert 1_000 < payout < 1_000_000
