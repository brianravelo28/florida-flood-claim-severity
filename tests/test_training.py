"""Year-grouped cross-validation: every loss year is predicted by a model that never saw it."""
import json

import numpy as np
import pandas as pd

import train_model as tm


def synthetic_dataset(n=21000, seed=0):  # the importance step samples 20,000 rows
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in tm.FEATURES})
    df["occupancy_group_encoded"] = rng.integers(0, 8, n)
    df["flood_zone_encoded"] = rng.integers(0, 5, n)
    df["yearOfLoss"] = rng.integers(1990, 2005, n)
    df["claimId"] = np.arange(n)
    df[tm.TARGET] = 8 + 0.5 * df["storm_wind_speed_kt"] + rng.normal(size=n)
    return df


def test_no_loss_year_appears_on_both_sides_of_a_fold(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    csv = data / "clean.csv"
    synthetic_dataset().to_csv(csv, index=False)
    monkeypatch.setattr(tm, "ROOT", tmp_path)
    monkeypatch.setattr(tm, "DATA", csv)
    monkeypatch.setattr(tm, "N_ROUNDS", 5)
    monkeypatch.setattr(tm, "STRESS_TRAIN_END", 1998)
    tm.main()

    pred = pd.read_csv(data / "predictions.csv.gz").merge(pd.read_csv(csv)[["claimId", "yearOfLoss"]], on="claimId")
    assert pred["fold"].nunique() == 5
    assert (pred.groupby("yearOfLoss")["fold"].nunique() == 1).all(), "a loss year was split across folds"
    assert len(pred) == 21000
    assert np.isfinite(pred["prediction_log"]).all()
    assert (pred["prediction_amount"] >= 0).all()

    metrics = json.loads((data / "model_metrics.json").read_text())
    assert len(metrics["fold_r2"]) == 5
    assert metrics["headline_year_grouped_cv"]["n"] == 21000
    assert metrics["temporal_stress_test"]["train_years"] == "1978-1998"
    assert metrics["headline_year_grouped_cv"]["r2_log"] > metrics["baseline_year_grouped_cv_predict_train_mean"]["r2_log"]
