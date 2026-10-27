"""The app and the pipeline use the same wind thresholds (knots)."""
import numpy as np
import pytest

import build_dataset as bd

dash_app = pytest.importorskip("app")


def test_estimator_and_pipeline_assign_the_same_category():
    for kt in range(1, 181):
        assert dash_app.wind_category(float(kt)) == int(bd.wind_to_category([kt])[0]), kt
    assert dash_app.wind_category(0) == -1 and dash_app.wind_category(np.nan) == -1
