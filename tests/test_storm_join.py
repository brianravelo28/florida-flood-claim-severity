"""The storm join: radius, time window, nearest-storm choice, and the no-leakage storm-frequency feature."""
import pandas as pd
import pytest

import build_dataset as bd

LAT, LON = 27.0, -81.0
MI_PER_DEG_LAT = 69.1


def hour_of(date):
    return (pd.Timestamp(date) - pd.Timestamp("1970-01-01")).total_seconds() / 3600


def track_point(storm_id, year, date, miles_north, wind=80.0, name=None):
    return dict(storm_id=storm_id, storm_name=name or storm_id, storm_year=year, hour=hour_of(date),
                lat=LAT + miles_north / MI_PER_DEG_LAT, lon=LON, wind=wind)


def claims_at_location(dates):
    dates = pd.to_datetime(dates)
    return pd.DataFrame({"latitude": LAT, "longitude": LON, "dateOfLoss": dates, "yearOfLoss": dates.year})


def test_claim_matches_the_nearest_storm_inside_the_radius_and_window():
    tracks = pd.DataFrame([
        track_point("A", 2000, "2000-09-10", miles_north=35, wind=90.0, name="Alpha"),
        track_point("B", 2000, "2000-09-10", miles_north=100, wind=120.0, name="Bravo"),
    ])
    out = bd.join_storms(claims_at_location(["2000-09-11"]), tracks)
    assert out["storm_id"].iloc[0] == "A"
    assert out["nearest_storm_name"].iloc[0] == "Alpha"
    assert out["distance_from_track_mi"].iloc[0] == pytest.approx(35, abs=1.5)
    assert out["storm_wind_speed_kt"].iloc[0] == 90.0
    assert out["days_to_storm"].iloc[0] == -1


def test_claim_outside_the_seven_day_window_does_not_match():
    tracks = pd.DataFrame([track_point("A", 2000, "2000-09-10", miles_north=35)])
    inside = bd.join_storms(claims_at_location(["2000-09-17"]), tracks)    # 7 days after
    outside = bd.join_storms(claims_at_location(["2000-09-18"]), tracks)   # 8 days after
    assert inside["storm_id"].notna().all()
    assert outside["storm_id"].isna().all()
    assert outside["distance_from_track_mi"].isna().all()


def test_claim_outside_the_150_mile_radius_does_not_match():
    tracks = pd.DataFrame([track_point("A", 2000, "2000-09-10", miles_north=172)])
    assert bd.join_storms(claims_at_location(["2000-09-10"]), tracks)["storm_id"].isna().all()


def test_storm_frequency_counts_only_earlier_years_within_40_years_and_50_miles():
    tracks = pd.DataFrame([
        track_point("F1960", 1960, "1960-09-01", miles_north=20),   # 55 years before 2015: too old
        track_point("F1990", 1990, "1990-09-01", miles_north=20),
        track_point("F2000", 2000, "2000-09-01", miles_north=20),
        track_point("F2010", 2010, "2010-09-01", miles_north=20),
        track_point("F2015", 2015, "2015-09-01", miles_north=20),   # the claim's own year: not counted
        track_point("FAR", 2005, "2005-09-01", miles_north=80),     # inside 150 miles but beyond 50
    ])
    out = bd.join_storms(claims_at_location(["2015-06-01", "1990-06-01"]), tracks)
    assert out["historical_storm_freq"].tolist() == [3, 1]  # 1990, 2000, 2010 | 1960 only (1990 itself is excluded)
