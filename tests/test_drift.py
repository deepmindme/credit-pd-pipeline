import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    build_profile,
    build_reference_profile,
    drift_report,
    psi,
    psi_from_proportions,
    psi_status,
)
from src.monitoring.simulate_traffic import apply_drift

rng = np.random.default_rng(0)
REFERENCE = rng.normal(0, 1, 5_000)


def test_psi_of_identical_distributions_is_zero():
    assert psi_from_proportions([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0)


def test_psi_known_value():
    # (0.5-0.25)·ln(0.5/0.25) + (0.5-0.75)·ln(0.5/0.75)
    expected = 0.25 * np.log(2) - 0.25 * np.log(2 / 3)
    assert psi_from_proportions([0.25, 0.75], [0.5, 0.5]) == pytest.approx(expected)


def test_psi_handles_empty_bins():
    assert np.isfinite(psi_from_proportions([0.5, 0.5, 0.0], [0.4, 0.4, 0.2]))


def test_same_population_is_stable():
    profile = build_profile(REFERENCE)
    sample = rng.normal(0, 1, 2_000)
    assert psi_status(psi(profile, sample)) == "stable"


def test_shifted_population_is_flagged():
    profile = build_profile(REFERENCE)
    shifted = rng.normal(1, 1, 2_000)
    assert psi_status(psi(profile, shifted)) == "significant"


def test_numeric_profile_proportions_sum_to_one():
    profile = build_profile(REFERENCE, n_bins=10)
    assert profile["type"] == "numeric"
    assert len(profile["proportions"]) == len(profile["cuts"]) + 1
    assert sum(profile["proportions"]) == pytest.approx(1)


def test_low_cardinality_feature_is_categorical():
    profile = build_profile(pd.Series([-1, 0, 0, 1, 2, 2, 2]))
    assert profile["type"] == "categorical"
    assert profile["categories"] == [-1, 0, 1, 2]


def test_unseen_category_increases_psi():
    profile = build_profile(pd.Series([1, 2, 3] * 100))
    assert psi(profile, pd.Series([1, 2, 3] * 30)) == pytest.approx(0, abs=1e-9)
    assert psi(profile, pd.Series([1, 2, 3] * 30 + [9] * 30)) > 0.25


@pytest.mark.parametrize(
    "value, status", [(0.05, "stable"), (0.1, "moderate"), (0.3, "significant")]
)
def test_psi_status_thresholds(value, status):
    assert psi_status(value) == status


def test_drift_report_and_simulated_drift(clean_df):
    features = ["LIMIT_BAL", "AGE", "PAY_0"]
    scores = rng.uniform(0, 1, len(clean_df))
    profile = build_reference_profile(clean_df, scores, features)

    same = drift_report(profile, clean_df, scores)
    assert list(same["variable"]) == ["score", *features]
    assert (same["psi"] < 1e-9).all()

    drifted = drift_report(profile, apply_drift(clean_df, seed=0), scores)
    flagged = set(drifted.loc[drifted["status"] != "stable", "variable"])
    assert {"LIMIT_BAL", "AGE"} <= flagged
