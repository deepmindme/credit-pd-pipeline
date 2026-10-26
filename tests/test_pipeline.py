import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import RAW_FEATURES, TARGET
from src.models.experiments import EXPERIMENTS
from src.models.pipeline import create_pipeline


@pytest.mark.parametrize("add_features", [True, False])
def test_pipeline_fits_and_returns_probabilities(clean_df, add_features):
    pipeline = create_pipeline(LogisticRegression(max_iter=1000), add_features=add_features)
    pipeline.fit(clean_df[RAW_FEATURES], clean_df[TARGET])
    proba = pipeline.predict_proba(clean_df[RAW_FEATURES])
    assert proba.shape == (len(clean_df), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_pipeline_learns_signal(clean_df):
    # В синтетике дефолт зависит от PAY_0, модель должна это уловить.
    pipeline = create_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(clean_df[RAW_FEATURES], clean_df[TARGET])
    proba = pipeline.predict_proba(clean_df[RAW_FEATURES])[:, 1]
    late = clean_df["PAY_0"] > 0
    assert proba[late].mean() > proba[~late].mean()


def test_pipeline_handles_missing_values_and_new_codes(clean_df):
    pipeline = create_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(clean_df[RAW_FEATURES], clean_df[TARGET])

    new_clients = clean_df[RAW_FEATURES].head(3).astype(float)
    new_clients.loc[new_clients.index[0], "BILL_AMT3"] = np.nan
    new_clients.loc[new_clients.index[1], "EDUCATION"] = 6
    proba = pipeline.predict_proba(new_clients)[:, 1]
    assert np.isfinite(proba).all()


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_search_space_matches_pipeline_params(name):
    """Опечатка в имени гиперпараметра всплыла бы только посреди обучения."""
    exp = EXPERIMENTS[name]
    pipeline = create_pipeline(exp.make_classifier(0), add_features=exp.add_features)
    assert set(exp.param_space) <= set(pipeline.get_params())
