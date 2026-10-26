import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TunedThresholdClassifierCV

from src.config import (
    BILL_COLUMNS,
    PAY_AMT_COLUMNS,
    PAY_STATUS_COLUMNS,
    RAW_FEATURES,
    RAW_TARGET,
    TARGET,
)
from src.models.pipeline import create_pipeline


def make_raw_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Синтетические данные в формате исходного CSV (с ID и исходным таргетом).

    Дефолт зависит от просрочки PAY_0, чтобы модели было что выучить.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "ID": np.arange(1, n + 1),
            "LIMIT_BAL": rng.choice([10_000, 50_000, 120_000, 200_000, 500_000], n),
            "SEX": rng.choice([1, 2], n),
            "EDUCATION": rng.choice(
                [0, 1, 2, 3, 4, 5, 6], n, p=[0.01, 0.35, 0.45, 0.15, 0.01, 0.02, 0.01]
            ),
            "MARRIAGE": rng.choice([0, 1, 2, 3], n, p=[0.01, 0.45, 0.52, 0.02]),
            "AGE": rng.integers(21, 75, n),
        }
    )
    for col in PAY_STATUS_COLUMNS:
        df[col] = rng.choice([-2, -1, 0, 1, 2, 3], n, p=[0.1, 0.2, 0.5, 0.1, 0.08, 0.02])
    for col in BILL_COLUMNS:
        df[col] = rng.integers(-2_000, 150_000, n)
    for col in PAY_AMT_COLUMNS:
        df[col] = rng.integers(0, 20_000, n)
    p_default = 0.12 + 0.35 * (df["PAY_0"] > 0)
    df[RAW_TARGET] = (rng.random(n) < p_default).astype(int)
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture
def clean_df(raw_df) -> pd.DataFrame:
    return raw_df.drop(columns="ID").rename(columns={RAW_TARGET: TARGET})


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    """Маленькая, но настоящая модель того же вида, что и в train.py."""
    df = make_raw_frame(n=600, seed=1)
    pipeline = create_pipeline(LogisticRegression(max_iter=1000), add_features=True)
    model = TunedThresholdClassifierCV(pipeline, scoring="f1", cv=3)
    model.fit(df[RAW_FEATURES], df[RAW_TARGET])

    path = tmp_path_factory.mktemp("models") / "model.joblib"
    joblib.dump(model, path)
    meta = {"experiment": "test_model", "registered_version": "1"}
    path.with_name("model_meta.json").write_text(json.dumps(meta))
    return path
