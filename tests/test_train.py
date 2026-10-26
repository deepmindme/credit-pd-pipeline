"""Интеграционные тесты обучения: маленький эксперимент целиком проходит
через поиск, подбор порога и логирование в MLflow (во временное хранилище)."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.config import RAW_FEATURES, TARGET, load_params
from src.models.evaluate import decile_table
from src.models.experiments import EXPERIMENTS, Experiment
from src.models.train import make_search, run_experiment
from tests.conftest import make_raw_frame

TINY = Experiment(
    name="tiny_logreg",
    description="тестовый эксперимент",
    make_classifier=lambda seed: LogisticRegression(max_iter=500),
    add_features=True,
    search="grid",
    param_space={"classifier__C": [0.1, 1.0]},
)


@pytest.fixture
def params():
    params = load_params()
    params["train"].update(cv_folds=3, n_jobs=1)
    return params


@pytest.fixture
def mlflow_tmp(tmp_path):
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_id = mlflow.create_experiment(
        "test", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    yield MlflowClient()
    mlflow.set_tracking_uri(previous)


def make_data(n: int = 600) -> dict:
    df = make_raw_frame(n=n, seed=3).rename(columns={"default.payment.next.month": TARGET})
    train, test = df.iloc[: n * 3 // 4], df.iloc[n * 3 // 4 :]
    return {
        "train_X": train[RAW_FEATURES],
        "train_y": train[TARGET],
        "test_X": test[RAW_FEATURES],
        "test_y": test[TARGET],
        "train_dataset": mlflow.data.from_pandas(train, name="train", targets=TARGET),
        "test_dataset": mlflow.data.from_pandas(test, name="test", targets=TARGET),
    }


def test_make_search_uses_grid_or_random(params):
    assert isinstance(make_search(EXPERIMENTS["logreg_fe"], params), GridSearchCV)
    search = make_search(EXPERIMENTS["hist_gb_fe"], params)
    assert isinstance(search, RandomizedSearchCV)
    assert search.n_iter == EXPERIMENTS["hist_gb_fe"].n_iter


def test_run_experiment_logs_everything(params, mlflow_tmp):
    result = run_experiment(TINY, make_data(), params)
    run = mlflow_tmp.get_run(result["run_id"])

    assert {"cv_roc_auc_mean", "test_roc_auc", "test_f1", "test_recall_at_0.5"} <= set(
        run.data.metrics
    )
    assert run.data.params["C"] in {"0.1", "1.0"}
    assert 0 < float(run.data.params["decision_threshold"]) < 1
    assert run.data.tags["feature_set"] == "engineered"

    artifacts = {a.path for a in mlflow_tmp.list_artifacts(run.info.run_id, "evaluation")}
    assert "evaluation/roc_curve.png" in artifacts
    assert "evaluation/cv_results.csv" in artifacts

    children = mlflow_tmp.search_runs(
        [run.info.experiment_id], f"tags.mlflow.parentRunId = '{run.info.run_id}'"
    )
    assert len(children) == 2  # по одному вложенному run на кандидата сетки

    model = mlflow.sklearn.load_model(result["model_uri"])
    proba = model.predict_proba(make_data()["test_X"])[:, 1]
    assert np.allclose(proba, result["model"].predict_proba(make_data()["test_X"])[:, 1])


def test_decile_table_is_ordered_and_complete():
    rng = np.random.default_rng(0)
    proba = rng.uniform(0, 1, 1000)
    y = (rng.uniform(0, 1, 1000) < proba).astype(int)
    table = decile_table(pd.Series(y), proba)

    assert len(table) == 10
    assert table["clients"].sum() == 1000
    assert table["pd_mean"].is_monotonic_decreasing
    assert table["cum_default_capture"].iloc[-1] == pytest.approx(1.0)
