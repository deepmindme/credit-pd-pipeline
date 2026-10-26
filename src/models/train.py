"""Стадия train: серия экспериментов с подбором гиперпараметров и логированием в MLflow.

Для каждого эксперимента из params.yaml:
  1. GridSearchCV / RandomizedSearchCV по ROC-AUC на 5 фолдах train;
  2. подбор порога классификации по CV (TunedThresholdClassifierCV, максимум F1);
  3. метрики на отложенном test, графики и модель логируются в MLflow.

Лучшая модель выбирается по CV ROC-AUC, а не по test, чтобы тестовая выборка
оставалась честной оценкой. Она сохраняется в models/ и регистрируется в
MLflow Model Registry с алиасом champion.

Запуск: python -m src.models.train
"""

import json
import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import skops.io
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    TunedThresholdClassifierCV,
)

from src.config import RAW_FEATURES, TARGET, load_params, resolve
from src.models.experiments import EXPERIMENTS, Experiment
from src.models.metrics import (
    classification_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
)
from src.models.pipeline import create_pipeline

logger = logging.getLogger(__name__)

MODELS_DIR = resolve("models")
REPORTS_DIR = resolve("reports")


def load_split(processed_dir: Path, name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(processed_dir / f"{name}.csv")
    return df[RAW_FEATURES], df[TARGET]


def make_search(exp: Experiment, params: dict):
    seed = params["seed"]
    train_cfg = params["train"]
    pipeline = create_pipeline(
        exp.make_classifier(seed),
        add_features=exp.add_features,
        age_bins=params["features"]["age_bins"],
    )
    cv = StratifiedKFold(n_splits=train_cfg["cv_folds"], shuffle=True, random_state=seed)
    common = dict(
        scoring=train_cfg["search_scoring"],
        cv=cv,
        n_jobs=train_cfg["n_jobs"],
        refit=True,
    )
    if exp.search == "grid":
        return GridSearchCV(pipeline, exp.param_space, **common)
    return RandomizedSearchCV(
        pipeline, exp.param_space, n_iter=exp.n_iter, random_state=seed, **common
    )


def tune_threshold(estimator, X, y, params: dict) -> TunedThresholdClassifierCV:
    """Порог, максимизирующий F1 на кросс-валидации по train.

    Дефолтов ~22%, поэтому порог 0.5 даёт слишком низкий recall. Подбирать
    порог на test нельзя, иначе метрики на нём перестанут быть честными.
    """
    cv = StratifiedKFold(
        n_splits=params["train"]["cv_folds"], shuffle=True, random_state=params["seed"]
    )
    tuned = TunedThresholdClassifierCV(
        estimator,
        scoring=params["train"]["threshold_scoring"],
        cv=cv,
        n_jobs=params["train"]["n_jobs"],
    )
    return tuned.fit(X, y)


def raw_feature_importance(model, X, y, seed: int) -> pd.Series:
    """Permutation importance по входным признакам (снижение ROC-AUC).

    Считается одинаково для любых моделей и сразу в терминах полей,
    которые приходят в API.
    """
    result = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=5, random_state=seed, n_jobs=-1
    )
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)


def log_search_candidates(search, exp: Experiment) -> None:
    """Каждый кандидат поиска пишем вложенным run: в UI MLflow по ним удобно
    строить parallel coordinates и смотреть влияние гиперпараметров."""
    cv_results = pd.DataFrame(search.cv_results_)
    for _, row in cv_results.iterrows():
        with mlflow.start_run(run_name=f"{exp.name}-cand", nested=True):
            mlflow.log_params({k.replace("classifier__", ""): v for k, v in row["params"].items()})
            mlflow.log_metrics(
                {
                    "cv_roc_auc_mean": row["mean_test_score"],
                    "cv_roc_auc_std": row["std_test_score"],
                    "rank": row["rank_test_score"],
                    "fit_time_sec": row["mean_fit_time"],
                }
            )


def run_experiment(exp: Experiment, data: dict, params: dict) -> dict:
    X_train, y_train, X_test, y_test = (
        data["train_X"],
        data["train_y"],
        data["test_X"],
        data["test_y"],
    )
    logger.info("=== %s: %s", exp.name, exp.description)

    with mlflow.start_run(run_name=exp.name) as run:
        mlflow.set_tags(
            {
                "model_family": exp.make_classifier(0).__class__.__name__,
                "feature_set": "engineered" if exp.add_features else "raw",
                "search_type": exp.search,
                "mlflow.note.content": exp.description,
            }
        )
        mlflow.log_input(data["train_dataset"], context="training")
        mlflow.log_input(data["test_dataset"], context="testing")

        started = time.perf_counter()
        search = make_search(exp, params)
        search.fit(X_train, y_train)
        tuned = tune_threshold(search.best_estimator_, X_train, y_train, params)
        fit_time = time.perf_counter() - started
        log_search_candidates(search, exp)

        best_idx = search.best_index_
        cv_std = float(search.cv_results_["std_test_score"][best_idx])
        threshold = float(tuned.best_threshold_)

        mlflow.log_params(
            {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
        )
        mlflow.log_params(
            {
                "experiment": exp.name,
                "add_features": exp.add_features,
                "search": exp.search,
                "n_candidates": len(search.cv_results_["params"]),
                "cv_folds": params["train"]["cv_folds"],
                "seed": params["seed"],
                "decision_threshold": round(threshold, 4),
            }
        )

        proba = tuned.predict_proba(X_test)[:, 1]
        test_metrics = classification_metrics(y_test, proba, threshold)
        default_metrics = classification_metrics(y_test, proba, 0.5)
        mlflow.log_metrics(
            {
                "cv_roc_auc_mean": float(search.best_score_),
                "cv_roc_auc_std": cv_std,
                "fit_time_sec": fit_time,
                **{f"test_{k}": v for k, v in test_metrics.items() if k != "threshold"},
                **{f"test_{k}_at_0.5": default_metrics[k] for k in ("precision", "recall", "f1")},
            }
        )

        importance = raw_feature_importance(tuned, X_test, y_test, params["seed"])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plot_roc_curve(y_test, proba, f"ROC — {exp.name}", tmp / "roc_curve.png")
            plot_pr_curve(y_test, proba, threshold, tmp / "pr_curve.png")
            plot_confusion_matrix(y_test, proba, threshold, tmp / "confusion_matrix.png")
            plot_feature_importance(
                importance.index,
                importance.values,
                "Permutation importance (ΔROC-AUC)",
                tmp / "feature_importance.png",
            )
            pd.DataFrame(search.cv_results_).drop(columns="params").to_csv(
                tmp / "cv_results.csv", index=False
            )
            importance.rename("importance").to_csv(tmp / "feature_importance.csv")
            mlflow.log_artifacts(str(tmp), artifact_path="evaluation")

        # Сигнатура во float: пайплайн умеет обрабатывать пропуски, а
        # integer-колонки в схеме MLflow их не допускают.
        signature = infer_signature(X_test.astype(float), proba)
        model_info = mlflow.sklearn.log_model(
            tuned,
            name="model",
            signature=signature,
            input_example=X_test.head(3).astype(float),
            # MLflow сохраняет модель в безопасном формате skops и требует
            # явно перечислить типы, которым мы доверяем. Объект только что
            # обучен нами же, поэтому доверяем всем его типам.
            skops_trusted_types=skops.io.get_untrusted_types(data=skops.io.dumps(tuned)),
            # FeatureBuilder — наш класс, без кода src/ модель не загрузится.
            code_paths=[str(resolve("src"))],
        )
        logger.info(
            "%s: CV ROC-AUC %.4f ± %.4f, test ROC-AUC %.4f, F1 %.4f (порог %.3f), %.0f c",
            exp.name,
            search.best_score_,
            cv_std,
            test_metrics["roc_auc"],
            test_metrics["f1"],
            threshold,
            fit_time,
        )

    return {
        "experiment": exp.name,
        "run_id": run.info.run_id,
        "model_uri": model_info.model_uri,
        "model": tuned,
        "cv_roc_auc_mean": float(search.best_score_),
        "cv_roc_auc_std": cv_std,
        "fit_time_sec": round(fit_time, 1),
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }


def register_champion(best: dict, model_name: str) -> str:
    client = MlflowClient()
    version = mlflow.register_model(best["model_uri"], model_name)
    client.set_registered_model_alias(model_name, "champion", version.version)
    client.set_tag(best["run_id"], "champion", "true")
    logger.info("Модель %s v%s помечена как champion", model_name, version.version)
    return version.version


def main() -> None:
    params = load_params()
    processed_dir = resolve(params["data"]["processed_dir"])
    train_X, train_y = load_split(processed_dir, "train")
    test_X, test_y = load_split(processed_dir, "test")
    data = {
        "train_X": train_X,
        "train_y": train_y,
        "test_X": test_X,
        "test_y": test_y,
        "train_dataset": mlflow.data.from_pandas(
            pd.concat([train_X, train_y], axis=1),
            source=str(processed_dir / "train.csv"),
            name="train",
            targets=TARGET,
        ),
        "test_dataset": mlflow.data.from_pandas(
            pd.concat([test_X, test_y], axis=1),
            source=str(processed_dir / "test.csv"),
            name="test",
            targets=TARGET,
        ),
    }

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    results = [
        run_experiment(EXPERIMENTS[name], data, params) for name in params["train"]["experiments"]
    ]
    best = max(results, key=lambda r: r["cv_roc_auc_mean"])
    version = register_champion(best, params["mlflow"]["registered_model_name"])

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best["model"], MODELS_DIR / "model.joblib")
    meta = {
        "experiment": best["experiment"],
        "mlflow_run_id": best["run_id"],
        "registered_model": params["mlflow"]["registered_model_name"],
        "registered_version": version,
        "threshold": best["test_threshold"],
        "cv_roc_auc_mean": best["cv_roc_auc_mean"],
        "features": RAW_FEATURES,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (MODELS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))

    summary = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in ("model", "model_uri")} for r in results]
    )
    summary = summary.sort_values("cv_roc_auc_mean", ascending=False)
    REPORTS_DIR.mkdir(exist_ok=True)
    summary.round(4).to_csv(REPORTS_DIR / "experiments.csv", index=False)
    logger.info(
        "Итоги экспериментов:\n%s", summary.drop(columns="run_id").round(4).to_string(index=False)
    )
    logger.info("Лучшая модель: %s (CV ROC-AUC %.4f)", best["experiment"], best["cv_roc_auc_mean"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    main()
