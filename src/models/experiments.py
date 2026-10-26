"""Описание экспериментов: алгоритм, признаки и пространство гиперпараметров.

Пары *_raw / *_fe отличаются только наличием feature engineering. По ним
видно, сколько дают новые признаки для линейной модели и для бустинга.
"""

from dataclasses import dataclass, field
from typing import Callable

from scipy.stats import loguniform, randint, uniform
from sklearn.base import ClassifierMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    make_classifier: Callable[[int], ClassifierMixin]
    add_features: bool
    search: str  # "grid" или "random"
    param_space: dict = field(default_factory=dict)
    n_iter: int = 10


LOGREG_GRID = {
    "classifier__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
    "classifier__class_weight": [None, "balanced"],
}

HIST_GB_SPACE = {
    "classifier__learning_rate": loguniform(0.02, 0.2),
    "classifier__max_iter": randint(150, 500),
    "classifier__max_leaf_nodes": randint(8, 48),
    "classifier__min_samples_leaf": randint(20, 200),
    "classifier__l2_regularization": loguniform(1e-3, 10),
}


def _logreg(seed: int) -> ClassifierMixin:
    return LogisticRegression(max_iter=2000, random_state=seed)


def _hist_gb(seed: int) -> ClassifierMixin:
    # Ранняя остановка по внутренней валидации: меньше переобучения и быстрее.
    return HistGradientBoostingClassifier(
        early_stopping=True, validation_fraction=0.1, random_state=seed
    )


EXPERIMENTS = {
    exp.name: exp
    for exp in [
        Experiment(
            name="logreg_raw",
            description="Бейзлайн: логистическая регрессия на исходных признаках",
            make_classifier=_logreg,
            add_features=False,
            search="grid",
            param_space=LOGREG_GRID,
        ),
        Experiment(
            name="logreg_fe",
            description="Логистическая регрессия + агрегаты истории платежей",
            make_classifier=_logreg,
            add_features=True,
            search="grid",
            param_space=LOGREG_GRID,
        ),
        Experiment(
            name="random_forest_fe",
            description="Случайный лес + feature engineering",
            make_classifier=lambda seed: RandomForestClassifier(n_jobs=1, random_state=seed),
            add_features=True,
            search="random",
            param_space={
                "classifier__n_estimators": randint(200, 500),
                "classifier__max_depth": [6, 8, 10, 12, 16],
                "classifier__min_samples_leaf": randint(5, 50),
                "classifier__max_features": ["sqrt", 0.3, 0.5],
                "classifier__class_weight": [None, "balanced_subsample"],
            },
            n_iter=12,
        ),
        Experiment(
            name="gradient_boosting_fe",
            description="GradientBoostingClassifier + feature engineering",
            make_classifier=lambda seed: GradientBoostingClassifier(random_state=seed),
            add_features=True,
            search="random",
            param_space={
                "classifier__n_estimators": randint(100, 300),
                "classifier__learning_rate": loguniform(0.02, 0.2),
                "classifier__max_depth": [2, 3, 4],
                "classifier__subsample": uniform(0.6, 0.4),
                "classifier__min_samples_leaf": randint(20, 150),
            },
            n_iter=10,
        ),
        Experiment(
            name="hist_gb_raw",
            description="HistGradientBoosting на исходных признаках",
            make_classifier=_hist_gb,
            add_features=False,
            search="random",
            param_space=HIST_GB_SPACE,
            n_iter=20,
        ),
        Experiment(
            name="hist_gb_fe",
            description="HistGradientBoosting + feature engineering",
            make_classifier=_hist_gb,
            add_features=True,
            search="random",
            param_space=HIST_GB_SPACE,
            n_iter=20,
        ),
    ]
}
