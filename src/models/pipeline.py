"""Сборка sklearn Pipeline: признаки -> предобработка -> классификатор."""

from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import (
    DEFAULT_AGE_BINS,
    FeatureBuilder,
    feature_columns,
)


def create_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    # В текущем датасете пропусков нет, но импьютеры оставлены специально:
    # в API может прийти неполная запись, и пайплайн не должен на ней падать.
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical),
        ]
    )


def create_pipeline(
    classifier: ClassifierMixin,
    add_features: bool = True,
    age_bins=DEFAULT_AGE_BINS,
) -> Pipeline:
    numeric, categorical = feature_columns(add_features)
    return Pipeline(
        steps=[
            ("features", FeatureBuilder(add_features=add_features, age_bins=age_bins)),
            ("preprocessor", create_preprocessor(numeric, categorical)),
            ("classifier", classifier),
        ]
    )
