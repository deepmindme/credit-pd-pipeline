"""Feature engineering для PD-модели.

Всё, что здесь есть, выполняется внутри sklearn Pipeline (FeatureBuilder).
Поэтому одни и те же преобразования применяются при обучении и в API,
и модель принимает на вход «сырые» признаки клиента.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    BILL_COLUMNS,
    CATEGORICAL_RAW,
    PAY_AMT_COLUMNS,
    PAY_STATUS_COLUMNS,
    RAW_FEATURES,
)

DEFAULT_AGE_BINS = (20, 25, 30, 35, 40, 50, 60, 100)

ENGINEERED_NUMERIC = [
    "PAY_DELAY_MAX",
    "PAY_DELAY_COUNT",
    "PAY_NO_USE_COUNT",
    "PAY_FULL_COUNT",
    "UTIL_LAST",
    "UTIL_MEAN",
    "UTIL_MAX",
    "BILL_TREND",
    "PAY_TO_BILL_RATIO",
]
ENGINEERED_CATEGORICAL = ["AGE_BIN"]


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Сводит недокументированные коды к категории «другое».

    EDUCATION: 1 graduate school, 2 university, 3 high school, 4 others.
    Коды 0, 5, 6 в описании датасета не расшифрованы, их всего ~1%.
    MARRIAGE: 1 married, 2 single, 3 others; код 0 тоже не описан.
    """
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].isin([1, 2, 3]), 4)
    df["MARRIAGE"] = df["MARRIAGE"].where(df["MARRIAGE"].isin([1, 2]), 3)
    return df


def add_payment_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Агрегаты по 6 месяцам истории: просрочки, загрузка лимита, погашения.

    PAY_x: -2 картой не пользовался, -1 погасил полностью,
    0 минимальный платёж (револьверный долг), 1..8 просрочка в месяцах.
    """
    df = df.copy()
    pay = df[PAY_STATUS_COLUMNS]
    bills = df[BILL_COLUMNS]
    limit = df["LIMIT_BAL"].clip(lower=1)

    df["PAY_DELAY_MAX"] = pay.max(axis=1).clip(lower=0)
    df["PAY_DELAY_COUNT"] = (pay > 0).sum(axis=1)
    df["PAY_NO_USE_COUNT"] = (pay == -2).sum(axis=1)
    df["PAY_FULL_COUNT"] = (pay == -1).sum(axis=1)

    df["UTIL_LAST"] = df["BILL_AMT1"] / limit
    df["UTIL_MEAN"] = bills.mean(axis=1) / limit
    df["UTIL_MAX"] = bills.max(axis=1) / limit
    df["BILL_TREND"] = (df["BILL_AMT1"] - df["BILL_AMT6"]) / limit

    # PAY_AMT1 — платёж в сентябре, он гасит августовский счёт BILL_AMT2.
    # Поэтому сопоставляем платежи 1..5 со счетами 2..6.
    paid = df[PAY_AMT_COLUMNS[:5]].sum(axis=1).to_numpy(dtype=float)
    owed = bills[BILL_COLUMNS[1:]].clip(lower=0).sum(axis=1).to_numpy(dtype=float)
    # Нет долга — считаем, что клиент «платит всё» (коэффициент 1).
    ratio = np.divide(paid, owed, out=np.ones_like(paid), where=owed > 0)
    df["PAY_TO_BILL_RATIO"] = np.clip(ratio, 0, 2)
    return df


def add_age_bin(df: pd.DataFrame, bins=DEFAULT_AGE_BINS) -> pd.DataFrame:
    df = df.copy()
    bins = list(bins)
    labels = [f"{lo + 1}-{hi}" for lo, hi in zip(bins[:-1], bins[1:])]
    df["AGE_BIN"] = pd.cut(df["AGE"], bins=bins, labels=labels).astype(object)
    return df


def build_features(
    df: pd.DataFrame, add_features: bool = True, age_bins=DEFAULT_AGE_BINS
) -> pd.DataFrame:
    df = normalize_categories(df[RAW_FEATURES])
    if add_features:
        df = add_payment_history_features(df)
        df = add_age_bin(df, age_bins)
    return df


def feature_columns(add_features: bool = True) -> tuple[list[str], list[str]]:
    """Списки числовых и категориальных колонок на выходе FeatureBuilder."""
    numeric = [c for c in RAW_FEATURES if c not in CATEGORICAL_RAW]
    categorical = list(CATEGORICAL_RAW)
    if add_features:
        numeric += ENGINEERED_NUMERIC
        categorical += ENGINEERED_CATEGORICAL
    return numeric, categorical


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Обёртка над build_features для использования в sklearn Pipeline."""

    def __init__(self, add_features: bool = True, age_bins=DEFAULT_AGE_BINS):
        self.add_features = add_features
        self.age_bins = age_bins

    def fit(self, X: pd.DataFrame, y=None):
        missing = set(RAW_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"Не хватает признаков: {sorted(missing)}")
        self.n_features_in_ = len(RAW_FEATURES)
        self.feature_names_in_ = np.array(RAW_FEATURES, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return build_features(X, self.add_features, self.age_bins)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        out = list(RAW_FEATURES)
        if self.add_features:
            out += ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL
        return np.array(out, dtype=object)
