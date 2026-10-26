"""Population Stability Index (PSI) для мониторинга дрифта.

PSI = Σ (actual_i − expected_i) · ln(actual_i / expected_i) по бинам.
Общепринятая в скоринге шкала:
  PSI < 0.1          — распределение стабильно;
  0.1 ≤ PSI < 0.25   — умеренный сдвиг, стоит присмотреться;
  PSI ≥ 0.25         — значимый дрифт, модель нужно перепроверять.

Эталон (expected) хранится в reference_profile.json. Его строит стадия
evaluate по train, так что мониторингу не нужен доступ к обучающим данным.
"""

import numpy as np
import pandas as pd

EPS = 1e-4
# Признаки с небольшим числом уникальных значений (PAY_x, EDUCATION и т.п.)
# считаем категориальными: квантильные бины на них схлопываются.
MAX_CATEGORIES = 15


def psi_from_proportions(expected, actual) -> float:
    expected = np.clip(np.asarray(expected, dtype=float), EPS, None)
    actual = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_status(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"


def numeric_profile(values, n_bins: int = 10) -> dict:
    values = np.asarray(values, dtype=float)
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    cuts = np.unique(np.quantile(values, quantiles))
    return {
        "type": "numeric",
        "cuts": cuts.tolist(),
        "proportions": _numeric_proportions(values, cuts).tolist(),
    }


def _numeric_proportions(values, cuts) -> np.ndarray:
    # Бины (-inf, c1], (c1, c2], ..., (ck, +inf)
    idx = np.searchsorted(np.asarray(cuts), np.asarray(values, dtype=float), side="left")
    counts = np.bincount(idx, minlength=len(cuts) + 1)
    return counts / max(counts.sum(), 1)


def categorical_profile(values) -> dict:
    shares = pd.Series(values).value_counts(normalize=True).sort_index()
    return {
        "type": "categorical",
        "categories": [_to_builtin(c) for c in shares.index],
        "proportions": shares.tolist(),
    }


def _to_builtin(value):
    return value.item() if hasattr(value, "item") else value


def build_profile(values, n_bins: int = 10) -> dict:
    values = pd.Series(values)
    if values.nunique() <= MAX_CATEGORIES:
        return categorical_profile(values)
    return numeric_profile(values, n_bins)


def proportions_for(profile: dict, values) -> np.ndarray:
    if profile["type"] == "numeric":
        return _numeric_proportions(values, profile["cuts"])
    shares = pd.Series(values).value_counts(normalize=True)
    # Новые категории, которых не было в эталоне, не теряем: они сильно
    # увеличивают PSI через «пустую» эталонную долю.
    known = [shares.get(c, 0.0) for c in profile["categories"]]
    unseen = float(shares.drop(labels=profile["categories"], errors="ignore").sum())
    return np.array(known + ([unseen] if unseen > 0 else []))


def psi(profile: dict, values) -> float:
    actual = proportions_for(profile, values)
    expected = list(profile["proportions"])
    if len(actual) > len(expected):
        expected.append(0.0)
    return psi_from_proportions(expected, actual)


def build_reference_profile(X: pd.DataFrame, scores, features: list[str], n_bins: int = 10) -> dict:
    return {
        "n_rows": int(len(X)),
        "score": numeric_profile(scores, n_bins),
        "features": {col: build_profile(X[col], n_bins) for col in features},
    }


def drift_report(profile: dict, X: pd.DataFrame, scores) -> pd.DataFrame:
    rows = [{"variable": "score", "psi": psi(profile["score"], scores)}]
    for col, col_profile in profile["features"].items():
        rows.append({"variable": col, "psi": psi(col_profile, X[col])})
    report = pd.DataFrame(rows)
    report["status"] = report["psi"].map(psi_status)
    return report
