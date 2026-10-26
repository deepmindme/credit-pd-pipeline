"""Стадия evaluate: финальная оценка выбранной модели на отложенном test.

Результаты:
  reports/metrics.json         — метрики для `dvc metrics show`;
  reports/plots/roc.csv        — точки ROC-кривой для `dvc plots`;
  reports/score_deciles.csv    — таблица по децилям скора (gains table);
  reports/figures/*.png        — ROC, PR, матрица ошибок, распределение скоров;
  models/reference_profile.json — эталонные распределения для мониторинга PSI.

Запуск: python -m src.models.evaluate
"""

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.config import RAW_FEATURES, TARGET, load_params, resolve
from src.models.predict import load_model
from src.models.metrics import (
    classification_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from src.monitoring.drift import build_reference_profile

logger = logging.getLogger(__name__)


def decile_table(y_true, y_proba) -> pd.DataFrame:
    """Клиенты делятся на 10 групп по убыванию PD. Хорошая модель концентрирует
    дефолты в верхних децилях (cum_default_capture быстро растёт)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "pd": np.asarray(y_proba)})
    df["decile"] = pd.qcut(df["pd"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    table = df.groupby("decile", observed=True).agg(
        clients=("y", "size"),
        defaults=("y", "sum"),
        pd_min=("pd", "min"),
        pd_max=("pd", "max"),
        pd_mean=("pd", "mean"),
    )
    table["default_rate"] = table["defaults"] / table["clients"]
    table["cum_default_capture"] = table["defaults"].cumsum() / table["defaults"].sum()
    return table.reset_index()


def plot_score_distribution(y_true, y_proba, threshold: float, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 41)
    y_true = np.asarray(y_true)
    ax.hist(y_proba[y_true == 0], bins=bins, alpha=0.6, density=True, label="нет дефолта")
    ax.hist(y_proba[y_true == 1], bins=bins, alpha=0.6, density=True, label="дефолт")
    ax.axvline(threshold, color="black", ls="--", label=f"порог {threshold:.2f}")
    ax.set(xlabel="Предсказанная PD", ylabel="Плотность", title="Распределение скоров на test")
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    params = load_params()
    processed_dir = resolve(params["data"]["processed_dir"])
    reports_dir = resolve("reports")
    figures_dir = reports_dir / "figures"
    plots_dir = reports_dir / "plots"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(resolve("models/model.joblib"))
    meta, threshold = model.meta, model.threshold

    train = pd.read_csv(processed_dir / "train.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    proba = model.predict_proba(test)

    metrics = classification_metrics(test[TARGET], proba, threshold)
    metrics = {"model": meta["experiment"], **{k: round(v, 4) for k, v in metrics.items()}}
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fpr, tpr, _ = roc_curve(test[TARGET], proba)
    # Для dvc plots достаточно ~200 точек, полная кривая — тысячи строк.
    idx = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))
    pd.DataFrame({"fpr": fpr[idx], "tpr": tpr[idx]}).round(4).to_csv(
        plots_dir / "roc.csv", index=False
    )
    decile_table(test[TARGET], proba).round(4).to_csv(
        reports_dir / "score_deciles.csv", index=False
    )

    title = f"ROC-кривая на test — {meta['experiment']}"
    plot_roc_curve(test[TARGET], proba, title, figures_dir / "roc_curve.png")
    plot_pr_curve(test[TARGET], proba, threshold, figures_dir / "pr_curve.png")
    plot_confusion_matrix(test[TARGET], proba, threshold, figures_dir / "confusion_matrix.png")
    plot_score_distribution(test[TARGET], proba, threshold, figures_dir / "score_distribution.png")

    train_scores = model.predict_proba(train)
    profile = build_reference_profile(
        train[RAW_FEATURES], train_scores, RAW_FEATURES, params["monitoring"]["psi_bins"]
    )
    resolve("models/reference_profile.json").write_text(json.dumps(profile, indent=2))
    logger.info("Метрики на test: %s", metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
