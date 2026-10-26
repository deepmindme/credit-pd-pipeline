"""Метрики бинарной классификации и графики для отчётов."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true, y_proba) -> float:
    """Kolmogorov–Smirnov: максимальный разрыв между TPR и FPR."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_proba)
    return {
        "roc_auc": float(auc),
        # Gini = 2·AUC − 1, в кредитном скоринге принято смотреть его.
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_true, y_proba),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def _save(fig, path: Path | None) -> plt.Figure:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_roc_curve(y_true, y_proba, title: str = "ROC-кривая", path: Path | None = None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"модель (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="случайный прогноз")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_pr_curve(y_true, y_proba, threshold: float, path: Path | None = None):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    idx = np.argmin(np.abs(thresholds - threshold))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recall, precision, lw=2)
    ax.scatter(recall[idx], precision[idx], color="red", zorder=3, label=f"порог {threshold:.2f}")
    ax.axhline(np.mean(y_true), ls="--", color="grey", label="доля дефолтов")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_confusion_matrix(y_true, y_proba, threshold: float, path: Path | None = None):
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["нет дефолта", "дефолт"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Матрица ошибок (порог {threshold:.2f})")
    return _save(fig, path)


def plot_feature_importance(names, values, title: str, path: Path | None = None, top: int = 20):
    order = np.argsort(np.abs(values))[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(order) + 1))
    ax.barh(np.asarray(names)[order], np.asarray(values)[order])
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)
