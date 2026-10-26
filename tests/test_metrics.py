import numpy as np
import pytest

from src.models.metrics import (
    classification_metrics,
    ks_statistic,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
)

Y_TRUE = np.array([0, 0, 0, 0, 1, 1, 0, 1])
Y_PROBA = np.array([0.1, 0.2, 0.3, 0.45, 0.4, 0.8, 0.6, 0.9])


def test_perfect_ranking():
    y = np.array([0, 0, 1, 1])
    metrics = classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["roc_auc"] == 1.0
    assert metrics["gini"] == 1.0
    assert metrics["ks"] == 1.0
    assert metrics["f1"] == 1.0


def test_gini_is_linked_to_auc():
    metrics = classification_metrics(Y_TRUE, Y_PROBA)
    assert metrics["gini"] == pytest.approx(2 * metrics["roc_auc"] - 1)


def test_threshold_controls_precision_and_recall():
    # Порог 0.5: в «дефолт» попадают 0.6, 0.8, 0.9 -> 2 из 3 верные, найдено 2 из 3.
    m05 = classification_metrics(Y_TRUE, Y_PROBA, threshold=0.5)
    assert m05["precision"] == pytest.approx(2 / 3)
    assert m05["recall"] == pytest.approx(2 / 3)
    # Порог 0.35 ловит все дефолты, но ценой лишнего ложного срабатывания.
    m035 = classification_metrics(Y_TRUE, Y_PROBA, threshold=0.35)
    assert m035["recall"] == 1.0
    assert m035["precision"] == pytest.approx(3 / 5)


def test_no_positive_predictions_does_not_crash():
    metrics = classification_metrics(Y_TRUE, Y_PROBA, threshold=0.99)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_ks_is_zero_for_constant_score():
    assert ks_statistic(Y_TRUE, np.full(len(Y_TRUE), 0.5)) == 0.0


def test_plots_are_saved(tmp_path):
    plot_roc_curve(Y_TRUE, Y_PROBA, path=tmp_path / "roc.png")
    plot_pr_curve(Y_TRUE, Y_PROBA, 0.5, path=tmp_path / "pr.png")
    plot_confusion_matrix(Y_TRUE, Y_PROBA, 0.5, path=tmp_path / "cm.png")
    plot_feature_importance(["a", "b"], [0.2, -0.1], "fi", path=tmp_path / "fi.png")
    for name in ("roc", "pr", "cm", "fi"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0
