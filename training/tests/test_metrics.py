from pathlib import Path
import pytest

from training.src.training.metrics import calculate_binary_metrics, plot_confusion_matrix


def test_metrics_calculation():
    y_true = [0, 0, 1, 1, 1, 0, 1, 0]
    y_pred = [0, 0, 1, 1, 0, 0, 1, 0]
    y_prob = [0.1, 0.2, 0.9, 0.85, 0.4, 0.05, 0.95, 0.15]

    metrics = calculate_binary_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 7 / 8
    assert pytest.approx(metrics["defect_recall"], 0.01) == 0.75
    assert pytest.approx(metrics["defect_precision"], 0.01) == 1.0
    assert metrics["confusion_matrix"]["false_negatives_missed_defect"] == 1
    assert metrics["confusion_matrix"]["true_positives_detected_defect"] == 3
    assert metrics["confusion_matrix"]["true_negatives_ok"] == 4
    assert metrics["confusion_matrix"]["false_positives_defect_alarm"] == 0
    assert metrics["roc_auc"] is not None


def test_plot_confusion_matrix(tmp_path: Path):
    cm = [[10, 2], [1, 12]]
    out_file = tmp_path / "cm_test.png"
    saved = plot_confusion_matrix(cm, out_file)

    assert saved.is_file()
    assert saved.stat().st_size > 0
