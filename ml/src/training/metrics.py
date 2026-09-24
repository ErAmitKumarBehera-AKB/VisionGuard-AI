from pathlib import Path
from typing import Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..utils.logging import get_logger

logger = get_logger(__name__)


def calculate_binary_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_prob: Optional[list[float] | np.ndarray] = None,
) -> dict[str, Any]:
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))

    auc = None
    if y_prob is not None and len(np.unique(y_true_arr)) > 1:
        try:
            auc = float(roc_auc_score(y_true_arr, np.array(y_prob)))
        except Exception as e:
            logger.warning(f"Failed to calculate ROC-AUC: {e}")

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": acc,
        "defect_precision": prec,
        "defect_recall": rec,
        "defect_f1": f1,
        "roc_auc": auc,
        "ok_specificity": specificity,
        "confusion_matrix": {
            "true_negatives_ok": int(tn),
            "false_positives_defect_alarm": int(fp),
            "false_negatives_missed_defect": int(fn),
            "true_positives_detected_defect": int(tp),
        },
        "total_samples": len(y_true_arr),
        "actual_defect_count": int(np.sum(y_true_arr == 1)),
        "actual_ok_count": int(np.sum(y_true_arr == 0)),
    }


def plot_confusion_matrix(
    cm_counts: list[list[int]] | np.ndarray,
    output_path: str | Path,
    title: str = "Manufacturing Defect Confusion Matrix",
    labels: Optional[list[str]] = None,
) -> Path:
    labels = labels or ["OK", "DEFECT"]
    cm = np.array(cm_counts)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
    ax.set_ylabel("Ground Truth Label", fontsize=10, labelpad=8)
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)

    logger.info(f"Confusion matrix plot saved to {out}")
    return out
