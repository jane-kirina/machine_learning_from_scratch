import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd

from ml_from_scratch.metrics import (mae_score, mse_score, r2_score,
                                     accuracy_score, precision_score, recall_score, f_beta_score, roc_auc_score)

# --------
# METRICS REPORTS
# --------
def regression_report(y_true, y_pred):
    return {
        "MAE": mae_score(y_true, y_pred),
        "MSE": mse_score(y_true, y_pred),
        "RMSE": np.sqrt(mse_score(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

def classification_metrics_report(y_true, y_pred, y_proba):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1_score": f_beta_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_proba)
    }
    return metrics

def classification_report(y_true, y_pred, digits=4): # TODO
    pass

# --------
# BUILDERS FOR REPORTS
# --------
def build_classification_comparison_df(scratch_metrics, sklearn_metrics, tuned_metrics=None):
    comparison = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1_score", "ROC_AUC"],
        "Scratch": [
            scratch_metrics["Accuracy"],
            scratch_metrics["Precision"],
            scratch_metrics["Recall"],
            scratch_metrics["F1_score"],
            scratch_metrics["ROC_AUC"],
        ],
        "Sklearn": [
            sklearn_metrics["Accuracy"],
            sklearn_metrics["Precision"],
            sklearn_metrics["Recall"],
            sklearn_metrics["F1_score"],
            sklearn_metrics["ROC_AUC"],
        ],
    })

    if tuned_metrics is not None:
        comparison["Tuned Scratch"] = comparison["Metric"].map({
            "Accuracy": tuned_metrics["Accuracy"],
            "Precision": tuned_metrics["Precision"],
            "Recall": tuned_metrics["Recall"],
            "F1_score": tuned_metrics["F1_score"],
            "ROC_AUC": tuned_metrics["ROC_AUC"],
        })

    return comparison

# --------
# PRINT & SAVE REPORTS
# --------
def print_report(title, metrics):
    print(title)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("-" * 40)

def save_metrics(filename, metrics_dict):
    with open(filename, "w") as f:
        json.dump(metrics_dict, f, indent=4)

# --------
# VISUAL
# --------
def plot_predictions(y_true, y_pred, title, save_path):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title(title)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.show()