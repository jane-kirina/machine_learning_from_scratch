import matplotlib.pyplot as plt
import numpy as np
import json

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

def classification_report(y_true, y_pred, y_proba):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_score": f_beta_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_proba)
    }
    return metrics

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