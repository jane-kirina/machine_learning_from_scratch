#Imports
import numpy as np

# Util methods for metrics
def mae_score(y_true, y_pred):  # Mean Absolute Error
    return np.mean(np.abs(y_true - y_pred))

def mse_score(y_true, y_pred): # Mean Squared Error
    return np.mean((y_true - y_pred) ** 2) 

def r2_score(y_true, y_pred):  # R² score
    mean_y = np.mean(y_true)
    ss_total = np.sum((y_true - mean_y) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total) if ss_total != 0 else 0

def accuracy_score(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    return correct_predictions / len(y_true)

def precision_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    denominator = true_positive + false_positive
    return true_positive / denominator if denominator > 0 else 0

def recall_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    denominator = true_positive + false_negative
    return true_positive / denominator if denominator > 0 else 0

def f_beta_score(y_true, y_pred, beta = 1):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    beta_2 = np.square(beta)
    denominator = (beta_2 * precision) + recall
    return (1 + beta_2) * (precision * recall) / denominator if denominator > 0 else 0

def roc_auc_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        return 0.0

    count = 0.0
    for pos in pos_scores:
        for neg in neg_scores:
            if pos > neg:
                count += 1
            elif pos == neg:
                count += 0.5

    return count / (n_pos * n_neg)
