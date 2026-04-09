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