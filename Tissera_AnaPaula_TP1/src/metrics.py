import numpy as np


def calculate_rmse(y_real, y_pred):
    """Calcula el error cuadrático medio (RMSE)."""
    return np.sqrt(np.mean((y_real - y_pred) ** 2))
