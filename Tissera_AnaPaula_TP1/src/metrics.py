import numpy as np

def calculate_mse(y_real, y_pred):
    """Calcula el error cuadrático medio (MSE)."""
    return np.mean((y_real - y_pred) ** 2)

def calculate_rmse(y_real, y_pred):
    """Calcula la raíz del error cuadrático medio (RMSE)."""
    return np.sqrt(calculate_mse(y_real, y_pred))

def calculate_mae(y_real, y_pred):
    """Calcula el error absoluto medio (MAE)."""
    return np.mean(np.abs(y_real - y_pred))

def calculate_r2(y_real, y_pred):
    """Calcula el coeficiente de determinación R^2."""
    ss_total = np.sum((y_real - np.mean(y_real)) ** 2)
    ss_residual = np.sum((y_real - y_pred) ** 2)
    return 1 - (ss_residual / ss_total) if ss_total != 0 else 0