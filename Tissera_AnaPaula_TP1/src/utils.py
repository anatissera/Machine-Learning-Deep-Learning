import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def missing_values(df):
    print("\nValores faltantes por columna:")
    print(df.isna().sum())

    
def missing_percentages(df):
    total_filas = len(df)
    
    n_nan_age = df["age"].isna().sum()
    porcentaje_nan_age = (n_nan_age / total_filas) * 100
    
    n_nan_rooms = df["rooms"].isna().sum()
    porcentaje_nan_rooms = (n_nan_rooms / total_filas) * 100

    print(f"Valores NaN en 'age': {n_nan_age} sobre un total de {total_filas} filas ({porcentaje_nan_age:.2f}%)")
    print(f"Valores NaN en 'rooms': {n_nan_rooms} sobre un total de {total_filas} filas ({porcentaje_nan_rooms:.2f}%)")

def save_csv(df, nombre_archivo):
    """Guarda un DataFrame en un archivo CSV."""
    df.to_csv(nombre_archivo, index=False)
    print(f"Dataset guardado como '{nombre_archivo}'.")


def complete_data(df, to_drop):
    """Elimina filas con valores faltantes en las columnas especificadas."""
    return df.dropna(subset=to_drop)

def normalize_given_μ_σ(X_new, mean_train, std_train):
    """Normaliza nuevos datos usando los valores de X_train."""
    return (X_new - mean_train) / (std_train + 1e-8)

def add_bias(X):
    """Añade una columna de unos para el término de sesgo (intercepto)."""
    return np.c_[np.ones(X.shape[0]), X]

def generate_polynomial_features(X, grado=1):
    """Genera términos polinómicos hasta el grado especificado."""
    return np.hstack([X ** g for g in range(1, grado + 1)])

def load_data(ruta, features, target):
    import pandas as pd
    df = pd.read_csv(ruta)
    X = df[features].values
    y = df[target].values
    return X, y
