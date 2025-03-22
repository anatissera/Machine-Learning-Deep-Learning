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
    
from models import log_predict

def complete_missing_rooms_values(df, W, b, mean_train, std_train):
    """Completa los valores faltantes en la columna 'rooms' usando el modelo entrenado."""
    # 1. Filtrar filas con valores faltantes en "rooms"
    df_faltantes = df[df['rooms'].isna()].copy()
    
    # 2. Tomar X (area) de las filas faltantes
    X_faltantes = df_faltantes[['area']].values

    # 3. Normalizar X usando las estadísticas de entrenamiento
    X_faltantes = (X_faltantes - mean_train) / std_train

    # 4. Predecir los valores de "rooms"
    y_pred_faltantes = log_predict(X_faltantes, W, b)

    # 5. Insertar los valores predichos en el dataframe original
    df.loc[df['rooms'].isna(), 'rooms'] = y_pred_faltantes

    print(f"{len(df_faltantes)} valores faltantes en 'rooms' completados.")
    return df