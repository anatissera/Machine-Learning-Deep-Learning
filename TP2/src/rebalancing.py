import numpy as np
import pandas as pd

def undersampling(df, target_col, random_state=42):
    """
    Undersampling aleatorio: se reducen muestras de la clase mayoritaria para igualar proporciones.

    Parameters:
    - df (pd.DataFrame): DataFrame de entrada.
    - target_col (str): Nombre de la columna objetivo que contiene las etiquetas de clase.
    - random_state (int, optional): Semilla para la reproducibilidad del muestreo aleatorio. Por defecto es 42.
    - X (pd.DataFrame): DataFrame con las características balanceadas.
    - y (pd.Series): Serie con las etiquetas balanceadas.
    
    Returns:
    - X (pd.DataFrame): Características balanceadas por undersampling.
    - y (pd.Series): Etiquetas balanceadas por undersampling.
    
    """
    np.random.seed(random_state)
    counts = df[target_col].value_counts()
    min_class_count = counts.min()

    sampled_frames = [
        df[df[target_col] == label].sample(n=min_class_count, random_state=random_state)
        for label in counts.index
    ]

    balanced_df = pd.concat(sampled_frames).sample(frac=1, random_state=random_state).reset_index(drop=True)
    X = balanced_df.drop(columns=[target_col])
    y = balanced_df[target_col]
    return X, y

def oversampling_duplicate_minority_class(df, target_col, random_state=42):
    """
    Oversampling por duplicación: se duplican ejemplos de la clase minoritaria.

    Parameters:
    - df (pd.DataFrame): DataFrame de entrada.
    - target_col (str): Nombre de la columna objetivo que contiene las etiquetas de clase.
    - random_state (int, optional): Semilla para la reproducibilidad. Por defecto es 42.
    
    Returns:
    - X (pd.DataFrame): Características balanceadas por duplicación.
    - y (pd.Series): Etiquetas balanceadas por duplicación.
    """
    np.random.seed(random_state)
    counts = df[target_col].value_counts()
    max_class_count = counts.max()

    resampled_frames = [
        df[df[target_col] == label].sample(n=max_class_count, replace=True, random_state=random_state)
        for label in counts.index
    ]

    balanced_df = pd.concat(resampled_frames).sample(frac=1, random_state=random_state).reset_index(drop=True)
    X = balanced_df.drop(columns=[target_col])
    y = balanced_df[target_col]
    return X, y


def basic_SMOTE(df, columna_objetivo, k=5, semilla=42):
    """
    Implementación manual de SMOTE para clasificación binaria sin usar librerías externas.
    
    Parameters:
    - df (pd.DataFrame): DataFrame con las características y la columna objetivo.
    - columna_objetivo (str): Nombre de la columna que contiene las clases (binaria).
    - k (int): Número de vecinos cercanos para interpolación.
    - semilla (int): Semilla para la generación aleatoria.

    Returns:
    - X (pd.DataFrame): Datos con características extendidas (originales + sintéticos).
    - y (pd.Series): Etiquetas correspondientes.
    """
    np.random.seed(semilla)
    datos = df.copy()

    # Identificar clases y encontrar la minoritaria automáticamente
    conteos = datos[columna_objetivo].value_counts()
    clase_min = conteos.idxmin()
    clase_max = conteos.idxmax()

    # Separar datos por clase
    datos_min = datos[datos[columna_objetivo] == clase_min]
    datos_max = datos[datos[columna_objetivo] == clase_max]

    X_min = datos_min.drop(columns=[columna_objetivo]).to_numpy()
    total_sinteticos = len(datos_max) - len(datos_min)

    if total_sinteticos <= 0:
        X = datos.drop(columns=[columna_objetivo]).reset_index(drop=True)
        y = datos[columna_objetivo].reset_index(drop=True)
        return X, y

    muestras_generadas = []

    for _ in range(total_sinteticos):
        i = np.random.randint(0, len(X_min))
        base = X_min[i]

        # Calcular distancias a otros puntos minoritarios
        distancias = np.linalg.norm(X_min - base, axis=1)
        vecinos = np.argsort(distancias)[1:k+1]  # excluir el propio

        elegido = X_min[np.random.choice(vecinos)]
        factor = np.random.rand()
        nuevo = base + factor * (elegido - base)
        muestras_generadas.append(nuevo)

    # Construir DataFrame sintético
    columnas = datos_min.columns.drop(columna_objetivo)
    sinteticos = pd.DataFrame(muestras_generadas, columns=columnas)
    sinteticos[columna_objetivo] = clase_min

    # Unir y devolver
    datos_final = pd.concat([datos, sinteticos], ignore_index=True)
    datos_final = datos_final.sample(frac=1, random_state=semilla).reset_index(drop=True)

    X_final = datos_final.drop(columns=[columna_objetivo])
    y_final = datos_final[columna_objetivo]

    return X_final, y_final

def cost_sensitive_weights(y):
    """
    Cálculo de pesos para cada clase según sus proporciones (π2 / π1).

    Parameters:
    - y (pd.Series): Serie de pandas que contiene las etiquetas de clase.
    
    Returns:
    - np.ndarray: Array de pesos para cada muestra
    """
    class_counts = y.value_counts(normalize=True)
    pi_1 = class_counts.min()
    pi_2 = class_counts.max()
    weight_ratio = pi_2 / pi_1

    weights = y.apply(lambda cls: weight_ratio if class_counts[cls] == pi_1 else 1.0)
    return weights.to_numpy()
