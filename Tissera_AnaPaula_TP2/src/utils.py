import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from itertools import product
def run_grid_search(ModelClass, param_grid, X_train, y_train, X_val, y_val, metric_fn):
    """
    Ejecuta una búsqueda de grilla para encontrar los mejores hiperparámetros de un modelo.

    Parameters:
    - ModelClass (class): Clase del modelo (no instanciado).
    - param_grid (dict): Diccionario con listas de valores para cada hiperparámetro.
    - X_train (pd.DataFrame o np.ndarray): Datos de entrenamiento (features).
    - y_train (pd.Series o np.ndarray): Etiquetas de entrenamiento.
    - X_val (pd.DataFrame o np.ndarray): Datos de validación (features).
    - y_val (pd.Series o np.ndarray): Etiquetas de validación.
    - metric_fn (function): Función que recibe (y_true, y_pred) y devuelve el score.

    Returns:
    - best_model (object): Mejor modelo entrenado.
    - best_score (float): Mejor score obtenido.
    - best_params (dict): Mejores hiperparámetros encontrados.
    """
    best_score = -np.inf
    best_model = None
    best_params = {}

    keys = list(param_grid.keys())
    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))
        model = ModelClass(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        score = metric_fn(y_val, y_pred)

        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    return best_model, best_score, best_params


def comparar_metricas(dict1, dict2, title, nombre1="Modelo 1", nombre2="Modelo 2"):
    """
    Compara dos diccionarios de métricas y muestra una tabla en formato Markdown en Jupyter.

    Parámetros:
        dict1, dict2: diccionarios con claves "Métrica" y "Valor"
        nombre1, nombre2: nombres que se usarán en la tabla para identificar cada conjunto de métricas
    """
    df1 = pd.DataFrame(dict1)
    df2 = pd.DataFrame(dict2)

    df_comparacion = pd.merge(df1, df2, on="Métrica", suffixes=(f" ({nombre1})", f" ({nombre2})"))
    df_comparacion = df_comparacion.round(4)

    display(Markdown(f"### {title}"))

    tabla_md = df_comparacion.to_markdown(index=False)
    display(Markdown(tabla_md))