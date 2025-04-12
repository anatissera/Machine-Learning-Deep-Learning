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

from functools import reduce
def compare_metrics(list_of_dicts, title, list_of_names=None):
    """
    Compara múltiples diccionarios de métricas y muestra una tabla en formato Markdown en Jupyter.

    Parameters:
    - list_of_dicts (list of dict): lista de diccionarios con claves "Métrica" y "Valor"
    - title (str): título a mostrar encima de la tabla
    - list_of_names (list of str, optional): lista de nombres para los modelos (por defecto "Modelo 1", "Modelo 2", etc.)
    
    Returns:
    - None
    """
    if list_of_names is None:
        list_of_names = [f"Model {i+1}" for i in range(len(list_of_dicts))]

    dfs = []
    for i, (d, name) in enumerate(zip(list_of_dicts, list_of_names)):
        df = pd.DataFrame(d).copy()
        df.rename(columns={"Valor": f"Value ({name})"}, inplace=True)
        dfs.append(df)

    comparison_df = reduce(lambda left, right: pd.merge(left, right, on="Métrica"), dfs)
    comparison_df.rename(columns={"Métrica": "Metric"}, inplace=True)
    comparison_df = comparison_df.round(4)

    display(Markdown(f"### {title}"))
    display(Markdown(comparison_df.to_markdown(index=False)))