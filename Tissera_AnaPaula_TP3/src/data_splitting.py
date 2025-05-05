import numpy as np

def split_data(X, y, ratio=0.2, random_seed=None):
    """
    Divide X e y en conjuntos dev y test, manteniendo el emparejamiento.

    Parámetros
    ----------
    X : array-like, shape (N, ...)
        Datos de entrada.
    y : array-like, shape (N,)
        Etiquetas asociadas.
    ratio : float, opcional (default=0.2)
        Fracción de ejemplos que irán al conjunto *dev*. Debe estar entre 0 y 1.
    random_seed : int o None, opcional
        Semilla para el muestreo aleatorio. Si es None, usa la aleatoriedad global.

    Devuelve
    -------
    X_dev, X_test, y_dev, y_test : tuple de arrays
        - X_dev, y_dev: datos para desarrollo/validación  
        - X_test, y_test: datos para test
    """
    # Validaciones básicas
    if not (0.0 < ratio < 1.0):
        raise ValueError("`ratio` debe estar entre 0 y 1 (excluyendo extremos)")
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X e y deben tener el mismo número de muestras en la primera dimensión")

    # Generar permutación reproducible
    rng = np.random.RandomState(random_seed)
    perm = rng.permutation(X.shape[0])

    # Índice de corte
    n_dev = int(np.floor(ratio * X.shape[0]))

    # Seleccionar índices
    idx_dev  = perm[:n_dev]
    idx_test = perm[n_dev:]

    # Partir los arrays
    X_dev,  X_test  = X[idx_dev],  X[idx_test]
    y_dev,  y_test  = y[idx_dev],  y[idx_test]

    return X_dev, X_test, y_dev, y_test
