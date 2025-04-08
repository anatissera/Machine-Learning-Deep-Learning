import pandas as pd
import numpy as np

def normalize_dataframe(df, is_training=True, stats=None, target_col='diagnosis'):
    """
    Standardizes numerical features by subtracting the mean and dividing by the standard deviation.
    Saves the mean and std in a dictionary for consistent transformation during validation.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - is_training (bool): Whether the function is being called on training data.
    - stats (dict): Dictionary to store or reuse mean and std values.
    - target_col (str): The name of the target column to exclude from normalization.

    Returns:
    - pd.DataFrame: DataFrame with normalized numeric features.
    - dict: Dictionary containing the mean and std of each normalized feature.
    """
    if stats is None:
        stats = {}

    df_copy = df.copy()
    numeric_columns = df.select_dtypes(include='number').columns

    for col in numeric_columns:
        if col == target_col:
            continue

        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1}):
            continue  # Skip binary/categorical columns

        if is_training:
            mean_val = df[col].mean()
            std_val = df[col].std()
            stats[col] = {'mean': mean_val, 'std': std_val}
        else:
            mean_val = stats[col]['mean']
            std_val = stats[col]['std']

        df_copy[col] = (df[col] - mean_val) / std_val

    return df_copy, stats

def one_hot_encode_column(df, column, corrections=None, drop_first=False):
    """
    Aplica one-hot encoding a una única columna, corrigiendo valores y devolviendo 0s y 1s.

    Parameters:
    - df (pd.DataFrame): El DataFrame de entrada.
    - column (str): Nombre de la columna a codificar.
    - corrections (dict): Diccionario opcional de correcciones {valor_mal: valor_bien}.
    - drop_first (bool): Si se desea eliminar la primera categoría para evitar colinealidad.

    Returns:
    - pd.DataFrame: DataFrame con la columna codificada como 0/1.
    """
    df_copy = df.copy()

    if corrections:
        df_copy[column] = df_copy[column].replace(corrections)

    dummies = pd.get_dummies(df_copy[column], prefix=column, drop_first=drop_first)

    dummies = dummies.astype(int)

    df_copy = pd.concat([df_copy.drop(columns=[column]), dummies], axis=1)

    return df_copy


def binary_encode_column(df, column, mapping):
    """
    Codifica una columna binaria usando un mapeo explícito, devolviendo 0s y 1s sin warnings.

    Parameters:
    - df (pd.DataFrame): DataFrame de entrada.
    - column (str): Columna a codificar.
    - mapping (dict): Diccionario de mapeo, ej. {'Presnt': 1, 'Absnt': 0}

    Returns:
    - pd.DataFrame: DataFrame con la columna codificada como enteros.
    """
    df_copy = df.copy()

    # Reemplazar y luego convertir a tipo int de forma explícita
    df_copy[column] = df_copy[column].replace(mapping)
    
    if df_copy[column].isnull().any():
        raise ValueError(f"Valores no reconocidos en columna '{column}' después de aplicar el mapeo: {df_copy[column].unique()}")

    # Esto evita el FutureWarning:
    df_copy[column] = pd.to_numeric(df_copy[column], errors="raise", downcast="integer")

    return df_copy



def handle_missing_values(data, target_col, train=True, reference=None, intervals=None, stats_dict=None, neighbors=5):
    """
    Maneja valores faltantes reemplazando con media/moda o imputación KNN.

    Parameters:
    - data (pd.DataFrame): DataFrame de entrada.
    - target_col (str): Nombre de la variable objetivo, que se excluye del normalizado.
    - train (bool): Si se está en fase de entrenamiento.
    - reference (pd.DataFrame): DF de referencia para imputación si test.
    - intervals (dict): Rango válido por variable. Fuera de rango se reemplaza por NaN.
    - stats_dict (dict): Diccionario con estadísticas precomputadas si test.
    - neighbors (int): Número de vecinos para imputación KNN.

    Returns:
    - pd.DataFrame: DataFrame limpio.
    """
    clean_data = enforce_valid_ranges(data, intervals or {})
    filled_data = clean_data.copy()

    if stats_dict is None:
        stats_dict = {}

    for col in filled_data.columns:
        if filled_data[col].isna().sum() == 0:
            continue

        if train:
            if filled_data[col].dtype == object or filled_data[col].dtype == bool:
                mode_val = filled_data[col].mode()[0]
                stats_dict[col] = {"mode": mode_val}
                filled_data[col] = filled_data[col].fillna(mode_val)
            else:
                mean_val = filled_data[col].mean()
                stats_dict[col] = {"mean": mean_val}
                filled_data[col] = filled_data[col].fillna(mean_val)
        else:
            if filled_data[col].dtype == object or filled_data[col].dtype == bool:
                filled_data[col] = filled_data[col].fillna(stats_dict[col]["mode"])
            else:
                filled_data[col] = filled_data[col].fillna(stats_dict[col]["mean"])

    result = knn_impute_missing(
        df=filled_data,
        reference=filled_data if train else reference, # No se puede usar el dataset de test como referencia porque no hay que aprender del test
        base=clean_data,
        stats=stats_dict,
        k=neighbors,
        target_col=target_col
    )

    return result


def enforce_valid_ranges(dataframe, valid_ranges):
    """
    Reemplaza valores fuera de los rangos válidos por NaN.

    Parameters:
    - dataframe (pd.DataFrame): Datos a validar.
    - valid_ranges (dict): Diccionario con límites {col: (min, max)}.

    Returns:
    - pd.DataFrame: Con valores fuera de rango reemplazados por NaN.
    """
    df = dataframe.copy()
    for feature, (low, high) in valid_ranges.items():
        out_of_bounds = (df[feature] < low) | (df[feature] > high)
        df.loc[out_of_bounds, feature] = np.nan
    return df

def knn_impute_missing(df, reference, base, stats, k, target_col):
    """
    Imputa valores faltantes usando vecinos más cercanos.

    Parameters:
    - df (pd.DataFrame): Datos a imputar.
    - reference (pd.DataFrame): Conjunto de comparación.
    - base (pd.DataFrame): Versión original con NaNs.
    - stats (dict): Estadísticas para normalizar.
    - k (int): Número de vecinos.
    - target_col (str): Columna objetivo a excluir del procesamiento.

    Returns:
    - pd.DataFrame: Con imputaciones.
    """
    from tqdm import tqdm

    df_filled = df.copy()
    num_features = df.select_dtypes(include=["float64", "int64"]).columns
    cat_features = df.select_dtypes(include=["object", "bool"]).columns

    norm_df = df.copy()
    norm_ref = reference.copy()

    for col in num_features:
        if col == target_col or df[col].nunique() <= 2:
            continue
        mean = stats[col]['mean']
        std = stats[col].get('std', df[col].std() or 1)
        norm_df[col] = (df[col] - mean) / std
        norm_ref[col] = (reference[col] - mean) / std

    ref_values = norm_ref[num_features].to_numpy()
    df_values = norm_df[num_features].to_numpy()

    for idx in tqdm(base.index[base.isnull().any(axis=1)], desc="KNN imputing"):
        row = df_values[df.index.get_loc(idx)]
        mask = ~np.isnan(row)

        if not mask.any():
            continue

        distances = np.linalg.norm(ref_values[:, mask] - row[mask], axis=1)
        nearest = reference.iloc[np.argpartition(distances, k)[:k]]

        for column in base.columns[base.loc[idx].isna()]:
            valid_neighbors = nearest[column].dropna()
            if not valid_neighbors.empty:
                if column in num_features:
                    df_filled.loc[idx, column] = valid_neighbors.mean()
                else:
                    df_filled.loc[idx, column] = valid_neighbors.mode().iloc[0]

    return df_filled

def convert_columns_to_int(df, columns):
    """
    Convierte las columnas especificadas de float a int si están presentes en el DataFrame.

    Parameters:
    - df: DataFrame de pandas
    - columns: lista de nombres de columnas a convertir

    Returns:
    - DataFrame con las columnas convertidas a int
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df
