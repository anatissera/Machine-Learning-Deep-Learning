import pandas as pd
import numpy as np
from IPython.display import display, Markdown

def missing_values(df, dataset_name):
    """
    Genera y muestra una tabla en formato Markdown con la cantidad de valores faltantes por columna 
    en un DataFrame dado.
    
    Parameters:
    - df (pd.DataFrame): DataFrame de entrada que será analizado para detectar valores faltantes.
    - dataset_name (str): Nombre del conjunto de datos, utilizado para el título de la tabla.
    
    Returns:
    - None
    """
    missing = df.isna().sum().sort_values(ascending=False)
    
    table_md = f"### Valores faltantes por columna en el **{dataset_name}** Set\n\n"
    table_md += "| Columna | Valores faltantes |\n"
    table_md += "|---------|-------------------|\n"
    for col, count in missing.items():
        table_md += f"| {col} | {count} |\n"

    display(Markdown(table_md))
    
def duplicated_rows(df, dataset_name):
    """
    Genera y muestra una tabla en formato Markdown que indica la cantidad de filas duplicadas en un DataFrame.
    
    Parameters:
    - df (pd.DataFrame): DataFrame de entrada que será analizado.
    - dataset_name (str): Nombre del conjunto de datos, utilizado para el título de la tabla.
    
    Returns:
    - None
    """
    num_duplicates = df.duplicated().sum()

    table_md = f"### Filas duplicadas en el **{dataset_name}** Set\n\n"
    table_md += "| Total de filas | Filas duplicadas |\n"
    table_md += "|----------------|------------------|\n"
    table_md += f"| {len(df)} | {num_duplicates} |\n"

    display(Markdown(table_md))

def describe_feature_ranges(datasets, dataset_names=None, cat_threshold=10):
    """
    Genera y muestra una tabla en formato Markdown con un resumen de los rangos 
    o valores únicos de las características presentes en múltiples datasets.

    Parameters:
    - datasets (list of pd.DataFrame): Lista de DataFrames que contienen los datos a analizar.
    - dataset_names (list of str, opcional): Nombres de los datasets para identificar las características. Si no se proporciona, se generarán nombres genéricos como "Dataset 1", "Dataset 2", etc.
    - cat_threshold (int, opcional): Umbral para considerar una característica numérica como categórica si tiene un número de valores únicos menor o igual a este valor. Por defecto es 10.
    
    Returns:
    - None
    """
    combined = pd.concat(datasets, axis=0, ignore_index=True)

    if dataset_names is None:
        dataset_names = [f"Dataset {i+1}" for i in range(len(datasets))]

    rows = []

    for col in combined.columns:
        is_categorical = (
            isinstance(combined[col].dtype, pd.CategoricalDtype) or 
            pd.api.types.is_object_dtype(combined[col]) or 
            (pd.api.types.is_numeric_dtype(combined[col]) and combined[col].nunique() <= cat_threshold)
        )

        if is_categorical:
            unique_vals = set()
            for df in datasets:
                unique_vals.update(df[col].dropna().unique())
            val = f"{sorted(list(unique_vals))}"
        elif pd.api.types.is_numeric_dtype(combined[col]):
            min_val = combined[col].min()
            max_val = combined[col].max()
            val = f"[{min_val:.2f} → {max_val:.2f}]"
        else:
            val = "Tipo de dato no reconocido"

        rows.append((col, "Categórica" if is_categorical else "Numérica", val))

    summary_df = pd.DataFrame(rows, columns=["Feature", "Tipo", "Rango o Valores Únicos"])
    markdown_table = summary_df.to_markdown(index=False)
    display(Markdown("### Rango de valores por feature en todos los datasets:\n" + markdown_table))


def detect_outliers(df, true_intervals):
    """
    Detecta outliers en un DataFrame en función de los intervalos válidos definidos
    y muestra un resumen en formato tabla Markdown.

    Parameters:
    - df (pd.DataFrame): DataFrame de entrada.
    - true_intervals (dict): Diccionario con los intervalos válidos para cada columna.

    Returns:
    - pd.DataFrame: DataFrame booleano con True donde hay outliers y False donde no.
    """
    outliers = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.columns:
        if col in true_intervals:
            min_val, max_val = true_intervals[col]
            outliers[col] = (df[col] < min_val) | (df[col] > max_val)

    # outliers `por columna
    outlier_counts = outliers.sum().reset_index()
    outlier_counts.columns = ['Feature', 'Cantidad de Outliers']

    markdown_table = outlier_counts.to_markdown(index=False)
    display(Markdown("### Cantidad de outliers por feature:\n" + markdown_table))

    return outliers

def class_balance(df, target_column, dataset_name):
    """
    Genera una tabla en formato Markdown que muestra la distribución o desbalanceo de clases 
    en un conjunto de datos específico, basado en una columna objetivo.
    
    Parameters:
    - df (pd.DataFrame): DataFrame de entrada que contiene los datos.
    - target_column (str): Nombre de la columna objetivo que contiene las clases.
    - dataset_name (str): Nombre del conjunto de datos (por ejemplo, "Train", "Test").
    
    Returns:
    - None: La función no retorna ningún valor, pero muestra la tabla en formato Markdown.
    """
    class_counts = df[target_column].value_counts()
    total = len(df)
    
    table_md = f"### Distribución/Desbalanceo de clases en el **{dataset_name}** Set (columna: `{target_column}`)\n\n"
    table_md += "| Clase | Frecuencia | Porcentaje |\n"
    table_md += "|--------|------------|------------|\n"

    for class_value, count in class_counts.items():
        percent = (count / total) * 100
        table_md += f"| {class_value} | {count} | {percent:.2f}% |\n"

    display(Markdown(table_md))