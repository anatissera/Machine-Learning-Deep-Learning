import pandas as pd
import numpy as np
from IPython.display import display, Markdown

def missing_values(df, dataset_name):
    missing = df.isna().sum().sort_values(ascending=False)
    
    table_md = f"### Valores faltantes por columna en el **{dataset_name}** Set\n\n"
    table_md += "| Columna | Valores faltantes |\n"
    table_md += "|---------|-------------------|\n"
    for col, count in missing.items():
        table_md += f"| {col} | {count} |\n"

    display(Markdown(table_md))
    
def duplicated_rows(df, dataset_name):
    num_duplicates = df.duplicated().sum()

    table_md = f"### Filas duplicadas en el **{dataset_name}** Set\n\n"
    table_md += "| Total de filas | Filas duplicadas |\n"
    table_md += "|----------------|------------------|\n"
    table_md += f"| {len(df)} | {num_duplicates} |\n"

    display(Markdown(table_md))

def describe_feature_ranges(datasets, dataset_names=None, cat_threshold=10):
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

    Parámetros:
    - df: DataFrame de pandas con los datos.
    - true_intervals: diccionario con los intervalos válidos para cada columna.

    Retorna:
    - DataFrame booleano con True donde hay outliers y False donde no.
    """
    outliers = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.columns:
        if col in true_intervals:
            min_val, max_val = true_intervals[col]
            outliers[col] = (df[col] < min_val) | (df[col] > max_val)

    # outliers por columna
    outlier_counts = outliers.sum().reset_index()
    outlier_counts.columns = ['Feature', 'Cantidad de Outliers']

    markdown_table = outlier_counts.to_markdown(index=False)
    display(Markdown("### Cantidad de outliers por feature:\n" + markdown_table))

    return outliers

def class_balance(df, target_column, dataset_name):
    class_counts = df[target_column].value_counts()
    total = len(df)
    
    table_md = f"### Distribución/Desbalanceo de clases en el **{dataset_name}** Set (columna: `{target_column}`)\n\n"
    table_md += "| Clase | Frecuencia | Porcentaje |\n"
    table_md += "|--------|------------|------------|\n"

    for class_value, count in class_counts.items():
        percent = (count / total) * 100
        table_md += f"| {class_value} | {count} | {percent:.2f}% |\n"

    display(Markdown(table_md))