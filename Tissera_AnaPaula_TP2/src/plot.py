import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_features_with_correlation(df, dataset_name, target_col, features=None, start=0, batch_size=14):
    if features is None:
        features = df.columns.drop(target_col)

    end = min(start + batch_size, len(features))
    selected_features = features[start:end]

    n_rows = (len(selected_features) + 1) // 2
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(14, 4 * n_rows))
    fig.suptitle(f"Distribuciones y correlación con '{target_col}' para el {dataset_name}", fontsize=16, y=0.995)

    best_feature = None
    best_corr = -np.inf

    for i, feature in enumerate(selected_features):
        row = i // 2
        col = i % 2
        ax = axes[row][col] if n_rows > 1 else axes[col]

        if pd.api.types.is_numeric_dtype(df[feature]):
            sns.histplot(df[feature], kde=True, ax=ax, color="cornflowerblue")
        else:
            sns.countplot(x=df[feature], hue=df[feature], ax=ax, palette="Set2", legend=False)
            ax.tick_params(axis='x', rotation=45)

        ax.set_title(f"{feature}", fontsize=12)

        try:   # Calcular correlación
            if pd.api.types.is_numeric_dtype(df[feature]):
                corr = df[[feature, target_col]].corr().iloc[0, 1]
            else:
                temp = pd.get_dummies(df[feature], drop_first=True)
                temp[target_col] = df[target_col]
                corr = temp.corr()[target_col].drop(target_col).abs().max()
        except:
            corr = np.nan

        if not np.isnan(corr) and abs(corr) > abs(best_corr):
            best_corr = corr
            best_feature = feature
            
        ax.text(ax.get_xlim()[1]*0.99, ax.get_ylim()[1]*0.8,
                f"Corr: {corr:.3f}", fontsize=15, ha='right', va='center', bbox=dict(boxstyle="round", facecolor='white', edgecolor='black'))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    if best_feature is not None:
        print(f"📈 Feature más correlacionada en este batch: **{best_feature}** (corr = {best_corr:.3f})")



def plot_correlations_with_target(df, dataset_name, target_col, plot=True, colormap="tab20b"):
    """
    Calcula y muestra (opcionalmente) la correlación entre las variables numéricas
    y una variable objetivo.

    Parámetros:
    - dataframe: DataFrame de entrada (pandas).
    - target: Nombre de la variable objetivo.
    - show_plot: Si es True, se muestra un gráfico tipo heatmap.
    - colormap: Paleta de colores para el gráfico.

    Retorna:
    - Una Serie de pandas con las correlaciones ordenadas por valor absoluto (ascendente).
    """

    numeric_data = df.select_dtypes(include=["float64", "int64"]).copy()

    if target_col not in numeric_data.columns:
        numeric_data[target_col] = df[target_col]

    target_correlations = numeric_data.corr()[target_col].drop(target_col)

    sorted_correlations = target_correlations.reindex(
        target_correlations.abs().sort_values(ascending=True).index
    )

    if plot:
        plt.figure(figsize=(6, max(1.5, 0.4 * len(sorted_correlations))))
        sns.heatmap(
            sorted_correlations.to_frame().T,
            annot=True,
            cmap=colormap,
            center=0,
            cbar_kws={"label": "Correlación"},
            fmt=".2f"
        )
        plt.title(f"Correlación de características del {dataset_name} con la variable objetivo: {target_col}")
        plt.yticks([])
        plt.tight_layout()
        plt.show()
    
    else:
        return sorted_correlations
    
    
# def plot_pairplot(df, target_col):
#     """
#     Genera un gráfico de pares (pair plot) para visualizar las relaciones entre columnas numéricas.

#     Parameters:
#     df (pd.DataFrame): El DataFrame de entrada.
#     target_col (str): El nombre de la columna objetivo para el color del gráfico.

#     Returns:
#     None
#     """

#     numeric_df = df.select_dtypes(include=['number'])
    
#     if numeric_df.shape[1] < 2:
#         print("No hay suficientes columnas numéricas para crear un pair plot.")
#         return

#     # sns.pairplot(numeric_df, diag_kind='kde', hue=target_col)
#     custom_colors = ["palevioletred", "lightseagreen"]  # Rojo y azul
#     sns.pairplot(numeric_df.join(df[target_col]), diag_kind='kde', hue=target_col, palette=custom_colors)

#     plt.show()
    
    
# import seaborn as sns
# import matplotlib.pyplot as plt

def plot_pairplot(df, target_col, palette=["palevioletred", "cadetblue"]):
    """
    Genera un gráfico de pares (pair plot) para visualizar las relaciones entre columnas numéricas,
    excluyendo todas las categóricas excepto la columna objetivo, que se usa para colorear.

    Parameters:
    df (pd.DataFrame): El DataFrame de entrada.
    target_col (str): El nombre de la columna objetivo para el color del gráfico.
    palette (list or dict, optional): Lista o diccionario de colores personalizados.

    Returns:
    None
    """

    numeric_df = df.select_dtypes(include=['number'])

    if numeric_df.shape[1] < 2:
        print("No hay suficientes columnas numéricas para crear un pair plot.")
        return

    sns.pairplot(numeric_df, diag_kind='kde', hue=target_col, palette=palette)
    plt.show()
