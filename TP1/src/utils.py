import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def missing_values(df):
    print("\nValores faltantes por columna:")
    print(df.isna().sum())
    
def missing_values_in_column(df, column):
    return df[column].isna().sum()
    
def missing_percentages(df):
    total_filas = len(df)
    
    n_nan_age = df["age"].isna().sum()
    porcentaje_nan_age = (n_nan_age / total_filas) * 100
    
    n_nan_rooms = df["rooms"].isna().sum()
    porcentaje_nan_rooms = (n_nan_rooms / total_filas) * 100

    print(f"Valores NaN en 'age': {n_nan_age} sobre un total de {total_filas} filas ({porcentaje_nan_age:.2f}%)")
    print(f"Valores NaN en 'rooms': {n_nan_rooms} sobre un total de {total_filas} filas ({porcentaje_nan_rooms:.2f}%)")

def save_csv(df, nombre_archivo):
    df.to_csv(nombre_archivo, index=False)
    print(f"Dataset guardado como '{nombre_archivo}'.")

def complete_data(df, to_drop):
    return df.dropna(subset=to_drop)

def normalize_given_μ_σ(X, mean, std):
    return (X - mean) / std

def add_bias(X):
    return np.c_[np.ones(X.shape[0]), X]

def generate_polynomial_features(X, grado=1):
    return np.hstack([X ** g for g in range(1, grado + 1)])

def load_data(path, df, features, target, is_df=False):
    if not is_df:
        df = pd.read_csv(path)
    X = df[features].values
    y = df[target].values
    return X, y

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def hist_plot(df, column):
    plt.figure(figsize=(4, 2))
    sns.histplot(df[f"{column}"], bins=50, kde=True, color = "mediumaquamarine")
    plt.title(f"`{column}`'s distribution")
    plt.show()
    
def select_features(relevant_features, features, X_train, X_val, X_test):
    feature_indices = {feature: idx for idx, feature in enumerate(features)}
    selected_indices = [feature_indices[f] for f in relevant_features]

    X_train_subset = X_train[:, selected_indices]
    X_val_subset = X_val[:, selected_indices]
    X_test_subset = X_test[:, selected_indices]
    
    return X_train_subset, X_val_subset, X_test_subset

def normalize_dataset(dataset):
    return (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

def pca_with_svd(X, d):
    A = normalize_dataset(X)
    
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    VT_d = Vt[:d, :]
    V_d = V[:, :d]
    
    Z = np.dot(U_d, S_d)
    
    return Z, U_d, S_d, VT_d
