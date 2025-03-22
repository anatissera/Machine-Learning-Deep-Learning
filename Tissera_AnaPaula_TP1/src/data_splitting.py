import numpy as np
import pandas as pd

def split_and_save_train_val(df, train_path, val_path, train_ratio=0.8, seed=42):
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(df))
    
    train_size = int(len(df) * train_ratio)
    train_indices, val_indices = shuffled_indices[:train_size], shuffled_indices[train_size:]

    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]

    train_df.to_csv(f"data/{train_path}", index=False)
    val_df.to_csv(f"data/{val_path}", index=False)
    
    print(f"Train set: {train_df.shape[0]} rows, Validation set: {val_df.shape[0]} rows")
    
    return train_df, val_df

def divide_train_test(X, y, test_size=0.2, seed=42):
    """Divide los datos en conjunto de entrenamiento y prueba."""
    np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split = int(X.shape[0] * (1 - test_size))
    
    X_train, X_test = X[indices[:split]], X[indices[split:]]
    y_train, y_test = y[indices[:split]], y[indices[split:]]
    
    return X_train, X_test, y_train, y_test

def train_val_test_split(df, val_size=0.1, test_size=0.2, seed=42):
    """Divide el dataset en train, validación y test, usando solo datos completos."""
    df_completo = df.dropna(subset=['age'])
    np.random.seed(seed)
    indices = np.random.permutation(len(df_completo))
    
    test_split = int(len(df_completo) * (1 - test_size))
    val_split = int(test_split * (1 - val_size))
    
    return df_completo.iloc[indices[:val_split]], df_completo.iloc[indices[val_split:test_split]], df_completo.iloc[indices[test_split:]]

