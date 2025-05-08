import cupy as cp

def split_data(X, y, ratio=0.2, seed=42):
    assert 0 < ratio < 1, "val_ratio debe estar entre 0 y 1"

    # Mezclar los datos
    cp.random.seed(seed)
    indices = cp.arange(len(X))
    cp.random.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Partición
    N = len(X)
    val_size   = int(ratio * N)
    train_size = N - val_size

    X_train, y_train = X_shuffled[:train_size], y_shuffled[:train_size]
    X_val, y_val     = X_shuffled[train_size:], y_shuffled[train_size:]

    return X_train, y_train, X_val, y_val
