import numpy as np



def data_splitter(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios deben sumar 1"

    # Mezclar los datos
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Partición
    N = len(X)
    train_end = int(train_ratio * N)
    val_end = train_end + int(val_ratio * N)

    X_train, y_train = X_shuffled[:train_end], y_shuffled[:train_end]
    X_val, y_val = X_shuffled[train_end:val_end], y_shuffled[train_end:val_end]
    X_test, y_test = X_shuffled[val_end:], y_shuffled[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def split_data(X, y, ratio=0.2, seed=42):
    assert 0 < ratio < 1, "val_ratio debe estar entre 0 y 1"

    # Mezclar los datos
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Partición
    N = len(X)
    val_size = int(ratio * N)
    train_size = N - val_size

    X_train, y_train = X_shuffled[:train_size], y_shuffled[:train_size]
    X_val, y_val = X_shuffled[train_size:], y_shuffled[train_size:]

    return X_train, y_train, X_val, y_val