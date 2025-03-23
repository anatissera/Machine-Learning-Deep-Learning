import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.media_X = None
        self.std_X = None

        self.X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        self.y_train = y_train
        self.X_val = np.hstack((np.ones((X_val.shape[0], 1)), X_val))
        self.y_val = y_val
        self.coef = None
        self.historial_perdida_train = []
        self.historial_perdida_val = []

    def entrenar_pseudoinversa(self):
        U, S, Vt = np.linalg.svd(self.X_train, full_matrices=False)
        tol = 1e-5  
        S_inv = np.diag([1/s if s > tol else 0 for s in S])
        self.coef = Vt.T @ S_inv @ U.T @ self.y_train

    def entrenar_descenso_gradiente(self, lr=0.01, epochs=1000, early_stopping=True, tol=1e-5, paciencia=10):
        m, n = self.X_train.shape
        self.coef = np.zeros(n)
        self.historial_perdida_train = []
        self.historial_perdida_val = []
        mejor_perdida = float('inf')
        paciencia_contador = 0

        for epoch in range(epochs):
            predicciones_train = self.X_train @ self.coef
            error_train = predicciones_train - self.y_train
            gradiente = (1 / m) * (self.X_train.T @ error_train)
            self.coef -= lr * gradiente
            
            perdida_train = np.mean(error_train ** 2)
            perdida_val = np.mean((self.X_val @ self.coef - self.y_val) ** 2)
            
            self.historial_perdida_train.append(perdida_train)
            self.historial_perdida_val.append(perdida_val)
            
            if early_stopping:
                if perdida_val < mejor_perdida - tol:
                    mejor_perdida = perdida_val
                    paciencia_contador = 0
                else:
                    paciencia_contador += 1
                    if paciencia_contador >= paciencia:
                        print(f"Early stopping en epoch {epoch+1}")
                        break

            if np.isnan(self.coef).any():
                print("Error: Coeficientes nan detectados, reducir la tasa de aprendizaje.")
                return

    def predecir(self, X):
        if X.shape[1] + 1 == self.X_train.shape[1]:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.coef

    def evaluar(self, X_test, y_test):
        predicciones = self.predecir(X_test)
        mse = np.mean((predicciones - y_test) ** 2)
        print(f"Error cuadrático medio (RMSE) en test: {mse:.4f}")
        return mse

    def graficar_regresion_pseudoinversa(self, X, y, nombres):
        if X.shape[1] == 1:
            plt.scatter(X, y, color='blue', label='real data')
            X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_pred = self.predecir(X_line)
            plt.plot(X_line, y_pred, color='green', label='Pseudo-inverse')
            plt.xlabel(nombres[0])
            plt.ylabel('Target price')
            plt.legend()
            plt.title('Linear Regression - Pseudo-inverse')
            plt.show()

    def graficar_regresion_descenso_gradiente(self, X, y, nombres):
        if X.shape[1] == 1:
            plt.scatter(X, y, color='blue', label='Real data')
            X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_pred = self.predecir(X_line)
            plt.plot(X_line, y_pred, color='red', label=' Gradient Descent')
            plt.xlabel(nombres[0])
            plt.ylabel('Target price')
            plt.legend()
            plt.title('Linear Regression - Gradient Descent')
            plt.show()

    def graficar_perdida(self):
        if self.historial_perdida_train and self.historial_perdida_val:
            plt.plot(self.historial_perdida_train, label='Train')
            plt.plot(self.historial_perdida_val, label='Validation')
            plt.xlabel('Épocas')
            plt.ylabel('Error Cuadrático Medio (RMSE)')
            plt.title('Pérdida durante el entrenamiento')
            plt.legend()
            plt.show()
        else:
            print("No hay datos de entrenamiento con descenso por gradiente.")


from src.preprocessing import one_hot_encoding, softmax
from src.data_splitting import divide_train_test
from src.utils import complete_data, normalize_given_μ_σ

def multinomial_logistic(X, y, lr=0.1, epochs=30000, patience=500, val_size=0.1):
    """Entrena un modelo de regresión logística multinomial con normalización dentro de la función."""
    
    num_samples, num_features = X.shape
    num_classes = np.unique(y).size
    
    val_split = int(num_samples * (1 - val_size))
    X_train, X_val = X[:val_split], X[val_split:]
    y_train, y_val = y[:val_split], y[val_split:]

    mean_train = np.mean(X_train, axis=0)
    std_train = np.std(X_train, axis=0)

    X_train = (X_train - mean_train) / (std_train + 1e-8)  # +1e-8 para evitar división por cero
    X_val = (X_val - mean_train) / (std_train + 1e-8) # con los estadísticos de train

    y_train_one_hot = one_hot_encoding(y_train, num_classes)
    y_val_one_hot = one_hot_encoding(y_val, num_classes)

    # Inicializar pesos y sesgos
    W = np.random.randn(num_features, num_classes) * 0.01
    b = np.zeros((1, num_classes))

    best_loss = np.inf
    best_W, best_b = W.copy(), b.copy()
    wait = 0 
    
    for epoch in range(epochs):
        logits = np.dot(X_train, W) + b
        probs = softmax(logits)

        # pérdida -> cross-entropy
        loss_train = -np.mean(y_train_one_hot * np.log(probs + 1e-8))

        logits_val = np.dot(X_val, W) + b
        probs_val = softmax(logits_val)
        loss_val = -np.mean(y_val_one_hot * np.log(probs_val + 1e-8))
        
        grad_W = np.dot(X_train.T, (probs - y_train_one_hot)) / X_train.shape[0]
        grad_b = np.mean(probs - y_train_one_hot, axis=0, keepdims=True)

        W -= lr * grad_W
        b -= lr * grad_b

        # Early stopping
        if loss_val < best_loss:
            best_loss = loss_val
            best_W, best_b = W.copy(), b.copy()
            wait = 0 
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping en época {epoch}, mejor pérdida de validación: {best_loss:.4f}")
            break

        if epoch % 5000 == 0:
            print(f"Época {epoch}, Pérdida Train: {loss_train:.4f}, Pérdida Val: {loss_val:.4f}")

    return best_W, best_b, mean_train, std_train 


# con área debería hacer min max scaling, no estandarización, tengo que arreglar eso

def log_predict(X, W, b):
    """Realiza predicciones con el modelo entrenado."""
    logits = np.dot(X, W) + b
    probs = softmax(logits)
    return np.argmax(probs, axis=1)

def precision(y_real, y_pred):
    """Calcula la precisión del modelo."""
    return np.mean(y_real == y_pred)


def complete_missing_rooms_values(df, W, b, mean_train, std_train):
    """Completa los valores faltantes en la columna 'rooms' usando el modelo entrenado."""

    df_faltantes = df[df['rooms'].isna()].copy()
    X_faltantes = df_faltantes[['area']].values
    X_faltantes = (X_faltantes - mean_train) / std_train # usando las estadísticas de train

    y_pred_faltantes = log_predict(X_faltantes, W, b)
    df.loc[df['rooms'].isna(), 'rooms'] = y_pred_faltantes
    print(f"{len(df_faltantes)} valores faltantes en 'rooms' completados.")
    
    return df


def predict_rooms_train_test(df):
    datos = complete_data(df, ['rooms'])

    X = datos[['area']].values
    y = datos['rooms'].values.astype(int)

    y = y - np.min(y)

    X_train, X_test, y_train, y_test = divide_train_test(X, y)

    mean_train = np.mean(X_train, axis=0)
    print("mean_train", mean_train)
    std_train = np.std(X_train, axis=0)

    X_test = (X_test - mean_train) / std_train  # Se usa la misma normalización que para train que se normaliza adentro de la función porque se hace un split de validación también

    W, b, mean_train, std_train = multinomial_logistic(X_train, y_train, lr=0.1, epochs=30000)
    
    y_pred = log_predict(X_test, W, b)
    accuracy = precision(y_test, y_pred)
    print(f"Precisión en el conjunto de prueba de train_df: {accuracy:.4f}")
    
    return W, b, mean_train, std_train


def predict_rooms_no_split(df):
    datos = complete_data(df, ['rooms'])

    X = datos[['area']].values
    y = datos['rooms'].values.astype(int)

    y = y - np.min(y)

    W, b, mean_train, std_train = multinomial_logistic(X, y, lr=0.1, epochs=30000)
    
    return W, b, mean_train, std_train


from src.data_splitting import train_val_test_split
from src.utils import generate_polynomial_features, add_bias
from src.metrics import calculate_rmse

# area y price hay que min max scaling, no estandarización, tengo que arreglar eso

def train_regression_for_age(df_train, features, grado=1):
    """Entrena regresión polinómica en train y devuelve parámetros."""
    X_train = generate_polynomial_features(df_train[features].values, grado)
    mean_train = np.mean(X_train, axis=0)
    std_train = np.std(X_train, axis=0) + 1e-8  # para evitar división por cero
    X_train = normalize_given_μ_σ(X_train, mean_train, std_train)
    X_train = add_bias(X_train)
    y_train = df_train['age'].values
    
    theta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
    return theta, mean_train, std_train

def reg_predict_age(X, theta, mean_train, std_train, grado=1):
    """Predice valores de 'age' normalizando con las estadísticas de train."""
    X_poly = generate_polynomial_features(X, grado)
    X_poly = normalize_given_μ_σ(X_poly, mean_train, std_train)
    X_poly = add_bias(X_poly)
    return X_poly @ theta

def evaluate_reg_model_age(df, theta, mean_train, std_train, features, grado):
    """Evalúa el modelo calculando el RMSE en un dataset dado."""
    X = generate_polynomial_features(df[features].values, grado)
    X = normalize_given_μ_σ(X, mean_train, std_train)
    X = add_bias(X)
    y_pred = X @ theta
    return calculate_rmse(df['age'].values, y_pred)

def complete_missing_age_values(df, theta, mean_train, std_train, features, grado=1):
    """Completa valores faltantes en 'age' usando el modelo entrenado."""
    mask_missing = df['age'].isna()
    if mask_missing.sum() == 0:
        print("No hay valores faltantes en 'age'.")
        return df
    
    X_faltantes = df.loc[mask_missing, features].values
    df.loc[mask_missing, 'age'] = reg_predict_age(X_faltantes, theta, mean_train, std_train, grado)
    print(f"{mask_missing.sum()} valores faltantes en 'age' completados.")
    return df

def evaluate_and_impute(df, features, grado=1):
    """Entrena el modelo, lo evalúa y luego imputa valores faltantes."""
    df_train, df_val, df_test = train_val_test_split(df)
    theta, mean_train, std_train = train_regression_for_age(df_train, features, grado)
    
    print(f"Train de Train_df RMSE: {evaluate_reg_model_age(df_train, theta, mean_train, std_train, features, grado):.4f}")
    print(f"Validation de Train_df RMSE: {evaluate_reg_model_age(df_val, theta, mean_train, std_train, features, grado):.4f}")
    print(f"Test de Train_df RMSE: {evaluate_reg_model_age(df_test, theta, mean_train, std_train, features, grado):.4f}")
    
    # Entrenar con todos los datos completos antes de imputar
    theta_full, mean_full, std_full = train_regression_for_age(df.dropna(subset=['age']), features, grado)
    df = complete_missing_age_values(df, theta_full, mean_full, std_full, features, grado)
    return df
