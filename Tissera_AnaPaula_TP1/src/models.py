import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, X, y, normalizar=False):
        """
        Inicializa el modelo de regresión lineal.
        Parámetros:
        - X: Matriz de características (muestras x características)
        - y: Vector de valores objetivo
        - normalizar: Si es True, normaliza las características (media=0, std=1)
        """
        self.normalizar = normalizar
        self.media_X = None
        self.std_X = None

        if self.normalizar:
            self.media_X = np.mean(X, axis=0)
            self.std_X = np.std(X, axis=0)
            X = (X - self.media_X) / (self.std_X + 1e-8)  # Evita divisiones por 0

        self.X = np.hstack((np.ones((X.shape[0], 1)), X))  # Agrega el término de sesgo
        self.y = y
        self.coef = None

    def entrenar_pseudoinversa(self):
        """Entrena el modelo usando la pseudo-inversa con SVD manual."""
        U, S, Vt = np.linalg.svd(self.X, full_matrices=False)

        # Manejo de valores singulares pequeños (evitar división por cero)
        tol = 1e-5  # Umbral para valores singulares pequeños
        S_inv = np.diag([1/s if s > tol else 0 for s in S])

        self.coef = Vt.T @ S_inv @ U.T @ self.y


    def entrenar_descenso_gradiente(self, lr=0.01, epochs=1000):
        """Entrena el modelo con descenso por gradiente."""
        m, n = self.X.shape
        self.coef = np.zeros(n)
        self.historial_perdida = []

        for _ in range(epochs):
            predicciones = self.X @ self.coef
            error = predicciones - self.y
            gradiente = (1 / m) * (self.X.T @ error)
            self.coef -= lr * gradiente
            self.historial_perdida.append(np.mean(error ** 2))

            # Verificación de estabilidad numérica
            if np.isnan(self.coef).any():
                print("Error: Coeficientes nan detectados, reduce la tasa de aprendizaje.")
                return

    def predecir(self, X):
        """Realiza predicciones para nuevas muestras."""
        if self.normalizar:
            X = (X - self.media_X) / (self.std_X + 1e-8)

        if X.shape[1] + 1 == self.X.shape[1]:
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        return X @ self.coef

    def calcular_ecm(self, X, y):
        """Calcula el Error Cuadrático Medio (ECM)."""
        y_pred = self.predecir(X)
        return np.mean((y - y_pred) ** 2)

    def imprimir_coeficientes(self, nombres):
        """Imprime los coeficientes con los nombres de las variables."""
        if self.coef is None:
            print("El modelo no ha sido entrenado aún.")
            return

        for nombre, coef in zip(["intercepto"] + nombres, self.coef):
            print(f"{nombre}: {coef:.4f}")

    def graficar_perdida(self):
        """Grafica la pérdida durante el entrenamiento."""
        if hasattr(self, 'historial_perdida') and self.historial_perdida:
            plt.plot(self.historial_perdida)
            plt.xlabel('Épocas')
            plt.ylabel('Error Cuadrático Medio')
            plt.title('Pérdida durante el entrenamiento')
            plt.show()
        else:
            print("No hay datos de entrenamiento con descenso por gradiente.")

    def graficar_regresion(self, X, y, nombres):
        """Grafica la regresión para una o dos características."""
        if X.shape[1] == 1:  # Una sola característica
            plt.scatter(X, y, color='blue', label='Datos reales')
            X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_pred = self.predecir(X_line)
            plt.plot(X_line, y_pred, color='red', label='Regresión lineal')
            plt.xlabel(nombres[0])
            plt.ylabel('Objetivo')
            plt.legend()
            plt.title('Regresión Lineal - Una característica')
            plt.show()

        elif X.shape[1] == 2:  # Dos características (gráfico 3D)
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[:, 0], X[:, 1], y, color='blue')

            X1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
            X2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
            X1, X2 = np.meshgrid(X1, X2)
            X_grid = np.c_[X1.ravel(), X2.ravel()]
            y_pred = self.predecir(X_grid).reshape(X1.shape)

            ax.plot_surface(X1, X2, y_pred, color='red', alpha=0.6)
            ax.set_xlabel(nombres[0])
            ax.set_ylabel(nombres[1])
            ax.set_zlabel('Objetivo')
            plt.title('Regresión Lineal - Dos características')
            plt.show()

# Función para cargar datos desde un CSV



from preprocessing import one_hot_encoding, softmax
from data_splitting import divide_train_test
from utils import complete_data, normalize_given_μ_σ

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
    std_train = np.std(X_train, axis=0)

    X_test = (X_test - mean_train) / std_train  # Se usa la misma normalización que para train que se normaliza adentro de la función porque se hace un split de validación también

    W, b, mean_train, std_train = multinomial_logistic(X_train, y_train, lr=0.1, epochs=28694)
    
    y_pred = log_predict(X_test, W, b)
    accuracy = precision(y_test, y_pred)
    print(f"Precisión en el conjunto de prueba de train_df: {accuracy:.4f}")
    
    return W, b, mean_train, std_train


def predict_rooms_no_split(df):
    datos = complete_data(df, ['rooms'])

    X = datos[['area']].values
    y = datos['rooms'].values.astype(int)

    y = y - np.min(y)

    W, b, mean_train, std_train = multinomial_logistic(X, y, lr=0.1, epochs=28694)
    
    return W, b, mean_train, std_train




from data_splitting import train_val_test_split
from utils import generate_polynomial_features, add_bias
from metrics import calculate_rmse


def train_regression_for_age(df_train, features, grado=1):
    """Entrena regresión polinómica en train y devuelve parámetros."""
    X_train = generate_polynomial_features(df_train[features].values, grado)
    mean_train = np.mean(X_train, axis=0)
    std_train = np.std(X_train, axis=0) + 1e-8  # para evitar la división por cero
    X_train = normalize_given_μ_σ(X_train, mean_train, std_train)
    X_train = add_bias(X_train)  # Agregar sesgo después de normalizar
    y_train = df_train['age'].values
    
    theta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
    return theta, mean_train, std_train

def reg_predict_age(X, theta, mean_train, std_train, grado=1):
    """Predice valores de 'age' normalizando con las estadísticas de train."""
    X_poly = generate_polynomial_features(X, grado)
    X_poly = normalize_given_μ_σ(X_poly, mean_train, std_train)  # Normalizar antes del sesgo
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
    df_faltantes = df[df['age'].isna()].copy()
    if df_faltantes.empty:
        print("No hay valores faltantes en 'age'.")
        return df
    
    X_faltantes = df_faltantes[features].values
    df.loc[df['age'].isna(), 'age'] = reg_predict_age(X_faltantes, theta, mean_train, std_train, grado)
    
    print(f"{len(df_faltantes)} valores faltantes en 'age' completados.")
    return df

def evaluate_complete_model_age(df, features, grado=1):
    """Entrena el modelo y evalúa su desempeño en train, validación y test."""
    df_train, df_val, df_test = train_val_test_split(df)
    theta, mean_train, std_train = train_regression_for_age(df_train, features, grado)
    
    print(f"Train RMSE: {evaluate_reg_model_age(df_train, theta, mean_train, std_train, features, grado):.4f}")
    print(f"Validation RMSE: {evaluate_reg_model_age(df_val, theta, mean_train, std_train, features, grado):.4f}")
    print(f"Test RMSE: {evaluate_reg_model_age(df_test, theta, mean_train, std_train, features, grado):.4f}")
    
    return theta, mean_train, std_train