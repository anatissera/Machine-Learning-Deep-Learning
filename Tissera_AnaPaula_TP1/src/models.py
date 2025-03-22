import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, X, y):
        self.X = np.hstack((np.ones((X.shape[0], 1)), X))  # Agrega columna de unos (término de sesgo)
        self.y = y
        self.coef = None

    def entrenar_pseudoinversa(self):
        """Entrena el modelo usando la pseudo-inversa."""
        self.coef = np.linalg.pinv(self.X) @ self.y

    def entrenar_descenso_gradiente(self, lr=0.01, epochs=1000):
        """Entrena el modelo usando descenso por gradiente."""
        m, n = self.X.shape
        self.coef = np.zeros(n)
        self.historial_perdida = []

        for _ in range(epochs):
            predicciones = self.X @ self.coef
            error = predicciones - self.y
            gradiente = (1 / m) * (self.X.T @ error)
            self.coef -= lr * gradiente
            self.historial_perdida.append(np.mean(error ** 2))

    def predecir(self, X):
        """Realiza predicciones para nuevas muestras."""
        if X.shape[1] + 1 == self.X.shape[1]:  # Verifica si falta agregar la columna de unos
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.coef

    def calcular_ecm(self, X, y):
        """Calcula el Error Cuadrático Medio (ECM)."""
        y_pred = self.predecir(X)
        return np.mean((y - y_pred) ** 2)

    def imprimir_coeficientes(self, nombres):
        """Imprime los coeficientes con los nombres de las variables."""
        for nombre, coef in zip(["intercepto"] + nombres, self.coef):
            print(f"{nombre}: {coef:.4f}")

    def graficar_perdida(self):
        """Grafica la pérdida durante el entrenamiento."""
        if hasattr(self, 'historial_perdida'):
            plt.plot(self.historial_perdida)
            plt.xlabel('Épocas')
            plt.ylabel('Error Cuadrático Medio')
            plt.title('Pérdida durante el entrenamiento')
            plt.show()

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

# Función auxiliar para cargar los datos
def cargar_datos(ruta, features, target):
    import pandas as pd
    df = pd.read_csv(ruta)
    X = df[features].values
    y = df[target].values
    return X, y




def multinomial_logistic(X, y, lr=0.1, epochs=30000, patience=500, val_size=0.1):
    """Entrena un modelo de regresión logística multinomial con normalización dentro de la función."""
    
    num_samples, num_features = X.shape
    num_classes = np.unique(y).size

    # Dividir en train y validación
    val_split = int(num_samples * (1 - val_size))
    X_train, X_val = X[:val_split], X[val_split:]
    y_train, y_val = y[:val_split], y[val_split:]

    # Calcular estadísticas de normalización SOLO con X_train
    mean_train = np.mean(X_train, axis=0)
    std_train = np.std(X_train, axis=0)

    # Normalizar X_train y X_val con los valores de X_train
    X_train = (X_train - mean_train) / (std_train + 1e-8)  # +1e-8 para evitar división por cero
    X_val = (X_val - mean_train) / (std_train + 1e-8)

    # One-hot encoding para y
    y_train_one_hot = one_hot_encoding(y_train, num_classes)
    y_val_one_hot = one_hot_encoding(y_val, num_classes)

    # Inicializar pesos y sesgos
    W = np.random.randn(num_features, num_classes) * 0.01
    b = np.zeros((1, num_classes))

    best_loss = np.inf
    best_W, best_b = W.copy(), b.copy()
    wait = 0  # Contador para early stopping

    for epoch in range(epochs):
        # Calcular logits y aplicar softmax
        logits = np.dot(X_train, W) + b
        probs = softmax(logits)

        # Calcular la pérdida (entropía cruzada)
        loss_train = -np.mean(y_train_one_hot * np.log(probs + 1e-8))

        # Validación
        logits_val = np.dot(X_val, W) + b
        probs_val = softmax(logits_val)
        loss_val = -np.mean(y_val_one_hot * np.log(probs_val + 1e-8))

        # Gradientes
        grad_W = np.dot(X_train.T, (probs - y_train_one_hot)) / X_train.shape[0]
        grad_b = np.mean(probs - y_train_one_hot, axis=0, keepdims=True)

        # Actualizar parámetros
        W -= lr * grad_W
        b -= lr * grad_b

        # Early stopping: guardar los mejores pesos si la pérdida en validación mejora
        if loss_val < best_loss:
            best_loss = loss_val
            best_W, best_b = W.copy(), b.copy()
            wait = 0  # Reiniciar paciencia
        else:
            wait += 1  # Aumentar paciencia

        # Detener si no mejora después de muchas épocas
        if wait >= patience:
            print(f"Early stopping en época {epoch}, mejor pérdida de validación: {best_loss:.4f}")
            break

        # Mostrar progreso cada 5000 épocas
        if epoch % 5000 == 0:
            print(f"Época {epoch}, Pérdida Train: {loss_train:.4f}, Pérdida Val: {loss_val:.4f}")

    return best_W, best_b, mean_train, std_train  # Devuelve los mejores pesos y los valores de normalización


def normalize_given_μ_σ(X_new, mean_train, std_train):
    """Normaliza nuevos datos usando los valores de X_train."""
    return (X_new - mean_train) / (std_train + 1e-8)

def log_predict(X, W, b):
    """Realiza predicciones con el modelo entrenado."""
    logits = np.dot(X, W) + b
    probs = softmax(logits)
    return np.argmax(probs, axis=1)  # Clase con mayor probabilidad

def precision(y_real, y_pred):
    """Calcula la precisión del modelo."""
    return np.mean(y_real == y_pred)

from preprocessing import one_hot_encoding, softmax
from data_splitting import complete_data, divide_train_test

def predict_rooms_train_test(df):
    
    datos = complete_data(df, ['rooms'])

    X = datos[['area']].values
    y = datos['rooms'].values.astype(int)

    y = y - np.min(y)

    X_train, X_test, y_train, y_test = divide_train_test(X, y)

    mean_train = np.mean(X_train, axis=0)
    std_train = np.std(X_train, axis=0)

    # X_train = (X_train - mean_train) / std_train  # Normalización dentro de la función
    X_test = (X_test - mean_train) / std_train  # Se usa la misma normalización

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


