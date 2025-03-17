import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class RegresionLineal:
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

            # Almacenar la pérdida
            self.historial_perdida.append(np.mean(error ** 2))

    def predecir(self, X):
        """Realiza predicciones para nuevas muestras."""
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
            ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Datos reales')

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
            plt.legend()
            plt.show()

# Función auxiliar para cargar los datos
def cargar_datos(ruta, features, target):
    df = pd.read_csv(ruta)
    X = df[features].values
    y = df[target].values
    return X, y

# Rutas de los archivos
ruta_train = 'Tissera_AnaPaula_TP1/casas_train.csv'
ruta_test = 'Tissera_AnaPaula_TP1/casas_test_corregido.csv'

# Características y objetivo
features = ['area', 'rooms', 'is_house']
target = 'price'

# Cargar los conjuntos de entrenamiento y prueba
X_train, y_train = cargar_datos(ruta_train, features, target)
X_test, y_test = cargar_datos(ruta_test, features, target)

# Entrenar y evaluar el modelo con pseudo-inversa
modelo_pi = RegresionLineal(X_train, y_train)
modelo_pi.entrenar_pseudoinversa()
print("Coeficientes (pseudo-inversa):")
modelo_pi.imprimir_coeficientes(features)
print("ECM en test (pseudo-inversa):", modelo_pi.calcular_ecm(X_test, y_test))
modelo_pi.graficar_regresion(X_train, y_train, features)

# Entrenar y evaluar el modelo con descenso por gradiente
modelo_gd = RegresionLineal(X_train, y_train)
modelo_gd.entrenar_descenso_gradiente(lr=0.01, epochs=10000)
print("\nCoeficientes (descenso por gradiente):")
modelo_gd.imprimir_coeficientes(features)
print("ECM en test (descenso por gradiente):", modelo_gd.calcular_ecm(X_test, y_test))

# Graficar la pérdida durante el entrenamiento
modelo_gd.graficar_perdida()
modelo_gd.graficar_regresion(X_train, y_train, features)
