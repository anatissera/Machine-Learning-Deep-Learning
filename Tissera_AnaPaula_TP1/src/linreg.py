import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionprueba:
    def __init__(self, X_train, y_train, X_val, y_val, feature_names):
        self.feature_names = feature_names
        self.stats = {}
        self._compute_statistics(X_train, y_train)
        
        self.X_train = self._transform(X_train)
        self.y_train = self._scale_target(y_train)
        self.X_val = self._transform(X_val)
        self.y_val = self._scale_target(y_val)
        
        self.coef = None
        self.historial_perdida_train = []
        self.historial_perdida_val = []
    
    def _compute_statistics(self, X, y):
        self.stats = {}
        for i, feature in enumerate(self.feature_names):
            if feature in ["age", "rooms"]:
                self.stats[feature] = {"mean": X[:, i].mean(), "std": X[:, i].std()}
            elif feature == "area":
                self.stats[feature] = {"min": X[:, i].min(), "max": X[:, i].max()}
        self.stats["price"] = {"min": y.min(), "max": y.max()}
    
    def _transform(self, X):
        X_transformed = X.copy()
        for i, feature in enumerate(self.feature_names):
            if feature in ["age", "rooms"]:
                X_transformed[:, i] = (X[:, i] - self.stats[feature]["mean"]) / self.stats[feature]["std"]
            elif feature == "area":
                X_transformed[:, i] = (X[:, i] - self.stats[feature]["min"]) / (self.stats[feature]["max"] - self.stats[feature]["min"])
        return np.hstack((np.ones((X.shape[0], 1)), X_transformed))
    
    def _scale_target(self, y):
        return (y - self.stats["price"]["min"]) / (self.stats["price"]["max"] - self.stats["price"]["min"])
    
    def _inverse_transform_target(self, y_scaled):
        return y_scaled * (self.stats["price"]["max"] - self.stats["price"]["min"]) + self.stats["price"]["min"]
    
    def entrenar_pseudoinversa(self):
        U, S, Vt = np.linalg.svd(self.X_train, full_matrices=False)
        tol = 1e-5  
        S_inv = np.diag([1/s if s > tol else 0 for s in S])
        self.coef = Vt.T @ S_inv @ U.T @ self.y_train
    
    def entrenar_descenso_gradiente(self, lr=0.01, epochs=1000, early_stopping=True, tol=1e-5, paciencia=10):
        m, n = self.X_train.shape
        self.coef = np.zeros(n)
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
    
    def predecir(self, X):
        X_transformed = self._transform(X)
        predicciones_scaled = X_transformed @ self.coef
        return self._inverse_transform_target(predicciones_scaled)
    
    def evaluar(self, X_test, y_test):
        predicciones = self.predecir(X_test)
        mse = np.mean((predicciones - y_test) ** 2)
        print(f"Error cuadrático medio (RMSE) en test: {mse:.4f}")
        return mse
    
    def graficar_regresion_pseudoinversa(self, X, y, nombres):
        if X.shape[1] == 1:
            plt.scatter(X, y, color='blue', label='Real data')
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
            plt.plot(X_line, y_pred, color='red', label='Gradient Descent')
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
