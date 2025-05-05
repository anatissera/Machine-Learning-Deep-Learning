import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, seed=None):
        """
        Inicializa la red neuronal.

        Parámetros:
        - layer_sizes: lista con número de neuronas en cada capa, incluyendo input y output, e.g. [784, 128, 10]
        - learning_rate: tasa de aprendizaje para el gradiente descendiente.
        - seed: semilla para reproducibilidad.
        """
        if seed is not None:
            np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.L = len(layer_sizes) - 1  # número de capas con parámetros
        self._initialize_weights()

    def _initialize_weights(self):
        """Inicializa pesos y biases con He initialization."""
        self.weights = {}
        self.biases = {}
        for l in range(1, len(self.layer_sizes)):
            in_size = self.layer_sizes[l-1]
            out_size = self.layer_sizes[l]
            self.weights[l] = np.random.randn(out_size, in_size) * np.sqrt(2. / in_size)
            self.biases[l] = np.zeros((out_size, 1))

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def _softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def forward(self, X):
        """Propagación hacia adelante."""
        A = X.T
        self.caches = {'A0': A}
        for l in range(1, self.L + 1):
            W, b = self.weights[l], self.biases[l]
            Z = W.dot(A) + b
            A = self._relu(Z) if l < self.L else self._softmax(Z)
            self.caches[f'Z{l}'] = Z
            self.caches[f'A{l}'] = A
        return A

    def compute_loss(self, Y_hat, Y):
        """Cross-entropy loss para clasificación multi-clase."""
        m = Y.shape[0]
        eps = 1e-15
        log_probs = -np.log(Y_hat.T[Y == 1] + eps)
        return np.sum(log_probs) / m

    def backward(self, Y):
        """Propagación hacia atrás para actualizar pesos y biases."""
        m = Y.shape[0]
        Y_hat = self.caches[f'A{self.L}']
        Y_true = Y.T

        dZ = Y_hat - Y_true
        for l in reversed(range(1, self.L + 1)):
            A_prev = self.caches[f'A{l-1}']
            dW = (1/m) * dZ.dot(A_prev.T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            self.weights[l] -= self.learning_rate * dW
            self.biases[l]  -= self.learning_rate * db
            if l > 1:
                Z_prev = self.caches[f'Z{l-1}']
                dA_prev = self.weights[l].T.dot(dZ)
                dZ = dA_prev * self._relu_derivative(Z_prev)

    def predict(self, X):
        """Retorna la etiqueta predicha para cada ejemplo."""
        Y_hat = self.forward(X)
        return np.argmax(Y_hat, axis=0)

    def train(self, X_train, Y_train, X_val=None, Y_val=None, epochs=100, batch_size=None, plot=True):
        """
        Entrena la red neuronal y opcionalmente grafica la evolución del loss.

        Parámetros:
        - X_train: array (m_train, n_features)
        - Y_train: one-hot (m_train, n_classes)
        - X_val: array (m_val, n_features), opcional
        - Y_val: one-hot (m_val, n_classes), opcional
        - epochs: número de épocas
        - batch_size: tamaño de batch; si None, usa batch completo
        - plot: si True, grafica train y val loss

        Retorna:
        - train_losses, val_losses (listas)
        """
        m = X_train.shape[0]
        train_losses, val_losses = [], []
        for epoch in range(1, epochs+1):
            # Forward y backward en train
            Y_hat_tr = self.forward(X_train)
            loss_tr = self.compute_loss(Y_hat_tr, Y_train)
            self.backward(Y_train)

            train_losses.append(loss_tr)

            # Cálculo de loss en validación si se proporcionó
            if X_val is not None and Y_val is not None:
                Y_hat_val = self.forward(X_val)
                loss_val = self.compute_loss(Y_hat_val, Y_val)
                val_losses.append(loss_val)

        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
            if val_losses:
                plt.plot(range(1, epochs+1), val_losses, label="Val Loss")
            plt.xlabel("Época")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("Evolución de la Función de Costo")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return train_losses, val_losses
