import numpy as np
from tqdm import trange

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True)) # evitar overflow
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def cross_entropy(y_pred, y_true):
    # y_true debe ser one-hot
    eps = 1e-12 # para evitar log(0)
    y_pred = np.clip(y_pred, eps, 1. - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

class NeuralNetwork:
    def __init__(self, hidden_layers, seed=42):
        self.hidden_layer_sizes = hidden_layers
        self.seed = seed
        self.is_initialized = False

    def initialize_parameters(self, input_size, output_size):
        np.random.seed(self.seed)
        layer_sizes = [input_size] + self.hidden_layer_sizes + [output_size]
        self.L = len(layer_sizes) - 1
        self.weights = []
        self.biases = []
        for i in range(self.L):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)
        self.is_initialized = True

    def forward(self, X):
        activations = [X]
        zs = []
        for l in range(self.L - 1):
            z = activations[-1] @ self.weights[l] + self.biases[l]
            zs.append(z)
            a = relu(z)
            activations.append(a)
        # Capa de salida con softmax
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        zs.append(z)
        a = softmax(z)
        activations.append(a)
        return activations, zs

    def backward(self, activations, zs, y_true):
        grads_W = [None] * self.L
        grads_b = [None] * self.L
        delta = activations[-1] - y_true
        grads_W[-1] = activations[-2].T @ delta / y_true.shape[0]
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True) / y_true.shape[0]
        for l in reversed(range(self.L - 1)):
            delta = (delta @ self.weights[l+1].T) * relu_deriv(zs[l])
            grads_W[l] = activations[l].T @ delta / y_true.shape[0]
            grads_b[l] = np.sum(delta, axis=0, keepdims=True) / y_true.shape[0]
        return grads_W, grads_b

    def update_params(self, grads_W, grads_b, lr):
        for l in range(self.L):
            self.weights[l] -= lr * grads_W[l]
            self.biases[l] -= lr * grads_b[l]

    def fit(self, X, y, X_val=None, y_val=None, epochs=100, lr=0.01, verbose=True):
        if not self.is_initialized:
            self.initialize_parameters(X.shape[1], y.shape[1])
        train_losses = []
        val_losses = []

        iterator = trange(epochs, desc="Training", disable=not verbose)
        for epoch in iterator:
            activations, zs = self.forward(X)
            loss = cross_entropy(activations[-1], y)
            train_losses.append(loss)
            grads_W, grads_b = self.backward(activations, zs, y)
            self.update_params(grads_W, grads_b, lr)

            if X_val is not None and y_val is not None:
                val_preds, _ = self.forward(X_val)
                val_loss = cross_entropy(val_preds[-1], y_val)
                val_losses.append(val_loss)
                iterator.set_postfix(loss=loss.item(), val_loss=val_loss.item())
            else:
                iterator.set_postfix(loss=loss.item())

        return train_losses, val_losses

    def predict(self, X):
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    def predict_proba(self, X):
        activations, _ = self.forward(X)
        return activations[-1]
