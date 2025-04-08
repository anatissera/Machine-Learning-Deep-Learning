import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, regularization_strength=0.0, multiclass_strategy='binary', reweight_cost=False, plot_loss=False):
        self.lr = learning_rate
        self.epochs = iterations
        self.l2_penalty = regularization_strength
        self.strategy = multiclass_strategy
        self.use_class_weights = reweight_cost
        self.plot_loss = plot_loss
        self.weights = None
        self.bias = None
        self.labels = None
        self.loss_history = []

    def _initialize_parameters(self, n_features, n_classes=None):
        if self.strategy == 'binary':
            self.weights = np.zeros(n_features)
            self.bias = 0.0
        else:
            self.weights = np.zeros((n_features, n_classes))
            self.bias = np.zeros((1, n_classes))

    def _compute_class_weights(self, y):
        # Reponderación de clases para datos desbalanceados
        samples_per_class = np.bincount(y.astype(int))
        total_samples = len(y)
        weights = np.ones_like(y, dtype=np.float64)
        for label in np.unique(y):
            weights[y == label] = total_samples / (len(samples_per_class) * samples_per_class[int(label)])
        return weights

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _softmax(self, logits):
        logits -= np.max(logits, axis=1, keepdims=True)  # Estabilidad numérica
        exp_scores = np.exp(logits)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def _one_hot_encode(self, y):
        one_hot = np.zeros((len(y), len(self.labels)))
        for idx, label in enumerate(y):
            one_hot[idx, int(label)] = 1
        return one_hot

    def _binary_loss(self, y_true, y_pred, weights):
        # Función de pérdida binaria con regularización
        eps = 1e-15  # Para evitar log(0)
        loss = -np.mean(weights * (y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)))
        reg = (self.l2_penalty / (2 * len(y_true))) * np.sum(self.weights ** 2)
        return loss + reg

    def _multiclass_loss(self, y_true, y_pred):
        eps = 1e-15
        loss = -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))
        reg = (self.l2_penalty / (2 * len(y_true))) * np.sum(self.weights ** 2)
        return loss + reg

    def _update_parameters_binary(self, X, y, weights):
        m = X.shape[0]
        for _ in range(self.epochs):
            predictions = self._sigmoid(np.dot(X, self.weights) + self.bias)
            errors = weights * (predictions - y)

            grad_w = (X.T @ errors) / m + (self.l2_penalty / m) * self.weights
            grad_b = np.sum(errors) / m

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            if self.plot_loss:
                loss = self._binary_loss(y, predictions, weights)
                self.loss_history.append(loss)

    def _update_parameters_multiclass(self, X, y_encoded):
        m = X.shape[0]
        for _ in range(self.epochs):
            scores = np.dot(X, self.weights) + self.bias
            probs = self._softmax(scores)
            error = probs - y_encoded

            grad_w = (X.T @ error) / m + (self.l2_penalty / m) * self.weights
            grad_b = np.sum(error, axis=0, keepdims=True) / m

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            if self.plot_loss:
                loss = self._multiclass_loss(y_encoded, probs)
                self.loss_history.append(loss)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.labels = np.unique(y)
        n_samples, n_features = X.shape

        if self.strategy == 'binary':
            self._initialize_parameters(n_features)
            sample_weights = self._compute_class_weights(y) if self.use_class_weights else np.ones_like(y)
            self._update_parameters_binary(X, y, sample_weights)

        elif self.strategy == 'multinomial':
            n_classes = len(self.labels)
            self._initialize_parameters(n_features, n_classes)
            y_one_hot = self._one_hot_encode(y)
            self._update_parameters_multiclass(X, y_one_hot)

        else:
            raise ValueError("La estrategia debe ser 'binary' o 'multinomial'.")

        if self.plot_loss:
            self._plot_training_loss()

    def _plot_training_loss(self):
        # Graficar la evolución de la función de pérdida
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history, label='Loss')
        plt.title("Evolución de la pérdida durante el entrenamiento")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.strategy == 'binary':
            return self._sigmoid(np.dot(X, self.weights) + self.bias)
        else:
            return self._softmax(np.dot(X, self.weights) + self.bias)

    def predict(self, X):
        probs = self.predict_proba(X)
        if self.strategy == 'binary':
            return (probs >= 0.5).astype(int)
        else:
            return np.argmax(probs, axis=1)
