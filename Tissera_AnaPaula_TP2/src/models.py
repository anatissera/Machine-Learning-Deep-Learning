import numpy as np
import matplotlib.pyplot as plt

EPSILON = 1e-15

# Problema 1 y 

class LogisticRegression:
    """
    Logistic Regression classifier.

    Esta clase implementa un modelo de regresión logística con soporte para
    clasificación binaria y multiclase. Permite la regularización L2, el ajuste
    de pesos de clase y la visualización de la pérdida durante el entrenamiento.

    Atributos:
        lr: float
            Tasa de aprendizaje utilizada para la optimización.
        epochs: int
            Número de iteraciones para el entrenamiento.
        l2_penalty: float
            Coeficiente de regularización L2.
        strategy: str
            Estrategia de clasificación ('binary' o 'multinomial').
        use_class_weights: bool
            Indica si se deben ajustar los pesos de las clases automáticamente.
        plot_loss: bool
            Indica si se debe graficar la pérdida durante el entrenamiento.
        weights: array
            Pesos del modelo ajustados durante el entrenamiento.
        bias: float o array
            Sesgo del modelo ajustado durante el entrenamiento.
        labels: array
            Etiquetas únicas de las clases.
        loss_history: list
            Historial de la pérdida durante el entrenamiento.
    """
    
    def __init__(self, learning_rate=0.01, iterations=1000, l2=0.0, multiclass_strategy='binary', reweight_cost=False, plot_loss=False):
        self.lr = learning_rate
        self.epochs = iterations
        self.l2_penalty = l2
        self.strategy = multiclass_strategy
        self.use_class_weights = reweight_cost
        self.plot_loss = plot_loss
        self.weights = None
        self.bias = None
        self.labels = None
        self.loss_history = []

    def initialize_parameters(self, n_features, n_classes=None):
        if self.strategy == 'binary':
            self.weights = np.zeros(n_features)
            self.bias = 0.0
        else:
            self.weights = np.zeros((n_features, n_classes))
            self.bias = np.zeros((1, n_classes))

    def compute_class_weights(self, y):
        samples_per_class = np.bincount(y.astype(int))
        total_samples = len(y)
        weights = np.ones_like(y, dtype=np.float64)
        for label in np.unique(y):
            weights[y == label] = total_samples / (len(samples_per_class) * samples_per_class[int(label)])
        return weights

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, logits):
        logits -= np.max(logits, axis=1, keepdims=True)  # para estabilidad numérica
        exp_scores = np.exp(logits)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def one_hot_encode(self, y):
        self.class_map = dict(zip(self.labels, np.arange(len(self.labels))))
        
        indices = np.vectorize(self.class_map.get)(y)
        encoded = np.zeros((len(y), len(self.labels)))
        encoded[np.arange(len(y)), indices] = 1

        return encoded

    def binary_loss(self, y_true, y_pred, weights): # con regularización
        loss = -np.mean(weights * (y_true * np.log(y_pred + EPSILON) + (1 - y_true) * np.log(1 - y_pred + EPSILON)))
        reg = (self.l2_penalty / (2 * len(y_true))) * np.sum(self.weights ** 2)
        return loss + reg
    
    def update_parameters_binary(self, X, y, weights):
        m = X.shape[0]
        for _ in range(self.epochs):
            predictions = self.sigmoid(np.dot(X, self.weights) + self.bias)
            errors = weights * (predictions - y)

            grad_w = (X.T @ errors) / m + (self.l2_penalty / m) * self.weights
            grad_b = np.sum(errors) / m

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            if self.plot_loss:
                loss = self.binary_loss(y, predictions, weights)
                self.loss_history.append(loss)

    def multiclass_loss(self, y_true, y_pred):
        loss = -np.mean(np.sum(y_true * np.log(y_pred + EPSILON), axis=1))
        reg = (self.l2_penalty / (2 * len(y_true))) * np.sum(self.weights ** 2)
        return loss + reg

    def update_parameters_multiclass(self, X, y_encoded):
        m = X.shape[0]
        for _ in range(self.epochs):
            scores = np.dot(X, self.weights) + self.bias
            probs = self.softmax(scores)
            error = probs - y_encoded

            grad_w = (X.T @ error) / m + (self.l2_penalty / m) * self.weights
            grad_b = np.sum(error, axis=0, keepdims=True) / m

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            if self.plot_loss:
                loss = self.multiclass_loss(y_encoded, probs)
                self.loss_history.append(loss)
    
    def fit(self, X, y, sample_weights=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.labels = np.unique(y)
        n_samples, n_features = X.shape

        if self.strategy == 'binary':
            self.initialize_parameters(n_features)
            if sample_weights is not None:
                weights = sample_weights
            elif self.use_class_weights:
                weights = self.compute_class_weights(y)
            else:
                weights = np.ones_like(y)
            self.update_parameters_binary(X, y, weights)

        elif self.strategy == 'multinomial':
            n_classes = len(self.labels)
            self.initialize_parameters(n_features, n_classes)
            y_one_hot = self.one_hot_encode(y)
            self.update_parameters_multiclass(X, y_one_hot)
        else:
            raise ValueError("La estrategia debe ser 'binary' o 'multinomial'.")

        if self.plot_loss:
            self.plot_training_loss()

    def plot_training_loss(self):
        plt.figure(figsize=(6, 4))
        plt.plot(self.loss_history, label='Loss', c="cadetblue")
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
            return self.sigmoid(np.dot(X, self.weights) + self.bias)
        else:
            return self.softmax(np.dot(X, self.weights) + self.bias)

    def predict(self, X):
        probs = self.predict_proba(X)
        if self.strategy == 'binary':
            return (probs >= 0.5).astype(int)
        else:
            return self.labels[np.argmax(probs, axis=1)]


# Problema 2

class LDA:
    """
    Linear Discriminant Analysis (LDA) classifier.

    Esta clase implementa el modelo generativo de LDA asumiendo que las
    clases comparten la misma matriz de covarianza. Se estiman los parámetros
    mediante la máxima verosimilitud bajo la suposición de que cada clase sigue
    una distribución gaussiana multivariada.

    Atributos:
        classes: array, shape (n_classes,)
            Etiquetas únicas de las clases.
        means: dict
            Diccionario que asigna a cada clase su vector de medias.
        priors: dict
            Diccionario que asigna a cada clase su probabilidad a priori.
        covariance: array, shape (n_features, n_features)
            Matriz de covarianza agrupada calculada a partir de los datos de entrenamiento.
        covariance_inv: array, shape (n_features, n_features)
            Inversa de la matriz de covarianza, almacenada para su uso en la función discriminante.
    """

    def __init__(self):
        self.classes = None
        self.means = {}
        self.priors = {}
        self.covariance = None
        self.covariance_inv = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes = np.unique(y)
        n_features = X.shape[1]
        total_samples = X.shape[0]

        self.covariance = np.zeros((n_features, n_features))

        for cls in self.classes:
            X_c = X[y == cls]
            self.means[cls] = np.mean(X_c, axis=0)
            self.priors[cls] = float(X_c.shape[0]) / total_samples
            # np.cov calcula la matriz de covarianza dividiendo por (n-1) -> multiplicamos por (n-1) para obtener la suma de cuadrados.
            self.covariance += (X_c.shape[0] - 1) * np.cov(X_c, rowvar=False)

        self.covariance /= (total_samples - len(self.classes))

        self.covariance_inv = np.linalg.inv(self.covariance)
        return self

    def _discriminant_function(self, X, mean, prior):
        # fórmula: X @ Σ⁻¹ @ mean - 0.5 * meanᵀ @ Σ⁻¹ @ mean + log(prior)
        return X @ self.covariance_inv @ mean - 0.5 * mean.T @ self.covariance_inv @ mean + np.log(prior)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        scores = np.array([
            self._discriminant_function(X, self.means[cls], self.priors[cls])
            for cls in self.classes
        ])
        max_indices = np.argmax(scores, axis=0)
        return self.classes[max_indices]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        scores = np.array([
            self._discriminant_function(X, self.means[cls], self.priors[cls])
            for cls in self.classes
        ])
        # para estabilidad numérica, se sustrae el máximo puntaje por muestra antes de exponenciar
        max_score = np.max(scores, axis=0)
        exp_scores = np.exp(scores - max_score)
        probabilities = exp_scores / np.sum(exp_scores, axis=0)
        return probabilities.T

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)



class DecisionTree:
    """
    Árbol de Decisión para clasificación.
    
    Esta clase implementa un Árbol de Decisión basado en la entropía como criterio
    de división. Permite ajustar el modelo a un conjunto de datos y realizar predicciones
    sobre nuevos datos.
    
    Atributos:
        max_depth: int, opcional (default=None)
            Profundidad máxima del árbol. Si es None, el árbol crecerá hasta que
            todas las hojas sean puras o hasta que el número mínimo de muestras
            por nodo sea alcanzado.
        min_samples: int, opcional (default=2)
            Número mínimo de muestras requerido para dividir un nodo.
        root: dict
            Nodo raíz del árbol de decisión, que contiene la estructura del árbol
            construido tras el ajuste.
    """
    
    def __init__(self, max_depth=None, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None

    def entropy(self, y):
        labels, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-10))

    def best_split(self, X, y):
        n_samples, n_features = X.shape
        base_entropy = self.entropy(y)

        best_gain = -np.inf
        best_feature = None
        best_thresh = None

        for feat_idx in range(n_features):
            column = X[:, feat_idx]

            # si la columna no es numérica -> saltar
            if not np.issubdtype(column.dtype, np.number):
                continue

            thresholds = np.unique(column)
            for t in thresholds:
                left = y[column <= t]
                right = y[column > t]

                if len(left) == 0 or len(right) == 0:
                    continue

                p_left = len(left) / len(y)
                p_right = 1 - p_left
                gain = base_entropy - (p_left * self.entropy(left) + p_right * self.entropy(right))

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_thresh = t

        return best_feature, best_thresh

    def build(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or depth == self.max_depth or len(y) < self.min_samples:
            values, counts = np.unique(y, return_counts=True)
            majority = values[np.argmax(counts)]
            return {'is_leaf': True, 'class': majority}

        feat, thresh = self.best_split(X, y)
        if feat is None:
            values, counts = np.unique(y, return_counts=True)
            majority = values[np.argmax(counts)]
            return {'is_leaf': True, 'class': majority}

        left_idx = X[:, feat] <= thresh
        right_idx = ~left_idx

        return {
            'is_leaf': False,
            'feature': feat,
            'threshold': thresh,
            'left': self.build(X[left_idx], y[left_idx], depth + 1),
            'right': self.build(X[right_idx], y[right_idx], depth + 1)
        }

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.root = self.build(X, y)

    def predict_single(self, x, node):
        if node['is_leaf']:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self.predict_single(x, node['left'])
        else:
            return self.predict_single(x, node['right'])

    def predict(self, X):
        return np.array([self.predict_single(x, self.root) for x in X])

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.root = self.build(X, y, depth=0)

    def predict_proba(self, x, node):
        if node['is_leaf']:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self.predict_proba(x, node['left'])
        else:
            return self.predict_proba(x, node['right'])

    def predict(self, X):
        return np.array([self.predict_proba(x, self.root) for x in X])


class RandomForest:
    """
    Random Forest classifier.
    
    Esta clase implementa un modelo de Random Forest, que combina múltiples
    árboles de decisión entrenados en subconjuntos aleatorios de los datos
    para realizar clasificación. Utiliza el método de muestreo bootstrap
    para generar subconjuntos de datos y votación mayoritaria para predecir
    las etiquetas finales.
    
    Atributos:
        n_estimators: int
            Número de árboles en el bosque.
        max_depth: int o None
            Profundidad máxima permitida para cada árbol. Si es None, los árboles
            crecerán hasta que todas las hojas sean puras o contengan menos de
            min_samples.
        min_samples: int
            Número mínimo de muestras requeridas para dividir un nodo.
        trees: list
            Lista de árboles de decisión individuales que componen el bosque.
        classes_: array, shape (n_classes,)
            Etiquetas únicas de las clases presentes en los datos de entrenamiento.
    """
    
    def __init__(self, n_trees=10, max_depth=None, min_samples=2):
        self.n_estimators = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.trees = []
        self.classes_ = None

    def bootstrap_sample(self, X, y):
        n = len(X)
        idxs = np.random.choice(n, size=n, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        self.trees = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth, min_samples=self.min_samples)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        X = np.array(X)
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        votes = []

        for i in range(X.shape[0]):
            row = all_preds[:, i]
            values, counts = np.unique(row, return_counts=True)
            votes.append(values[np.argmax(counts)])

        return np.array(votes)

    def predict_proba(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        class_indices = {cls: idx for idx, cls in enumerate(self.classes_)}

        proba = np.zeros((n_samples, n_classes))

        for tree in self.trees:
            preds = tree.predict(X)
            for i in range(n_samples):
                class_idx = class_indices[preds[i]]
                proba[i, class_idx] += 1

        proba /= self.n_estimators
        return proba
