import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
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

    # def _one_hot_encode(self, y):
    #     one_hot = np.zeros((len(y), len(self.labels)))
    #     for idx, label in enumerate(y):
    #         one_hot[idx, int(label)] = 1
    #     return one_hot
    
    def _one_hot_encode(self, y):
        self.classes_ = np.unique(y)
        y_index = np.array([np.where(self.classes_ == c)[0][0] for c in y])
        one_hot = np.zeros((len(y), len(self.classes_)))
        one_hot[np.arange(len(y)), y_index] = 1
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
    
    def fit(self, X, y, sample_weights=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.labels = np.unique(y)
        n_samples, n_features = X.shape

        if self.strategy == 'binary':
            self._initialize_parameters(n_features)
            # Si se especifican sample_weights, usarlos. Si no, calcularlos o usar 1s.
            if sample_weights is not None:
                weights = sample_weights
            elif self.use_class_weights:
                weights = self._compute_class_weights(y)
            else:
                weights = np.ones_like(y)
            self._update_parameters_binary(X, y, weights)

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
            return self._sigmoid(np.dot(X, self.weights) + self.bias)
        else:
            return self._softmax(np.dot(X, self.weights) + self.bias)

    def predict(self, X):
        probs = self.predict_proba(X)
        if self.strategy == 'binary':
            return (probs >= 0.5).astype(int)
        else:
            # return np.argmax(probs, axis=1)
            return self.labels[np.argmax(probs, axis=1)]


class LDA:
    def __init__(self):
        # Diccionarios para guardar medias y probabilidades a priori por clase
        self.means_ = {}
        self.priors_ = {}
        self.shared_covariance_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        # Inicializamos la matriz de covarianza acumulada
        pooled_cov = np.zeros((n_features, n_features))

        for label in self.classes_:
            # Extraemos las muestras que pertenecen a la clase actual
            class_samples = X[y == label]
            n_class_samples = class_samples.shape[0]

            # Calculamos la media por clase y la almacenamos
            self.means_[label] = np.mean(class_samples, axis=0)

            # Calculamos la probabilidad a priori de la clase
            self.priors_[label] = n_class_samples / n_samples

            # Sumamos la covarianza (ponderada) de la clase al total
            class_cov = np.cov(class_samples, rowvar=False)
            pooled_cov += (n_class_samples - 1) * class_cov

        # Covarianza compartida entre todas las clases (con corrección por grados de libertad)
        self.shared_covariance_ = pooled_cov / (n_samples - len(self.classes_))

    def _compute_score(self, X, mean, prior, cov_inv):
        # Calcula el valor de la función discriminante para una clase
        linear_term = X @ cov_inv @ mean
        constant_term = -0.5 * mean.T @ cov_inv @ mean + np.log(prior)
        return linear_term + constant_term

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        inv_cov = np.linalg.inv(self.shared_covariance_)

        # Evaluamos la función discriminante para cada clase
        scores = np.array([
            self._compute_score(X, self.means_[label], self.priors_[label], inv_cov)
            for label in self.classes_
        ])

        # Elegimos la clase con el mayor score para cada muestra
        return self.classes_[np.argmax(scores, axis=0)]

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        inv_cov = np.linalg.inv(self.shared_covariance_)

        # Calculamos los scores discriminantes para cada clase
        raw_scores = np.array([
            self._compute_score(X, self.means_[label], self.priors_[label], inv_cov)
            for label in self.classes_
        ])

        # Aplicamos softmax columna por columna para obtener probabilidades
        stabilized = raw_scores - np.max(raw_scores, axis=0)
        exp_scores = np.exp(stabilized)
        probs = exp_scores / np.sum(exp_scores, axis=0)
        return probs.T  # Cada fila es una muestra, cada columna una clase


class DecisionTree:
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

            # Saltar si la columna no es numérica
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

    def _predict_single(self, x, node):
        if node['is_leaf']:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])


    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.root = self.build(X, y, depth=0)

    def predict_sample(self, x, node):
        if node['is_leaf']:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self.predict_sample(x, node['left'])
        else:
            return self.predict_sample(x, node['right'])

    def predict(self, X):
        return np.array([self.predict_sample(x, self.root) for x in X])


class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.trees = []
        self.classes_ = None

    def _bootstrap_sample(self, X, y):
        n = len(X)
        idxs = np.random.choice(n, size=n, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        self.trees = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
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
