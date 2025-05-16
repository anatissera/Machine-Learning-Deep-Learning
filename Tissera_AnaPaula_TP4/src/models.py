import numpy as np

class KMeans:
    """
    Una implementación de K-Means similar a la de sklearn.
    Parámetros
    ----------
    n_clusters : int
        El número de clusters a formar.
    init : {'k-means++', 'random'} or ndarray of shape (n_clusters, n_features)
        Método de inicialización de los centroides.
    n_init : int
        Número de veces que el algoritmo se ejecutará con diferentes centroides
        iniciales. Se guarda la mejor solución en términos de inercia.
    max_iter : int
        Máximo número de iteraciones por ejecución única.
    tol : float
        Tolerancia relativa al cambio de inercia para declarar convergencia.
    random_state : int, RandomState instance or None
        Semilla para el generador de números aleatorios.
    """

    def __init__(self, n_clusters=8, init="k-means++", n_init=10, max_iter=300,
                 tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = np.random.RandomState(random_state)

        # Atributos que se rellenarán al hacer fit
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def _initialize_centroids(self, X):
        n_samples, n_features = X.shape

        if isinstance(self.init, np.ndarray):
            # Inicialización por array dado
            centroids = self.init.copy()
        elif self.init == "random":
            # Escoge K muestras aleatorias
            indices = self.random_state.choice(n_samples, self.n_clusters, replace=False)
            centroids = X[indices]
        elif self.init == "k-means++":
            # Implementación de k-means++
            centroids = np.empty((self.n_clusters, n_features), dtype=X.dtype)
            # 1) Escoge el primer centroide al azar
            idx = self.random_state.randint(n_samples)
            centroids[0] = X[idx]
            # 2) Para cada centroide restante
            closest_dist_sq = np.full(n_samples, np.inf)
            for c in range(1, self.n_clusters):
                # Distancia al centroid más cercano actual
                dist_sq = np.sum((X - centroids[c-1])**2, axis=1)
                closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)
                # Probabilidad proporcional a D(x)^2
                probs = closest_dist_sq / closest_dist_sq.sum()
                cumulative_probs = np.cumsum(probs)
                r = self.random_state.rand()
                next_idx = np.searchsorted(cumulative_probs, r)
                centroids[c] = X[next_idx]
        else:
            raise ValueError(f"init debe ser 'k-means++', 'random' o un ndarray, no {self.init}")

        return centroids

    def _assign_labels(self, X, centroids):
        # Distancias euclídeas y asignación al centroide más cercano
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        return np.argmin(dists, axis=1)

    def _compute_centroids(self, X, labels):
        # Recalcula centroides como medias de puntos asignados
        n_features = X.shape[1]
        centroids = np.empty((self.n_clusters, n_features), dtype=X.dtype)
        for k in range(self.n_clusters):
            members = X[labels == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)
            else:
                # Si un cluster queda vacío, reubica el centroide aleatoriamente
                centroids[k] = X[self.random_state.randint(X.shape[0])]
        return centroids

    def _inertia(self, X, centroids, labels):
        # Suma de distancias al cuadrado (inercia)
        return np.sum((X - centroids[labels]) ** 2)

    def fit(self, X):
        """
        Computa K-Means clustering sobre X.
        Guarda los centroides en cluster_centers_, las etiquetas en labels_,
        la inercia mínima en inertia_ y el número de iteraciones
        de la ejecución final en n_iter_.
        """
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_n_iter = None

        for init_no in range(self.n_init):
            # 1) Inicializar centroides
            centroids = self._initialize_centroids(X)
            for i in range(1, self.max_iter + 1):
                # 2) Asignar etiquetas
                labels = self._assign_labels(X, centroids)
                # 3) Recalcular centroides
                new_centroids = self._compute_centroids(X, labels)
                # 4) Calcular inercia y comprobar convergencia
                inertia = self._inertia(X, new_centroids, labels)
                if init_no == 0 and i == 1:
                    prev_inertia = inertia
                if abs(prev_inertia - inertia) <= self.tol * prev_inertia:
                    break
                centroids = new_centroids
                prev_inertia = inertia

            # Guardar mejor ejecución
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = i

        # Asignar atributos finales
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def predict(self, X):
        """Asigna nuevos puntos X a los centroides ya calculados."""
        if self.cluster_centers_ is None:
            raise ValueError("Este KMeans no está ajustado. Llama primero a fit().")
        return self._assign_labels(X, self.cluster_centers_)

    def fit_predict(self, X):
        """Conveniencia: ajusta y devuelve las etiquetas."""
        self.fit(X)
        return self.labels_
