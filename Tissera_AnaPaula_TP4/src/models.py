import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters=8, init="k-means++", n_init=10,
                 max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.rng = cp.random.RandomState(random_state)
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def _initialize_centroids(self, X):
        n_samples, n_features = X.shape

        if isinstance(self.init, cp.ndarray):
            centroids = self.init.copy()

        elif self.init == "random":
            # Escoge K muestras aleatorias como centroides iniciales
            idx = self.rng.choice(n_samples, self.n_clusters, replace=False)
            centroids = X[idx]

        elif self.init == "k-means++":
            centroids = cp.empty((self.n_clusters, n_features), dtype=X.dtype)
            # 1) Primer centroide al azar
            idx0 = int(self.rng.randint(n_samples))
            centroids[0] = X[idx0]
            closest_dist_sq = cp.full(n_samples, cp.inf)

            for c in range(1, self.n_clusters):
                # Distancia al centroide más cercano actual
                dist_sq = cp.sum((X - centroids[c-1])**2, axis=1)
                closest_dist_sq = cp.minimum(closest_dist_sq, dist_sq)
                # Probabilidades proporcionales a D(x)^2
                probs = closest_dist_sq / cp.sum(closest_dist_sq)
                cumprobs = cp.cumsum(probs)
                # Generar un escalar en CPU y pasarlo a GPU
                r = self.rng.rand()
                r_gpu = cp.array(r, dtype=X.dtype)
                next_idx = int(cp.searchsorted(cumprobs, r_gpu))
                centroids[c] = X[next_idx]

        else:
            raise ValueError("init debe ser 'k-means++', 'random' o un ndarray")

        return centroids

    def _assign_labels(self, X, centroids):
        dists = cp.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        return cp.argmin(dists, axis=1)

    def _compute_centroids(self, X, labels):
        n_features = X.shape[1]
        centroids = cp.empty((self.n_clusters, n_features), dtype=X.dtype)
        for k in range(self.n_clusters):
            members = X[labels == k]
            if members.shape[0] > 0:
                centroids[k] = cp.mean(members, axis=0)
            else:
                centroids[k] = X[int(self.rng.randint(X.shape[0]))]
        return centroids

    def _inertia(self, X, centroids, labels):
        diffs = X - centroids[labels]
        return cp.sum(diffs**2)

    def fit(self, X):
        best_inertia = cp.inf
        for _ in range(self.n_init):
            centroids = self._initialize_centroids(X)
            prev_inertia = None
            for i in range(1, self.max_iter + 1):
                labels = self._assign_labels(X, centroids)
                centroids = self._compute_centroids(X, labels)
                inertia = self._inertia(X, centroids, labels)
                if prev_inertia is not None and abs(prev_inertia - inertia) <= self.tol * prev_inertia:
                    break
                prev_inertia = inertia
            if inertia < best_inertia:
                best_inertia, best_centroids, best_labels, best_n_iter = (
                    inertia, centroids.copy(), labels.copy(), i
                )
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)
        self.n_iter_ = best_n_iter
        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Debes ajustar el modelo con fit() antes de predecir.")
        return self._assign_labels(X, self.cluster_centers_)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

# ---------------------------------------------------
# Código principal: calcular L vs K, detectar codo
# ---------------------------------------------------

def find_elbow(Ks, Ls):
    """
    Detecta el codo en la curva L vs K.
    Ks, Ls: listas o arrays de igual longitud.
    Devuelve (elbow_idx, distances_np).
    """
    # Convertir a CuPy arrays
    Ks_cp = cp.array(Ks, dtype=cp.float32)
    Ls_cp = cp.array(Ls, dtype=cp.float32)

    # Puntos extremos
    pt1 = cp.array([Ks_cp[0], Ls_cp[0]])
    pt2 = cp.array([Ks_cp[-1], Ls_cp[-1]])

    # Construir matriz de puntos [(K,L), ...]
    pts = cp.stack([Ks_cp, Ls_cp], axis=1)  # shape (n,2)

    # Cálculo vectorial de distancias punto–línea
    # área del paralelogramo: |(pt2-pt1) x (pt-pt1)|, dividido entre longitud de base
    vec = pt2 - pt1  # base
    num = cp.abs(cp.cross(vec, pts - pt1))
    den = cp.linalg.norm(vec)
    distances_cp = num / den  # shape (n,)

    elbow_idx = int(cp.argmax(distances_cp))
    distances_np = cp.asnumpy(distances_cp)
    return elbow_idx, distances_np
