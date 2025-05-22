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
        self.distortion_ = None
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
        self.distortion_ = float(best_inertia)
        self.n_iter_ = best_n_iter
        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Debes ajustar el modelo con fit() antes de predecir.")
        return self._assign_labels(X, self.cluster_centers_)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_







class GMM:
    def __init__(self, n_components=3, tol=1e-4, max_iter=100,
                 reg_covar=1e-6, random_state=None, init_params=None):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.rng = cp.random.RandomState(random_state)
        self.init_params = init_params
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.log_likelihood_ = None
        self.n_iter_ = None

    def _initialize(self, X):
        N, D = X.shape
        if self.init_params is not None:
            self.means_       = self.init_params['means']
            self.weights_     = self.init_params['weights']
            self.covariances_ = self.init_params['covs']
            return

        # si no hay init_params, inicializar con KMeans
        km = KMeans(
            n_clusters=self.n_components,
            init="k-means++",
            n_init=10,
            max_iter=300,
            tol=1e-4,
            random_state=int(self.rng.randint(1e6))
        )
        labels = km.fit_predict(X)
        self.means_ = km.cluster_centers_.astype(cp.float64)
        counts = cp.bincount(labels, minlength=self.n_components).astype(cp.float64)
        self.weights_ = counts / N

        covs = cp.zeros((self.n_components, D, D), dtype=cp.float64)
        
        for k in range(self.n_components):
            Xk = X[labels == k].astype(cp.float64)
            if Xk.shape[0] > 1:
                diff = Xk - self.means_[k]
                covs[k] = (diff.T @ diff) / Xk.shape[0]
            else:
                covs[k] = cp.eye(D, dtype=cp.float64)
        for k in range(self.n_components):
            covs[k] += cp.eye(D, dtype=cp.float64) * self.reg_covar
        self.covariances_ = covs

    def _estimate_log_gaussian_prob(self, X):
        X = X.astype(cp.float64)
        N, D = X.shape
        K = self.n_components
        log_prob = cp.zeros((N, K), dtype=cp.float64)
        for k in range(K):
            mu = self.means_[k]
            cov = self.covariances_[k]
            chol = cp.linalg.cholesky(cov)
            log_det = 2 * cp.sum(cp.log(cp.diag(chol)))
            diff = X - mu
            sol = cp.linalg.solve(chol, diff.T)
            maha = cp.sum(sol**2, axis=0)
            log_prob[:, k] = -0.5 * (D * cp.log(2 * cp.pi) + log_det + maha)
        return log_prob

    def fit(self, X):
        X = X.astype(cp.float64)
        N, D = X.shape
        self._initialize(X)
        prev_ll = None

        for i in range(1, self.max_iter + 1):
            # E-step sin logsumexp
            log_prob = self._estimate_log_gaussian_prob(X)
            log_resp = log_prob + cp.log(self.weights_)
            row_max = cp.max(log_resp, axis=1)
            log_norm = row_max + cp.log(cp.sum(cp.exp(log_resp - row_max[:, None]), axis=1))
            resp = cp.exp(log_resp - log_norm[:, None])
            ll = cp.sum(log_norm)

            # M-step
            Nk = resp.sum(axis=0)
            self.weights_ = Nk / N
            self.means_ = (resp.T @ X) / Nk[:, None]

            covs = cp.zeros((self.n_components, D, D), dtype=cp.float64)
            for k in range(self.n_components):
                diff = X - self.means_[k]
                covs[k] = (resp[:, k][:, None] * diff).T @ diff / Nk[k]
                covs[k] += cp.eye(D, dtype=cp.float64) * self.reg_covar
            self.covariances_ = covs

            if prev_ll is not None and abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self.log_likelihood_ = float(ll)
        self.n_iter_ = i
        return self

    def predict(self, X):
        log_prob = self._estimate_log_gaussian_prob(X)
        return cp.argmax(log_prob + cp.log(self.weights_), axis=1)

    def fit_predict(self, X):
        return self.fit(X).predict(X)





from collections import deque

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.n_clusters_ = None

    def fit(self, X):
        N = X.shape[0]
        labels = -cp.ones(N, dtype=cp.int32)
        visited = cp.zeros(N, dtype=cp.bool_)
        cluster_id = 0

        # Matriz completa de distancias
        dists = cp.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

        for i in range(N):
            if visited[i]:
                continue
            visited[i] = True

            # Vecinos de i
            neigh = cp.where(dists[i] <= self.eps)[0]
            if neigh.size < self.min_samples:
                labels[i] = -1
                continue

            # Nuevo cluster
            labels[i] = cluster_id
            queue = deque(neigh.tolist())
            while queue:
                j = queue.popleft()
                if not visited[j]:
                    visited[j] = True
                    neigh_j = cp.where(dists[j] <= self.eps)[0]
                    if neigh_j.size >= self.min_samples:
                        for nb in neigh_j.tolist():
                            if not visited[nb]:
                                queue.append(nb)
                if labels[j] == -1:
                    labels[j] = cluster_id

            cluster_id += 1

        self.labels_ = labels
        self.n_clusters_ = int(cluster_id)
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_