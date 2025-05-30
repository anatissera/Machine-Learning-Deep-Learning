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




import numpy as np
from sklearn.neighbors import KDTree
from collections import deque
from typing import Optional

class DBSCAN:
    """
    DBSCAN con KD‑Tree para acelerar la búsqueda de vecinos.
    """
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """
        Args:
            eps (float): radio de vecindad
            min_samples (int): mínimo de puntos para región densa
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: Optional[np.ndarray] = None
        self.n_clusters_: Optional[int] = None

    def fit(self, X: np.ndarray):
        """
        Ajusta el modelo DBSCAN a X.
        
        Args:
            X (np.ndarray): array de forma (n_samples, n_features)
        """
        n_samples = X.shape[0]
        self.labels_ = -np.ones(n_samples, dtype=int)  # -1 = ruido
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0
        
        # Construyo KDTree una sola vez
        tree = KDTree(X)

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True

            # vecinos en un solo query
            neighbors = tree.query_radius(X[i:i+1], r=self.eps)[0]
            if neighbors.size < self.min_samples:
                # ruido
                self.labels_[i] = -1
            else:
                # expandir nuevo cluster
                self._expand_cluster(i, neighbors, cluster_id, visited, tree)
                cluster_id += 1

        self.n_clusters_ = cluster_id
        return self

    def _expand_cluster(self, 
                        idx: int, 
                        neighbors: np.ndarray, 
                        cluster_id: int, 
                        visited: np.ndarray,
                        tree: KDTree):
        """
        Dada una semilla idx y sus vecinos, expande el cluster.
        """
        self.labels_[idx] = cluster_id
        queue = deque(neighbors.tolist())

        while queue:
            j = queue.popleft()
            if not visited[j]:
                visited[j] = True
                # Sólo un query más al árbol
                neigh_j = tree.query_radius(tree.data[j:j+1], r=self.eps)[0]
                if neigh_j.size >= self.min_samples:
                    queue.extend(neigh_j.tolist())
            # Si era ruido, lo asigno ahora al cluster
            if self.labels_[j] == -1:
                self.labels_[j] = cluster_id

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Ajusta y devuelve labels en un solo paso.
        """
        return self.fit(X).labels_

# from collections import deque

# class DBSCAN:
#     def __init__(self, eps=0.5, min_samples=5):
#         self.eps = eps
#         self.min_samples = min_samples
#         self.labels_ = None
#         self.n_clusters_ = None

#     def fit(self, X):
#         N = X.shape[0]
#         labels = -cp.ones(N, dtype=cp.int32)
#         visited = cp.zeros(N, dtype=cp.bool_)
#         cluster_id = 0

#         # Matriz completa de distancias
#         dists = cp.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

#         for i in range(N):
#             if visited[i]:
#                 continue
#             visited[i] = True

#             # Vecinos de i
#             neigh = cp.where(dists[i] <= self.eps)[0]
#             if neigh.size < self.min_samples:
#                 labels[i] = -1
#                 continue

#             # Nuevo cluster
#             labels[i] = cluster_id
#             queue = deque(neigh.tolist())
#             while queue:
#                 j = queue.popleft()
#                 if not visited[j]:
#                     visited[j] = True
#                     neigh_j = cp.where(dists[j] <= self.eps)[0]
#                     if neigh_j.size >= self.min_samples:
#                         for nb in neigh_j.tolist():
#                             if not visited[nb]:
#                                 queue.append(nb)
#                 if labels[j] == -1:
#                     labels[j] = cluster_id

#             cluster_id += 1

#         self.labels_ = labels
#         self.n_clusters_ = int(cluster_id)
#         return self

#     def fit_predict(self, X):
#         return self.fit(X).labels_
    
    
    
    
# 2

# class PCA:
#     def __init__(self, n_components):
#         self.n_components = n_components
#         self.mean_ = None          # media de cada variable (en GPU)
#         self.components_ = None    # vectores propios seleccionados (en GPU)
#         self.S_ = None             # valores singulares completos (en GPU)

#     def fit(self, X):
#         """
#         Ajusta el modelo PCA sobre X (cp.ndarray, n_samples×n_features).
#         Calcula SVD de la matriz centrada y guarda:
#           - self.mean_
#           - self.S_ (todos los valores singulares)
#           - self.components_ (primeras n_components filas de Vt)
#         """
#         # 1) media de cada característica
#         self.mean_ = cp.mean(X, axis=0)
#         X_centered = X - self.mean_

#         # 2) SVD completo: X_centered = U @ diag(S) @ Vt
#         U, S, Vt = cp.linalg.svd(X_centered, full_matrices=False)
#         self.S_ = S
#         # 3) quedarnos sólo con los n_components primeros vectores propios
#         self.components_ = Vt[:self.n_components]
#         return self

#     def transform(self, X):
#         """
#         Proyecta X al espacio reducido (n_samples×n_components).
#         """
#         X_centered = X - self.mean_
#         return X_centered.dot(self.components_.T)

#     def inverse_transform(self, X_proj):
#         """
#         Reconstruye una aproximación en el espacio original.
#         """
#         return X_proj.dot(self.components_) + self.mean_
    
#     def cummulative_variance(self):
#         """
#         Calcula la varianza acumulada explicada por las componentes principales.
#         Devuelve un array con la varianza acumulada para cada componente.
#         """
#         n_samples = self.S_.shape[0]
#         eigvals = (self.S_ ** 2) / (n_samples - 1)
#         total_var = cp.sum(eigvals)
#         explained_var = eigvals / total_var
#         cum_var = cp.cumsum(explained_var)
        
#         return cum_var

#     def plot_explained_variance(self):
#         """
#         Genera dos figuras:
#          1) Barras con la varianza explicada individual por cada componente.
#          2) Línea con la varianza acumulada.
#         """
#         # número de muestras (igual a len(S_))
#         n = self.S_.shape[0]
#         # eigenvalores λ_i = S_i^2 / (n_samples - 1)
#         eigvals = (self.S_ ** 2) / (n - 1)
#         total_var = cp.sum(eigvals)
#         explained_var = eigvals / total_var
#         cum_var = cp.cumsum(explained_var)

#         # pasar a CPU
#         ev = explained_var.get()
#         cv = cum_var.get()
#         idx = range(1, len(ev) + 1)

#         # figura 1: varianza individual
#         plt.figure(figsize=(10,6))
#         plt.bar(idx, ev, alpha=0.7, color = "teal")
#         plt.xlabel('Componente principal', fontsize=15)
#         plt.ylabel('Varianza explicada individual', fontsize=15)
#         plt.title('Varianza explicada por componente', fontsize=18)
#         plt.xticks(fontsize=13)
#         plt.yticks(fontsize=13)
#         plt.xlim(1, len(ev))
#         plt.tight_layout()
        
#         # figura 1.b: varianza individual zoomed
#         plt.figure(figsize=(8,6))
#         plt.bar(idx, ev, alpha=0.7, color = "teal")
#         plt.xlabel('Componente principal', fontsize=15)
#         plt.ylabel('Varianza explicada individual', fontsize=15)
#         plt.title('Varianza explicada hasta el componente 100', fontsize=18)
#         plt.xticks(fontsize=13)
#         plt.yticks(fontsize=13)
#         plt.xlim(1, 100)
#         plt.tight_layout()


#         # figura 2: varianza acumulada
#         plt.figure(figsize=(10,6))
#         plt.plot(idx, cv, marker='o', linewidth=1, markersize=1, color = "crimson")
#         plt.xlabel('Componente principal', fontsize=15)
#         plt.ylabel('Varianza explicada acumulada', fontsize=15)
#         plt.title('Curva de varianza explicada acumulada', fontsize=18)
#         plt.xticks(fontsize=13)
#         plt.yticks(fontsize=13)
#         plt.ylim(0, 1.03)
#         plt.xlim(1, len(cv))
#         plt.grid(True, linestyle='--', alpha=0.5)
#         plt.tight_layout()
#         plt.show()
        
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_       = None
        self.components_ = None
        self.S_          = None

    def fit(self, X):
        self.mean_ = cp.mean(X, axis=0)
        Xc = X - self.mean_
        U, S, Vt = cp.linalg.svd(Xc, full_matrices=False)
        self.S_ = S
        self.components_ = Vt[:self.n_components]
        return self

    def transform(self, X):
        Xc = X - self.mean_
        return Xc.dot(self.components_.T)

    def inverse_transform(self, X_proj, 
                          clamp: bool = True,
                          binarize: bool = False,
                          threshold: float = 128.0):
        """
        Reconstruye la aproximación en el espacio original.

        Parámetros:
        - X_proj: proyecciones (n_samples×n_components)
        - clamp: si True, limita valores a [0,255]
        - binarize: si True, convierte a 0 o 255 (eliminando grises)
        - threshold: umbral para binarizar (default 128)
        """
        # reconstrucción sin modificaciones
        X_rec = X_proj.dot(self.components_) + self.mean_
        
        if clamp:
            # aseguramos que no salgan valores <0 o >255
            X_rec = cp.clip(X_rec, 0, 255)
        
        if binarize:
            # todo <= threshold → 0; > threshold → 255
            X_rec = cp.where(X_rec > threshold, 255.0, 0.0)

        return X_rec

    def cummulative_variance(self):
        n = self.S_.shape[0]
        eigvals = (self.S_**2) / (n - 1)
        explained = eigvals / cp.sum(eigvals)
        return cp.cumsum(explained)

    def plot_explained_variance(self):
        n = self.S_.shape[0]
        eigvals = (self.S_**2) / (n - 1)
        explained = eigvals / cp.sum(eigvals)
        cum_var   = cp.cumsum(explained)
        ev = explained.get(); cv = cum_var.get()
        idx = range(1, len(ev)+1)

        # Varianza individual
        plt.figure(figsize=(10,6))
        plt.bar(idx, ev, alpha=0.7, color='teal')
        plt.xlabel('Componente principal'); plt.ylabel('Varianza individual')
        plt.title('Varianza explicada por componente'); plt.xlim(1, len(ev))
        plt.tight_layout()

        # Zoom hasta comp. 100
        plt.figure(figsize=(8,6))
        plt.bar(idx, ev, alpha=0.7, color='teal')
        plt.xlabel('Componente principal'); plt.ylabel('Varianza individual')
        plt.title('Varianza explicada (componentes 1–100)'); plt.xlim(1, 100)
        plt.tight_layout()

        # Varianza acumulada
        plt.figure(figsize=(10,6))
        plt.plot(idx, cv, marker='o', linewidth=1, color='crimson')
        plt.xlabel('Componente principal'); plt.ylabel('Varianza acumulada')
        plt.title('Curva de varianza acumulada'); plt.ylim(0,1.03); plt.xlim(1, len(cv))
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        # Encoder
        self.fc1       = nn.Linear(input_dim,  hidden_dim)
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,  input_dim)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.relu(self.fc2(z))
        return self.sigmoid(self.fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        x_rec      = self.decode(z)
        return x_rec, mu, logvar

    @staticmethod
    def loss_function(x, x_rec, mu, logvar):
        """Reconstrucción + KL divergence."""
        recon_loss = nn.functional.binary_cross_entropy(x_rec, x, reduction='sum')
        kl_loss    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss

    def fit(self, train_loader: DataLoader, 
                  val_loader: DataLoader, 
                  n_epochs: int = 20, 
                  lr: float = 1e-3, 
                  device=None):
        """
        Entrena el VAE usando los DataLoaders de entrenamiento y validación.
        Devuelve listas con el historial de pérdidas.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        train_losses = []
        val_losses   = []

        for epoch in range(1, n_epochs + 1):
            # ----- Entrenamiento -----
            self.train()
            running_train_loss = 0.0
            n_train_samples    = 0
            for xb, _ in train_loader:
                xb = xb.to(device).float()
                optimizer.zero_grad()
                x_rec, mu, logvar = self.forward(xb)
                loss = VAE.loss_function(xb, x_rec, mu, logvar)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
                n_train_samples    += xb.size(0)
            epoch_train_loss = running_train_loss / n_train_samples
            train_losses.append(epoch_train_loss)

            # ----- Validación -----
            self.eval()
            running_val_loss = 0.0
            n_val_samples    = 0
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(device).float()
                    x_rec, mu, logvar = self.forward(xb)
                    loss = VAE.loss_function(xb, x_rec, mu, logvar)
                    running_val_loss += loss.item()
                    n_val_samples    += xb.size(0)
            epoch_val_loss = running_val_loss / n_val_samples
            val_losses.append(epoch_val_loss)

            print(f"Epoch {epoch:02d} — Train loss: {epoch_train_loss:.4f}, Val loss: {epoch_val_loss:.4f}")

        return train_losses, val_losses

    def plot_training_curves(self, train_losses, val_losses):
        """Grafica la evolución de la loss de entrenamiento y validación."""
        plt.figure(figsize=(7,4))
        plt.plot(train_losses, label='Train loss')
        plt.plot(val_losses,   label='Val loss')
        plt.xlabel('Época')
        plt.ylabel('Loss promedio')
        plt.title('Curvas de entrenamiento VAE')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
