import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgb
import cupy as cp
from models import PCA

# 1

def plot_lvsk(Ks, Ls, bestK, bestL):
    plt.figure(figsize=(8, 5.5))
    plt.plot(Ks, Ls, marker='o', label='Inercia (L) vs K', color = "lightseagreen")
    plt.scatter([bestK], [bestL], color='red', s=100, label=f'Elbow en K={bestK}')
    plt.xlabel("Número de clusters K", fontsize=15)
    plt.ylabel("Inercia (L)", fontsize=15)
    plt.title("Método del codo: L vs K", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(Ks)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.show()
    
    
def plot_kmeans(km, X, labels):
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(20)]  

    fig, ax = plt.subplots(figsize=(11,5.5))
    for k in range(km.n_clusters):
        pts_k = X[labels == k]
        x_k = pts_k[:, 0].get()
        y_k = pts_k[:, 1].get()
        ax.scatter(x_k, y_k, s=30, color=colors[k % 20], label=f"Cluster {k+1}", alpha=0.5)

    centers = km.cluster_centers_
    cx = centers[:, 0].get()
    cy = centers[:, 1].get()
    ax.scatter(cx, cy, marker='X', s=100, c='black', label="Centroides")

    ax.set_xlabel("X1", fontsize=16)
    ax.set_ylabel("X2", fontsize=16)
    ax.set_title(f"KMeans con k={km.n_clusters}", fontsize=18)

    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)


    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=12.5)

    plt.tight_layout(rect=(0,0,0.75,1))
    plt.show()
    
    
def plot_GMM(gmm, X, labels, bestK):
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(20)]  

    fig, ax = plt.subplots(figsize=(11,5))

    # Scatter de puntos
    for k in range(bestK):
        pts = X[labels == k]
        x_cpu = pts[:, 0].get()
        y_cpu = pts[:, 1].get()
        ax.scatter(x_cpu, y_cpu, s=30, color=colors[k], alpha=0.5, zorder=1)

    # Elipses y medias
    for k in range(bestK):
        mean_cpu = gmm.means_[k].get()
        cov_cpu  = gmm.covariances_[k].get()
        if cov_cpu.ndim == 1:
            cov_cpu = np.diag(cov_cpu)

        vals, vecs = np.linalg.eigh(cov_cpu)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
        
        width  = 2 * np.sqrt(vals[0])
        height = 2 * np.sqrt(vals[1])

        dark_color = tuple(np.array(to_rgb(colors[k])) * 0.75)

        ellipse = Ellipse(
            xy=mean_cpu,
            width=width, height=height,
            angle=angle,
            edgecolor=dark_color,
            facecolor='none',
            linewidth=1.7,
            linestyle='--',
            zorder=2
        )
        
        ax.add_patch(ellipse)

        ax.scatter(
            mean_cpu[0], mean_cpu[1],
            marker='X', s=100,
            color=colors[k], edgecolor='black', facecolor='black',
            linewidth=0.5,
            zorder=3
        )

    ax.set_xlabel("X1", fontsize=15)
    ax.set_ylabel("X2", fontsize=15)
    ax.set_title(f"GMM con K={bestK}", fontsize=18)
    ax.legend([f"Comp{k+1}" for k in range(bestK)],
            loc="center left", bbox_to_anchor=(1.02,0.5), fontsize=12)

    plt.tight_layout(rect=(0,0,0.75,1))
    plt.show()
    
    
def plot_DBSCAN(X, labels, min_samples, eps, ):
    fig, ax = plt.subplots(figsize=(11,5))
    cmap = plt.get_cmap("tab20")
    for lab in sorted(set(labels.tolist())):
        mask = (labels == lab)
        if lab == -1:
            ax.scatter(X[mask,0], X[mask,1],
                    c='k', marker='x', label='Ruido', alpha=0.6)
        else:
            ax.scatter(X[mask,0], X[mask,1],
                    c=[cmap(lab%20)], s=30, label=f"Cluster {lab}", alpha=0.5)

    ax.set_title(f"DBSCAN con ε={eps}, min_samples={min_samples}", fontsize=18)
    ax.set_xlabel("X1", fontsize=15)
    ax.set_ylabel("X2", fontsize=15)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.legend(loc="center left", bbox_to_anchor=(1.02,0.5), ncol=1, fontsize = 11.85)
    plt.tight_layout(rect=(0,0,0.75,1))
    plt.show()
    
    
# 2
    
def plot_mse_vs_components(components, mse_list):
    plt.figure(figsize=(8,5))
    plt.plot(components, np.array(mse_list), marker='^', linewidth=1.5, color = "mediumpurple")

    plt.axvline(x=43, color='crimson', linestyle='--', linewidth=1.2, label='x = 43')
    plt.axvline(x=59, color='forestgreen', linestyle='--', linewidth=1.2, label='x = 59')
    plt.axvline(x=87, color='darkorange', linestyle='--', linewidth=1.2, label='x = 87')

    plt.xlabel('Número de componentes principales', fontsize=14)
    plt.ylabel('Error cuadrático medio de reconstrucción', fontsize=13)
    plt.title('PCA: MSE vs. nº de componentes', fontsize=16)
    plt.yticks(fontsize=12)
    plt.xticks(components, fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    
def plot_PCA(X, X_rec, k):
    n_show = 10
    originals = X[:n_show].get().reshape(n_show, 28, 28)
    reconstructions = X_rec[:n_show].get().reshape(n_show, 28, 28)

    fig, axes = plt.subplots(2, n_show, figsize=(n_show*1.5, 3))
    for i in range(n_show):
        axes[0, i].imshow(originals[i], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')

        axes[1, i].imshow(reconstructions[i], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstruida\n(k=%d)' % k)
    plt.tight_layout()
    plt.show()
    

def plot_PCA_different_variances(X, pca_full, n_show):
    variances = [0.80, 0.85, 0.90]
    recons = {}  
    cum_var = pca_full.cummulative_variance() 

    for v in variances:
        k = int(cp.argmax(cum_var >= v) + 1)
        pca_k = PCA(n_components=k).fit(X)
        X_proj = pca_k.transform(X)
        X_rec = pca_k.inverse_transform(X_proj)
        recons[v] = {
            'k': k,
            'images': X_rec[:10].get().reshape(10, 28, 28)
        }
        print(f"{int(v*100)}% varianza → k = {k}")

    originals = X[:10].get().reshape(10, 28, 28)

    n_rows = 1 + len(variances) 
    fig, axes = plt.subplots(n_rows, n_show, figsize=(n_show*1.5, n_rows*1.5))

    for i in range(n_show):
        ax = axes[0, i]
        ax.imshow(originals[i], cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title("Original", fontsize=12)

    # reconstrucciones
    for row, v in enumerate(variances, start=1):
        imgs = recons[v]['images']
        k    = recons[v]['k']
        for i in range(n_show):
            ax = axes[row, i]
            ax.imshow(imgs[i], cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title(f"{int(v*100)}% varianza\n(k={k})", fontsize=12)

    plt.tight_layout()
    plt.show()
    
    

def plot_comparison_og_VAE_PCA(x_test_batch, rec_vae, rec_pca, test_size, best_latent, k):
    orig_np = x_test_batch.cpu().numpy().reshape(-1, 28, 28)
    vae_np  = rec_vae.reshape(-1, 28, 28)
    pca_np  = rec_pca.reshape(-1, 28, 28)

    fig, axes = plt.subplots(3, test_size, figsize=(test_size*1.5, 5))
    for i in range(test_size):
        # original
        axes[0, i].imshow(orig_np[i], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Original", fontsize=12)

        # VAE
        axes[1, i].imshow(vae_np[i], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title(f"VAE (z={best_latent})", fontsize=12)

        # PCA
        axes[2, i].imshow(pca_np[i], cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title(f"PCA (k={k})", fontsize=12)

    plt.tight_layout()
    plt.show()