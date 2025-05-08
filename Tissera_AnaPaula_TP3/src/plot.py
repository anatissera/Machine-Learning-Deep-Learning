import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

def plot_images(X, y, indices=None, n_cols=5, cmap='gray', figsize_scale=3, title_bg_color='#AEC6CF', title_color='#333333',
    title_alpha=0.6, suptitle=None, suptitle_bg='#FFB7C5', suptitle_color='#333333', suptitle_alpha=0.6, name_map=None, random_seed=42):
    """
    Muestra imágenes en mosaico con títulos pastel y alpha.
    Si indices=None, elige 1 imagen aleatoria de cada clase distinta (hasta n_cols).
    """
    
    # 1) Elegir índices si no vienen dados
    if indices is None:
        if random_seed is not None:
            np.random.seed(random_seed)
        clases = np.unique(y)

        if len(clases) >= n_cols:
            clases_seleccion = np.random.choice(clases, size=n_cols, replace=False)
        else:
            clases_seleccion = np.random.choice(clases, size=n_cols, replace=True)
        indices = []
        for cl in clases_seleccion:
            idxs_cl = np.where(y == cl)[0]
            indices.append(int(np.random.choice(idxs_cl)))

    N = len(indices)
    n_rows = (N + n_cols - 1) // n_cols

    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(1.95 * figsize_scale, n_rows * figsize_scale),
        facecolor='#FFFFFF'
    )
    axs = axs.flatten()

    custom_font = font_manager.FontProperties(family='sans-serif', weight='bold', size=16)

    if suptitle:
        fig.text(
            0.5, 0.98, suptitle,
            ha='center', va='top',
            fontsize=18, fontweight='bold',
            color=suptitle_color,
            bbox=dict(
                facecolor=suptitle_bg,
                edgecolor='none',
                alpha=suptitle_alpha,
                boxstyle='round,pad=0.4'
            )
        )

    for ax in axs:
        ax.axis('off')

    for i, idx in enumerate(indices):
        ax = axs[i]
        img = X[idx]
        if img.ndim == 1:
            img = img.reshape(28, 28)

        # Etiqueta amigable
        label = y[idx]
        display_name = name_map.get(label, str(label)) if name_map else str(label)

        ax.imshow(img, cmap=cmap)
        ax.set_title(
            display_name,
            fontproperties=custom_font,
            color=title_color,
            bbox=dict(
                facecolor=title_bg_color,
                edgecolor='none',
                alpha=title_alpha,
                boxstyle='round,pad=0.3'
            ),
            pad=4
        )

    plt.tight_layout(rect=[0.05, 0.01, 0.95, 0.92])
    plt.show()


def plot_class_distribution(y, name_map=None, figsize=(8, 4), color='skyblue'):
    """
    Muestra por pantalla el conteo de cada clase y dibuja un gráfico de barras.
    
    Parámetros:
    - y: array de etiquetas (shape (N,))
    - name_map: dict opcional {etiqueta_num: 'NombreClase'} para rotular el eje x
    - figsize: tupla (ancho, alto) de la figura
    - color: color de las barras
    """
    # Cálculo de clases y conteos
    clases, conteos = np.unique(y, return_counts=True)
    

    plt.figure(figsize=figsize)
    plt.bar(clases, conteos, color=color)
    plt.xlabel("Clase", fontsize=15)
    plt.ylabel("Cantidad de ejemplos", fontsize=15)
    plt.title("Distribución de clases", fontsize=18)
    plt.xticks(clases, fontsize = 10.5)  
    plt.yticks(fontsize = 12)
    plt.tight_layout()
    plt.show()
    
def plot_loss(epochs, train_losses, val_losses=None):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
    if val_losses:
        plt.plot(range(1, epochs+1), val_losses, label="Val Loss")
    plt.xlabel("Época")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Evolución de la Función de Costo")
    plt.legend()
    plt.tight_layout()
    plt.show()