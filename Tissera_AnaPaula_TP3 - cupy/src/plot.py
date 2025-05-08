# plot.py
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib import font_manager


def plot_images(X, y, indices=None, n_cols=5, cmap='gray', figsize_scale=3,
                title_bg_color='#AEC6CF', title_color='#333333', title_alpha=0.6,
                suptitle=None, suptitle_bg='#FFB7C5', suptitle_color='#333333',
                suptitle_alpha=0.6, name_map=None, random_seed=42):
    """
    Muestra imágenes en mosaico con títulos pastel y alpha.
    Si indices=None, elige 1 imagen aleatoria de cada clase distinta (hasta n_cols).
    Acepta X, y como arrays de CuPy o NumPy.
    """
    # Selección automática de índices
    if indices is None:
        cp.random.seed(random_seed)
        y_cp = cp.array(y)
        clases_cp = cp.unique(y_cp)
        clases = clases_cp.get()

        if len(clases) >= n_cols:
            clases_sel = cp.random.choice(clases, size=n_cols, replace=False).get()
        else:
            clases_sel = cp.random.choice(clases, size=n_cols, replace=True).get()

        indices = []
        for cl in clases_sel:
            idxs_cl = cp.where(y_cp == cl)[0]
            idx = int(cp.random.choice(idxs_cl, size=1)[0])
            indices.append(idx)

    N = len(indices)
    n_rows = (N + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(1.95 * figsize_scale, n_rows * figsize_scale),
                            facecolor='#FFFFFF')
    axs = axs.flatten()

    custom_font = font_manager.FontProperties(family='sans-serif',
                                              weight='bold', size=16)

    if suptitle:
        fig.text(0.5, 0.98, suptitle,
                 ha='center', va='top',
                 fontsize=18, fontweight='bold',
                 color=suptitle_color,
                 bbox=dict(facecolor=suptitle_bg, edgecolor='none',
                           alpha=suptitle_alpha,
                           boxstyle='round,pad=0.4'))

    for ax in axs:
        ax.axis('off')

    for i, idx in enumerate(indices):
        ax = axs[i]
        img = X[idx]
        # Si es array de CuPy, pasarlo a NumPy para imshow
        if isinstance(img, cp.ndarray):
            img = cp.asnumpy(img)

        if img.ndim == 1:
            img = img.reshape(28, 28)

        # Extraemos el escalar de la etiqueta antes de usarlo como clave
        raw_label = y[idx]
        label = int(raw_label.item()) if isinstance(raw_label, cp.ndarray) else int(raw_label)
        display_name = name_map.get(label, str(label)) if name_map else str(label)

        ax.imshow(img, cmap=cmap)
        ax.set_title(display_name,
                     fontproperties=custom_font,
                     color=title_color,
                     bbox=dict(facecolor=title_bg_color,
                               edgecolor='none',
                               alpha=title_alpha,
                               boxstyle='round,pad=0.3'),
                     pad=4)

    plt.tight_layout(rect=[0.05, 0.01, 0.95, 0.92])
    plt.show()

def plot_class_distribution(y, name_map=None, figsize=(8, 4), color='skyblue'):
    """
    Muestra por pantalla el conteo de cada clase y dibuja un gráfico de barras.
    Acepta y como array de CuPy o NumPy.
    """
    y_cp = cp.array(y)
    clases_cp, conteos_cp = cp.unique(y_cp, return_counts=True)
    clases   = clases_cp.get()
    conteos  = conteos_cp.get()

    plt.figure(figsize=figsize)
    plt.bar(clases, conteos, color=color)
    plt.xlabel("Clase", fontsize=15)
    plt.ylabel("Cantidad de ejemplos", fontsize=15)
    plt.title("Distribución de clases", fontsize=18)
    plt.xticks(clases, fontsize=10.5)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_loss(epochs, train_losses, val_losses=None):
    """
    Dibuja la evolución de la función de costo.
    train_losses y val_losses pueden ser listas o arrays de CuPy/NumPy.
    """
    # Si vienen como CuPy arrays, convertirlos a NumPy para matplotlib
    if hasattr(train_losses, 'get'):
        train_losses = cp.asnumpy(train_losses)
    if val_losses is not None and hasattr(val_losses, 'get'):
        val_losses = cp.asnumpy(val_losses)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    if val_losses is not None:
        plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Época")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Evolución de la Función de Costo")
    plt.legend()
    plt.tight_layout()
    plt.show()
