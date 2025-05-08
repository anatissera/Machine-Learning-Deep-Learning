import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

class Metrics:
    """
    Clase para calcular y mostrar métricas de performance:
    - Accuracy
    - Cross-Entropy Loss
    - Matriz de Confusión (texto y gráfica de calor)
    """
    def __init__(self, y_true, y_pred, y_proba=None, labels=None):
        """
        Parámetros:
        - y_true: array de etiquetas verdaderas (shape: (m,))
        - y_pred: array de etiquetas predichas (shape: (m,))
        - y_proba: array de probabilidades predichas (shape: (m, n_classes)), opcional, necesario para cross-entropy
        - labels: lista de etiquetas únicas (en orden) para la matriz de confusión, opcional
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_proba = np.array(y_proba) if y_proba is not None else None
        if labels is not None:
            self.labels = list(labels)
        else:
            self.labels = sorted(list(set(np.concatenate([self.y_true, self.y_pred]))))
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}

    def accuracy(self):
        """Calcula la exactitud (accuracy)."""
        return np.mean(self.y_true == self.y_pred)

    def cross_entropy(self):
        """Calcula la cross-entropy loss (requiere y_proba)."""
        if self.y_proba is None:
            raise ValueError("Se requieren probabilidades predichas (y_proba) para cross-entropy.")
        m = self.y_true.shape[0]
        eps = 1e-15
        probs = self.y_proba[np.arange(m), self.y_true]
        log_probs = -np.log(probs + eps)
        return np.mean(log_probs)

    def confusion_matrix(self):
        """Construye la matriz de confusión."""
        n = len(self.labels)
        cm = np.zeros((n, n), dtype=int)
        for true, pred in zip(self.y_true, self.y_pred):
            i = self.label_to_index[true]
            j = self.label_to_index[pred]
            cm[i, j] += 1
        return cm

    def plot_heatmap_confusion_matrix(self, cmap=None):
        """
        Plotea la matriz de confusión con un heatmap.
        Parámetro:
        - cmap: colormap de matplotlib (opcional)
        """
        cm = self.confusion_matrix()
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(cm, interpolation='nearest', aspect='auto', cmap=cmap)
        fig.colorbar(cax)
        ax.set_xticks(np.arange(len(self.labels)))
        ax.set_yticks(np.arange(len(self.labels)))
        ax.set_xticklabels(self.labels, fontsize= 7)
        ax.set_yticklabels(self.labels, fontsize= 8)
        ax.set_xlabel('Predicción', fontsize = 15)
        ax.set_ylabel('Verdadero', fontsize = 15)
        ax.set_title('Matriz de Confusión', fontsize = 18)
        plt.tight_layout()
        plt.show()

    def report_metrics(self, title="Métricas de Performance", show_matrix_table = True, plot_matrix_heatmap=True, cmap=None):
        """Muestra métricas en Markdown y heatmap opcional. """
        md = f"### {title}\n"
        acc = self.accuracy()
        md += f"**Accuracy:** {acc:.4f}  \n"
        if self.y_proba is not None:
            ce = self.cross_entropy()
            md += f"**Cross-Entropy Loss:** {ce:.4f}  \n"
        
        display(Markdown(md))

        if plot_matrix_heatmap:
            self.plot_heatmap_confusion_matrix(cmap=cmap)
        
        if show_matrix_table:
            cm = self.confusion_matrix()
            header = "|True\\Pred|" + "|".join(str(l) for l in self.labels) + "|\n"
            sep = "|---" * (len(self.labels) + 1) + "|\n"
            rows = ""
            for i, label in enumerate(self.labels):
                row = f"|{label}|" + "|".join(str(x) for x in cm[i]) + "|\n"
                rows += row
            md_cm = "**Matriz de Confusión:**  \n" + header + sep + rows
            display(Markdown(md_cm))


