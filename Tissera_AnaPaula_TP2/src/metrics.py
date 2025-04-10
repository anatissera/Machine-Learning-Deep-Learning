import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Markdown

EPSILON = 1e-15

def conf_matrix(y_true, y_pred):
    true_neg = np.sum((y_true == 0) & (y_pred == 0))
    false_pos = np.sum((y_true == 0) & (y_pred == 1))
    false_neg = np.sum((y_true == 1) & (y_pred == 0))
    true_pos = np.sum((y_true == 1) & (y_pred == 1))

    conf_mat = np.array([[true_neg, false_pos],
                         [false_neg, true_pos]])
    
    return conf_mat

def accuracy(y_true, y_pred):
    correct = (y_true == y_pred).sum()
    return correct / len(y_true)

def precision(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    pred_pos = (y_pred == 1).sum()
    return tp / (pred_pos + EPSILON)

def recall(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    real_pos = (y_true == 1).sum()
    return tp / (real_pos + EPSILON)

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * prec * rec / (prec + rec + EPSILON)

def auc(x, y):
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    return np.trapz(y_sorted, x_sorted)

def roc_curve(y_true, y_scores, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        preds = (y_scores >= threshold).astype(int)

        tp = ((y_true == 1) & (preds == 1)).sum()
        fp = ((y_true == 0) & (preds == 1)).sum()
        fn = ((y_true == 1) & (preds == 0)).sum()
        tn = ((y_true == 0) & (preds == 0)).sum()

        tpr = tp / (tp + fn + EPSILON)
        fpr = fp / (fp + tn + EPSILON)

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(fpr_list), np.array(tpr_list)

def pr_curve(y_true, y_scores, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    precisions = []
    recalls = []

    for threshold in thresholds:
        preds = (y_scores >= threshold).astype(int)

        tp = ((y_true == 1) & (preds == 1)).sum()
        fp = ((y_true == 0) & (preds == 1)).sum()
        fn = ((y_true == 1) & (preds == 0)).sum()

        prec = tp / (tp + fp + EPSILON)
        rec = tp / (tp + fn + EPSILON)

        precisions.append(prec)
        recalls.append(rec)

    return np.array(recalls), np.array(precisions)


# # gráficos
# def plot_conf_matrix(y_true, y_pred, labels=["Negative", "Positive"], title="Confusion Matrix"):
#     conf_mat = conf_matrix(y_true, y_pred)

#     fig, ax = plt.subplots()
#     matrix_plot = ax.imshow(conf_mat, cmap="RdPu")

#     ax.set_xticks([0, 1])
#     ax.set_yticks([0, 1])
#     ax.set_xticklabels(labels)
#     ax.set_yticklabels(labels)
#     ax.set_xlabel("Predicted Label")
#     ax.set_ylabel("True Label")
#     ax.set_title(title)

#     for row in range(2):
#         for col in range(2):
#             ax.text(col, row, str(conf_mat[row, col]), va="center", ha="center", color="black", fontsize=12)

#     plt.colorbar(matrix_plot)
#     plt.tight_layout()
#     plt.show()
    

def plot_conf_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    """
    Dibuja una matriz de confusión para clasificación multiclase o binaria.

    Parámetros:
    - y_true (array): etiquetas reales
    - y_pred (array): etiquetas predichas
    - class_names (list, opcional): nombres de las clases
    - title (str): título del gráfico
    """
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(unique_labels)

    if class_names is None:
        class_names = [str(cls) for cls in unique_labels]

    # Inicializar matriz vacía
    matrix = np.zeros((n_classes, n_classes), dtype=int)

    # Llenar la matriz contando ocurrencias
    for i, actual in enumerate(unique_labels):
        for j, predicted in enumerate(unique_labels):
            matrix[i, j] = np.sum((y_true == actual) & (y_pred == predicted))

    # Graficar la matriz
    fig, ax = plt.subplots()
    cmap = plt.cm.OrRd
    im = ax.imshow(matrix, interpolation='nearest', cmap=cmap)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    # Anotar valores en las celdas
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(matrix[i, j]),
                    ha='center', va='center', color='black')

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    
def plot_roc_curve(y_true, y_scores, label=None, show=True, plot_color=None):
    fpr_vals, tpr_vals = roc_curve(y_true, y_scores)
    auc_val = auc(fpr_vals, tpr_vals)


    if plot_color:
        plt.plot(fpr_vals, tpr_vals, label=label or f"ROC AUC = {auc_val:.4f}", color=plot_color)
    else:
        plt.plot(fpr_vals, tpr_vals, label=label or f"ROC AUC = {auc_val:.4f}") 
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.legend()
    
    if show:
        plt.show()

    return auc_val


def plot_pr_curve(y_true, y_scores, label=None, show=True, plot_color=None):
    rec_vals, prec_vals = pr_curve(y_true, y_scores)
    auc_val = auc(rec_vals[::-1], prec_vals[::-1])

    if plot_color:
        plt.plot(rec_vals, prec_vals, label=label or f"PR AUC = {auc_val:.4f}", color=plot_color) 
    else:
        plt.plot(rec_vals, prec_vals, label=label or f"PR AUC = {auc_val:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.legend()
    
    
    if show:
        plt.show()

    return auc_val


def report_metrics(y_true, y_scores, dataset_name, set_type, threshold=0.5, roc_color=None, pr_color=None):
    """
    Reporta todas las métricas de evaluación para un clasificador binario,
    mostrando los resultados en formato de tabla markdown.

    Parámetros:
        y_true (np.array): Etiquetas verdaderas (0 o 1).
        y_scores (np.array): Puntajes predichos (probabilidades o scores continuos).
        dataset_name (str): Nombre del dataset (e.g., 'Diagnosis').
        set_type (str): Tipo de set (e.g., 'Train', 'Val', 'Test').
        threshold (float): Umbral para convertir scores en etiquetas predichas (default: 0.5).
    """
    # Convertir probabilidades a etiquetas
    y_pred = (y_scores >= threshold).astype(int)

    # Métricas
    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = plot_roc_curve(y_true, y_scores, show=False)
    auc_pr = plot_pr_curve(y_true, y_scores, show=False)

    # Crear DataFrame con métricas
    metrics_df = pd.DataFrame({
        "Métrica": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC", "AUC-PR"],
        "Valor": [acc, prec, rec, f1, auc_roc, auc_pr]
    })

    markdown_table = metrics_df.to_markdown(index=False, floatfmt=".4f")
    display(Markdown(f"### Métricas de Evaluación para el conjunto de **{set_type}** del set **{dataset_name}**\n" + markdown_table))

    # Gráficos
    plt.close()
    plot_conf_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_scores, plot_color=roc_color)
    plot_pr_curve(y_true, y_scores, plot_color=pr_color)
    
    
    
    
    
    
    
    
def roc_curve_multiclass(y_true, y_scores, thresholds=None):
    """
    Calcula los puntos de la curva ROC (TPR vs FPR) para distintos umbrales.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, num=101)

    true_positive_rates = []
    false_positive_rates = []

    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)

        TP = np.sum((y_true == 1) & (preds == 1))
        FP = np.sum((y_true == 0) & (preds == 1))
        FN = np.sum((y_true == 1) & (preds == 0))
        TN = np.sum((y_true == 0) & (preds == 0))

        TPR = TP / (TP + FN + 1e-15)
        FPR = FP / (FP + TN + 1e-15)

        true_positive_rates.append(TPR)
        false_positive_rates.append(FPR)

    return np.array(false_positive_rates), np.array(true_positive_rates)

def pr_curve_multiclass(y_true, y_scores, thresholds=None):
    """
    Calcula los puntos de la curva Precisión-Recall para distintos umbrales.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, num=101)

    precision_values = []
    recall_values = []

    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)

        TP = np.sum((y_true == 1) & (preds == 1))
        FP = np.sum((y_true == 0) & (preds == 1))
        FN = np.sum((y_true == 1) & (preds == 0))

        prec = TP / (TP + FP + 1e-15)
        rec = TP / (TP + FN + 1e-15)

        precision_values.append(prec)
        recall_values.append(rec)

    return np.array(recall_values), np.array(precision_values)
