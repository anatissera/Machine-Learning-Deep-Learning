import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Markdown


# def accuracy(y_true, y_pred):
#     return np.sum(y_true == y_pred) / len(y_true)

# def precision(y_true, y_pred, labels=None):
#     targets = labels if labels is not None else np.unique(np.concatenate((y_true, y_pred)))
#     class_precisions = []

#     for label in targets:
#         predicted_positive = (y_pred == label)
#         actual_positive = (y_true == label)
#         true_positive = np.sum(predicted_positive & actual_positive)
#         false_positive = np.sum(predicted_positive & ~actual_positive)

#         precision_score = true_positive / (true_positive + false_positive + 1e-15)
#         class_precisions.append(precision_score)

#     return np.mean(class_precisions)

# def recall(y_true, y_pred, labels=None):
#     classes = labels if labels is not None else np.unique(np.concatenate((y_true, y_pred)))
#     recall_scores = []

#     for label in classes:
#         condition_positive = (y_true == label)
#         prediction_match = (y_pred == label)
#         true_positive = np.sum(condition_positive & prediction_match)
#         false_negative = np.sum(condition_positive & ~prediction_match)

#         recall_val = true_positive / (true_positive + false_negative + 1e-15)
#         recall_scores.append(recall_val)

#     return np.mean(recall_scores)

def f1_score_macro_binary(y_true, y_pred, labels=None):
    all_labels = labels if labels is not None else np.unique(np.concatenate((y_true, y_pred)))
    f1_scores = []

    for label in all_labels:
        actual = (y_true == label)
        predicted = (y_pred == label)
        tp = np.sum(actual & predicted)
        fp = np.sum(~actual & predicted)
        fn = np.sum(actual & ~predicted)

        prec = tp / (tp + fp + 1e-15)
        rec = tp / (tp + fn + 1e-15)
        f1 = 2 * prec * rec / (prec + rec + 1e-15)
        f1_scores.append(f1)

    return np.mean(f1_scores)

# # --------- Confusion Matrix ---------
# def plot_conf_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
#     classes = labels if labels is not None else np.unique(np.concatenate((y_true, y_pred)))
#     class_count = len(classes)
#     matrix = np.zeros((class_count, class_count), dtype=int)

#     for i, actual in enumerate(classes):
#         for j, predicted in enumerate(classes):
#             matrix[i, j] = np.sum((y_true == actual) & (y_pred == predicted))

#     fig, ax = plt.subplots()
#     color_map = ax.imshow(matrix, cmap="RdPu")

#     ax.set_xticks(np.arange(class_count))
#     ax.set_yticks(np.arange(class_count))
#     ax.set_xticklabels(classes)
#     ax.set_yticklabels(classes)

#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("True")
#     ax.set_title(title)

#     for i in range(class_count):
#         for j in range(class_count):
#             ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")

#     plt.colorbar(color_map)
#     plt.tight_layout()
#     plt.show()


# def roc_curve(y_true, y_scores, thresholds=None):
#     thresholds = thresholds if thresholds is not None else np.linspace(0, 1, 101)

#     tpr_values = []
#     fpr_values = []

#     for thresh in thresholds:
#         predictions = (y_scores >= thresh).astype(int)
#         tp = np.sum((y_true == 1) & (predictions == 1))
#         tn = np.sum((y_true == 0) & (predictions == 0))
#         fp = np.sum((y_true == 0) & (predictions == 1))
#         fn = np.sum((y_true == 1) & (predictions == 0))

#         tpr = tp / (tp + fn + 1e-15)
#         fpr = fp / (fp + tn + 1e-15)

#         tpr_values.append(tpr)
#         fpr_values.append(fpr)

#     return np.array(fpr_values), np.array(tpr_values)

# def pr_curve(y_true, y_scores, thresholds=None):
#     thresholds = thresholds if thresholds is not None else np.linspace(0, 1, 101)

#     precision_vals = []
#     recall_vals = []

#     for thresh in thresholds:
#         preds = (y_scores >= thresh).astype(int)
#         tp = np.sum((y_true == 1) & (preds == 1))
#         fp = np.sum((y_true == 0) & (preds == 1))
#         fn = np.sum((y_true == 1) & (preds == 0))

#         prec = tp / (tp + fp + 1e-15)
#         rec = tp / (tp + fn + 1e-15)

#         precision_vals.append(prec)
#         recall_vals.append(rec)

#     return np.array(recall_vals), np.array(precision_vals)

# def auc(x, y):
#     order = np.argsort(x)
#     return np.trapz(y[order], x[order])

# def plot_roc_curve(y_true, y_proba, labels=None, show=True):
#     classes = labels if labels is not None else np.unique(y_true)
#     y_true = np.asarray(y_true)
#     auc_values = []

#     plt.figure()
#     for idx, cls in enumerate(classes):
#         binary_true = (y_true == cls).astype(int)
#         scores = y_proba[:, idx]
#         fpr, tpr = roc_curve(binary_true, scores)
#         area = auc(fpr, tpr)
#         auc_values.append(area)
#         plt.plot(fpr, tpr, label=f"Class {cls} (AUC = {area:.4f})")

#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("Multiclass ROC Curve")
#     plt.grid(True)
#     plt.legend()
#     if show:
#         plt.show()

#     return auc_values

# def plot_pr_curve(y_true, y_proba, labels=None, show=True):
#     classes = labels if labels is not None else np.unique(y_true)
#     y_true = np.asarray(y_true)
#     auc_list = []

#     plt.figure()
#     for idx, cls in enumerate(classes):
#         binary_true = (y_true == cls).astype(int)
#         class_scores = y_proba[:, idx]
#         recall_seq, precision_seq = pr_curve(binary_true, class_scores)
#         area = auc(recall_seq[::-1], precision_seq[::-1])
#         auc_list.append(area)
#         plt.plot(recall_seq, precision_seq, label=f"Class {cls} (AUC = {area:.4f})")

#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title("Multiclass Precision-Recall Curve")
#     plt.grid(True)
#     plt.legend()
#     if show:
#         plt.show()

#     return auc_list


# def report_multiclass_metrics(y_true, y_pred, y_proba, dataset_name, set_type):
#     """
#     Reporta métricas de evaluación para clasificación multiclase.

#     Parámetros:
#         y_true (np.array): Etiquetas verdaderas.
#         y_pred (np.array): Etiquetas predichas.
#         y_proba (np.array): Probabilidades predichas para cada clase (n_samples, n_classes).
#         dataset_name (str): Nombre del dataset.
#         set_type (str): Tipo de conjunto (Train, Val, Test).
#     """
#     acc = accuracy(y_true, y_pred)
#     prec = precision(y_true, y_pred)
#     rec = recall(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)

#     auc_roc_per_class = plot_roc_curve(y_true, y_proba, show=False)
#     plt.close()
#     auc_pr_per_class = plot_pr_curve(y_true, y_proba, show=False)
#     plt.close()

#     mean_auc_roc = np.mean(auc_roc_per_class)
#     mean_auc_pr = np.mean(auc_pr_per_class)

#     metrics_df = pd.DataFrame({
#         "Métrica": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC", "AUC-PR"],
#         "Valor": [acc, prec, rec, f1, mean_auc_roc, mean_auc_pr]
#     })

#     markdown_table = metrics_df.to_markdown(index=False, floatfmt=".4f")
#     display(Markdown(f"### Métricas de Evaluación para el conjunto de **{set_type}** del set **{dataset_name}**\n\n{markdown_table}"))

#     # Mostrar matriz de confusión y curvas
    
#     plot_conf_matrix(y_true, y_pred)
#     plot_roc_curve(y_true, y_proba)
#     plot_pr_curve(y_true, y_proba)
    
    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

class MulticlassMetrics:
    def __init__(self, y_true, y_pred, y_proba, labels=None):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.y_proba = np.asarray(y_proba)
        self.labels = labels if labels is not None else np.unique(np.concatenate([self.y_true, self.y_pred]))

    def accuracy(self):
        return np.sum(self.y_true == self.y_pred) / len(self.y_true)

    def precision(self):
        scores = []
        for label in self.labels:
            pred_pos = (self.y_pred == label)
            true_pos = (self.y_true == label)
            tp = np.sum(pred_pos & true_pos)
            fp = np.sum(pred_pos & ~true_pos)
            score = tp / (tp + fp + 1e-15)
            scores.append(score)
        return np.mean(scores)

    def recall(self):
        scores = []
        for label in self.labels:
            actual = (self.y_true == label)
            predicted = (self.y_pred == label)
            tp = np.sum(actual & predicted)
            fn = np.sum(actual & ~predicted)
            score = tp / (tp + fn + 1e-15)
            scores.append(score)
        return np.mean(scores)

    def f1_score(self):
        scores = []
        for label in self.labels:
            actual = (self.y_true == label)
            predicted = (self.y_pred == label)
            tp = np.sum(actual & predicted)
            fp = np.sum(~actual & predicted)
            fn = np.sum(actual & ~predicted)
            prec = tp / (tp + fp + 1e-15)
            rec = tp / (tp + fn + 1e-15)
            f1 = 2 * prec * rec / (prec + rec + 1e-15)
            scores.append(f1)
        return np.mean(scores)

    def _roc_curve(self, binary_true, scores, thresholds=None):
        thresholds = thresholds or np.linspace(0, 1, 101)
        tpr, fpr = [], []

        for t in thresholds:
            pred = (scores >= t).astype(int)
            tp = np.sum((binary_true == 1) & (pred == 1))
            tn = np.sum((binary_true == 0) & (pred == 0))
            fp_val = np.sum((binary_true == 0) & (pred == 1))
            fn = np.sum((binary_true == 1) & (pred == 0))
            tpr.append(tp / (tp + fn + 1e-15))
            fpr.append(fp_val / (fp_val + tn + 1e-15))
        return np.array(fpr), np.array(tpr)

    def _pr_curve(self, binary_true, scores, thresholds=None):
        thresholds = thresholds or np.linspace(0, 1, 101)
        prec, rec = [], []

        for t in thresholds:
            pred = (scores >= t).astype(int)
            tp = np.sum((binary_true == 1) & (pred == 1))
            fp = np.sum((binary_true == 0) & (pred == 1))
            fn = np.sum((binary_true == 1) & (pred == 0))
            prec.append(tp / (tp + fp + 1e-15))
            rec.append(tp / (tp + fn + 1e-15))
        return np.array(rec), np.array(prec)

    def _auc(self, x, y):
        idx = np.argsort(x)
        return np.trapz(y[idx], x[idx])

    def plot_conf_matrix(self, title="Confusion Matrix"):
        n = len(self.labels)
        mat = np.zeros((n, n), dtype=int)
        for i, actual in enumerate(self.labels):
            for j, pred in enumerate(self.labels):
                mat[i, j] = np.sum((self.y_true == actual) & (self.y_pred == pred))

        fig, ax = plt.subplots()
        im = ax.imshow(mat, cmap="RdPu")
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(self.labels)
        ax.set_yticklabels(self.labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

        for i in range(n):
            for j in range(n):
                ax.text(j, i, mat[i, j], ha='center', va='center', color='black')

        plt.colorbar(im)
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, show=True):
        aucs = []
        plt.figure()
        for i, label in enumerate(self.labels):
            binary_true = (self.y_true == label).astype(int)
            class_scores = self.y_proba[:, i]
            fpr, tpr = self._roc_curve(binary_true, class_scores)
            area = self._auc(fpr, tpr)
            aucs.append(area)
            plt.plot(fpr, tpr, label=f"Class {label} (AUC = {area:.4f})")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multiclass ROC Curve")
        plt.grid(True)
        plt.legend()
        if show:
            plt.show()
        return aucs

    def plot_pr_curve(self, show=True):
        aucs = []
        plt.figure()
        for i, label in enumerate(self.labels):
            binary_true = (self.y_true == label).astype(int)
            class_scores = self.y_proba[:, i]
            recall_vals, precision_vals = self._pr_curve(binary_true, class_scores)
            area = self._auc(recall_vals[::-1], precision_vals[::-1])
            aucs.append(area)
            plt.plot(recall_vals, precision_vals, label=f"Class {label} (AUC = {area:.4f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Multiclass PR Curve")
        plt.grid(True)
        plt.legend()
        if show:
            plt.show()
        return aucs

    def report_metrics(self, dataset_name="Dataset", set_type="Test"):
        acc = self.accuracy()
        prec = self.precision()
        rec = self.recall()
        f1 = self.f1_score()
        auc_roc_vals = self.plot_roc_curve(show=False)
        plt.close()
        auc_pr_vals = self.plot_pr_curve(show=False)
        plt.close()

        auc_roc_mean = np.mean(auc_roc_vals)
        auc_pr_mean = np.mean(auc_pr_vals)

        df = pd.DataFrame({
            "Métrica": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC", "AUC-PR"],
            "Valor": [acc, prec, rec, f1, auc_roc_mean, auc_pr_mean]
        })

        display(Markdown(f"### Métricas de Evaluación para el conjunto de **{set_type}** del set **{dataset_name}**\n\n" + df.to_markdown(index=False, floatfmt=".4f")))
        self.plot_conf_matrix()
        self.plot_roc_curve()
        self.plot_pr_curve()

def f1_score_macro_multiclass(y_true, y_pred):
    labels = np.unique(np.concatenate((y_true, y_pred)))
    f1s = []
    for label in labels:
        actual = (y_true == label)
        predicted = (y_pred == label)
        tp = np.sum(actual & predicted)
        fp = np.sum(~actual & predicted)
        fn = np.sum(actual & ~predicted)
        prec = tp / (tp + fp + 1e-15)
        rec = tp / (tp + fn + 1e-15)
        f1 = 2 * prec * rec / (prec + rec + 1e-15)
        f1s.append(f1)
    return np.mean(f1s)