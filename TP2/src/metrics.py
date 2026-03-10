
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

EPSILON = 1e-15

# Problema 1

class BinaryMetrics:
    def __init__(self, y_true, y_scores, threshold=0.5):
        self.y_true = np.asarray(y_true)
        self.y_scores = np.asarray(y_scores)
        self.threshold = threshold
        self.y_pred = (self.y_scores >= threshold).astype(int)

    def conf_matrix(self):
        tn = np.sum((self.y_true == 0) & (self.y_pred == 0))
        fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        fn = np.sum((self.y_true == 1) & (self.y_pred == 0))
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        return np.array([[tn, fp], [fn, tp]])

    def accuracy(self):
        return np.mean(self.y_true == self.y_pred)

    def precision(self):
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        fp = np.sum(self.y_pred == 1) - tp
        return tp / (tp + fp + EPSILON)

    def recall(self):
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        fn = np.sum(self.y_true == 1) - tp
        return tp / (tp + fn + EPSILON)

    def f1_score(self):
        prec = self.precision()
        rec = self.recall()
        return 2 * prec * rec / (prec + rec + EPSILON)

    def auc(self, x, y):
        sorted_idx = np.argsort(x)
        return np.trapz(y[sorted_idx], x[sorted_idx])

    def roc_curve(self, thresholds=None):
        thresholds = thresholds or np.linspace(0, 1, 101)
        tpr_list = []
        fpr_list = []

        for t in thresholds:
            preds = (self.y_scores >= t).astype(int)
            tp = np.sum((self.y_true == 1) & (preds == 1))
            fp = np.sum((self.y_true == 0) & (preds == 1))
            tn = np.sum((self.y_true == 0) & (preds == 0))
            fn = np.sum((self.y_true == 1) & (preds == 0))

            tpr = tp / (tp + fn + EPSILON)
            fpr = fp / (fp + tn + EPSILON)

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        return np.array(fpr_list), np.array(tpr_list)

    def pr_curve(self, thresholds=None):
        thresholds = thresholds or np.linspace(0, 1, 101)
        precisions = []
        recalls = []

        for t in thresholds:
            preds = (self.y_scores >= t).astype(int)
            tp = np.sum((self.y_true == 1) & (preds == 1))
            fp = np.sum((self.y_true == 0) & (preds == 1))
            fn = np.sum((self.y_true == 1) & (preds == 0))

            prec = tp / (tp + fp + EPSILON)
            rec = tp / (tp + fn + EPSILON)

            precisions.append(prec)
            recalls.append(rec)

        return np.array(recalls), np.array(precisions)

    def plot_conf_matrix(self, labels=["Negative", "Positive"], title="Confusion Matrix"):
        matrix = self.conf_matrix()

        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        im = ax.imshow(matrix, cmap="tab20b", alpha=0.9)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='black')

        plt.colorbar(im)
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, label=None, show=True, plot_color=None):
        fpr, tpr = self.roc_curve()
        auc_val = self.auc(fpr, tpr)
        
        if plot_color:
            plt.plot(fpr, tpr, label=label or f"ROC AUC = {auc_val:.4f}", color=plot_color, linewidth=3.5)
        else:
            plt.plot(fpr, tpr, label=label or f"ROC AUC = {auc_val:.4f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.grid(True)
        plt.legend()
        if show:
            plt.show()
        return auc_val

    def plot_pr_curve(self, label=None, show=True, plot_color=None):
        recall_vals, precision_vals = self.pr_curve()
        auc_val = self.auc(recall_vals[::-1], precision_vals[::-1])

        if plot_color:
            plt.plot(recall_vals, precision_vals, label=label or f"PR AUC = {auc_val:.4f}", color=plot_color, linewidth=3.5)
        else:
            plt.plot(recall_vals, precision_vals, label=label or f"PR AUC = {auc_val:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(True)
        plt.legend()
        if show:
            plt.show()
        return auc_val

    def report_metrics(self, dataset_name, set_type, roc_color=None, pr_color=None):
        acc = self.accuracy()
        prec = self.precision()
        rec = self.recall()
        f1 = self.f1_score()
        auc_roc = self.plot_roc_curve(show=False, plot_color=roc_color)
        auc_pr = self.plot_pr_curve(show=False, plot_color=pr_color)
        plt.close()

        metrics_df = pd.DataFrame({
            "Métrica": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC", "AUC-PR"],
            "Valor": [acc, prec, rec, f1, auc_roc, auc_pr]
        })

        markdown_table = metrics_df.to_markdown(index=False, floatfmt=".4f")
        display(Markdown(f"### Métricas de Evaluación para el conjunto de **{set_type}** del set **{dataset_name}**\n\n{markdown_table}"))

        self.plot_conf_matrix()
        self.plot_roc_curve(plot_color=roc_color)
        self.plot_pr_curve(plot_color=pr_color)
        
        return metrics_df

def precision(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    pred_pos = (y_pred == 1).sum()
    return tp / (pred_pos + EPSILON)

def recall(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    real_pos = (y_true == 1).sum()
    return tp / (real_pos + EPSILON)

def f1_score_macro_binary(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * prec * rec / (prec + rec + EPSILON)
    

# Problema 2

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

    def roc_curve(self, binary_true, scores, thresholds=None):
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

    def pr_curve(self, binary_true, scores, thresholds=None):
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

    def auc(self, x, y):
        idx = np.argsort(x)
        return np.trapz(y[idx], x[idx])

    def plot_conf_matrix(self, title="Confusion Matrix"):
        n = len(self.labels)
        mat = np.zeros((n, n), dtype=int)
        for i, actual in enumerate(self.labels):
            for j, pred in enumerate(self.labels):
                mat[i, j] = np.sum((self.y_true == actual) & (self.y_pred == pred))

        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        im = ax.imshow(mat, cmap="GnBu")
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(self.labels, fontsize=10)
        ax.set_yticklabels(self.labels, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=14)
        ax.set_ylabel("True",fontsize=14)
        ax.set_title(title, fontsize=14)

        for i in range(n):
            for j in range(n):
                ax.text(j, i, mat[i, j], ha='center', va='center', color='black', fontsize=10)

        plt.colorbar(im)
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, show=True):
        aucs = []
        plt.figure(figsize=(4.5,3.5))
        colors = ['lightseagreen', 'slateblue', 'palevioletred']
        for i, label in enumerate(self.labels):
            binary_true = (self.y_true == label).astype(int)
            class_scores = self.y_proba[:, i]
            fpr, tpr = self.roc_curve(binary_true, class_scores)
            area = self.auc(fpr, tpr)
            aucs.append(area)
            plt.plot(fpr, tpr, label=f"Class {label} (AUC = {area:.4f})", linewidth=2, color= colors[i])

        plt.xlabel("False Positive Rate", fontsize= 14)
        plt.ylabel("True Positive Rate", fontsize= 14)
        plt.title("Multiclass ROC Curve", fontsize= 14)
        plt.xticks(fontsize= 9)
        plt.yticks(fontsize= 9)
        plt.grid(True)
        plt.legend(fontsize= 10)
        if show:
            plt.show()
        return aucs

    def plot_pr_curve(self, show=True):
        aucs = []
        plt.figure(figsize=(4.5,3.5))
        colors = ['lightseagreen', 'slateblue', 'palevioletred']
        for i, label in enumerate(self.labels):
            binary_true = (self.y_true == label).astype(int)
            class_scores = self.y_proba[:, i]
            recall_vals, precision_vals = self.pr_curve(binary_true, class_scores)
            area = self.auc(recall_vals[::-1], precision_vals[::-1])
            aucs.append(area)
            plt.plot(recall_vals, precision_vals, label=f"Class {label} (AUC = {area:.4f})", linewidth=2, color= colors[i])

        plt.xlabel("Recall", fontsize= 14)
        plt.ylabel("Precision", fontsize= 14)
        plt.title("Multiclass PR Curve", fontsize= 14)
        plt.xticks(fontsize= 9)
        plt.yticks(fontsize= 9)
        plt.grid(True)
        plt.legend(fontsize= 10)
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
        
        return df

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