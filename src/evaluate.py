"""
evaluate.py
-----------
Evaluation utilities: metrics, confusion matrix, ROC curve.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    classification_report,
)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, y_prob=None, model_name: str = "Model") -> dict:
    metrics = {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        metrics["ROC-AUC"] = round(roc_auc_score(y_true, y_prob), 4)
    else:
        metrics["ROC-AUC"] = None
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name: str, save: bool = True):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    if save:
        path = os.path.join(REPORTS_DIR, f"cm_{model_name.replace(' ', '_')}.png")
        fig.savefig(path, dpi=150)
        print(f"[INFO] Saved confusion matrix → {path}")
    plt.close(fig)


def plot_roc_curves(roc_data: list, save: bool = True):
    """roc_data: list of (model_name, fpr, tpr, auc) tuples"""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, fpr, tpr, auc in roc_data:
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save:
        path = os.path.join(REPORTS_DIR, "roc_curves.png")
        fig.savefig(path, dpi=150)
        print(f"[INFO] Saved ROC curves → {path}")
    plt.close(fig)


def plot_model_comparison(results_df: pd.DataFrame, save: bool = True):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    df_plot = results_df.set_index("Model")[metrics]
    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot.T.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="white")
    ax.set_ylim(0.5, 1.01)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend(loc="lower right", fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    if save:
        path = os.path.join(REPORTS_DIR, "model_comparison.png")
        fig.savefig(path, dpi=150)
        print(f"[INFO] Saved model comparison → {path}")
    plt.close(fig)


def save_comparison_table(results_df: pd.DataFrame):
    path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    results_df.to_csv(path, index=False)
    print(f"[INFO] Saved comparison table → {path}")
