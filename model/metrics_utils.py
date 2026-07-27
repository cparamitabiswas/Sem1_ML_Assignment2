"""Evaluation shared by the training notebook and the Streamlit app.

Keeping one implementation means the "benchmark" numbers computed while
training and the "live" numbers the app computes on an uploaded CSV are
always calculated the exact same way.
"""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Six required metrics, macro-averaged where the dataset's 3 classes need it."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def confusion_and_report(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    report = classification_report(
        y_true, y_pred, labels=class_names, output_dict=True, zero_division=0
    )
    return cm, report
