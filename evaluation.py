"""Evaluation utilities. Positive class: benign (1)."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    labels = np.asarray(y_test)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Expected binary model containing positive label 1.")
    probability_1 = model.predict_proba(X_test)[:, classes.index(1)]
    auc = float(roc_auc_score(labels, probability_1)) if len(np.unique(labels)) == 2 else np.nan
    return {
        "Accuracy": float(accuracy_score(labels, predictions)),
        "AUC": auc,
        "Precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "F1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, predictions)),
    }

def diagnostic_report(model, X_test, y_test):
    predictions = model.predict(X_test)
    return (
        confusion_matrix(y_test, predictions, labels=[0, 1]),
        classification_report(y_test, predictions, labels=[0, 1],
                              target_names=["Malignant (0)", "Benign (1)"],
                              zero_division=0),
        predictions,
    )
