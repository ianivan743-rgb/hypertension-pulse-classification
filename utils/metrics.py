import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             classification_report)

def classification_metrics(y_true, y_pred, y_prob=None, average="macro"):
    out = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall":    recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1":        f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    if y_prob is not None:
        try:
            if y_prob.ndim == 1 or y_prob.shape[1] == 2:
                p = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
                out["auc"] = roc_auc_score(y_true, p)
            else:
                out["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)
        except Exception:
            out["auc"] = float("nan")
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    out["per_class"] = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    return out

def print_metrics(m, prefix=""):
    keys = ["accuracy", "precision", "recall", "f1", "auc"]
    line = "  ".join(f"{k}={m[k]:.4f}" for k in keys if k in m)
    print(f"[{prefix}] {line}")
