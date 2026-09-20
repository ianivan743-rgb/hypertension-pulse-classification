"""传统 ML: RF / XGBoost / SVM / LR, 通过分层 CV + 类不均衡处理评估."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def build_models(task, imbalance="class_weight"):
    class_weight = "balanced" if imbalance == "class_weight" else None
    ms = {
        "rf":  RandomForestClassifier(n_estimators=500, max_depth=None,
                                      class_weight=class_weight,
                                      n_jobs=-1, random_state=42),
        "svm": Pipeline([("sc", StandardScaler()),
                         ("svm", SVC(probability=True, kernel="rbf",
                                     class_weight=class_weight, random_state=42))]),
        "lr":  Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=2000,
                                                   class_weight=class_weight,
                                                   random_state=42))]),
    }
    if HAS_XGB:
        ms["xgb"] = XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            eval_metric="mlogloss" if task != "binary" else "logloss",
            tree_method="hist", n_jobs=-1, random_state=42)
    return ms
