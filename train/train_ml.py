"""传统 ML 基线: 分层 K-fold 交叉验证 + SMOTE(可选) + 类权重. 三任务共用."""
import argparse, os, sys, json
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_config, load_dataset
from models.ml_baseline import build_models
from utils.metrics import classification_metrics, print_metrics
from utils.cv import outer_split, stratified_folds
from utils.seed import set_seed

def build_feature_matrix(bundle):
    F = bundle["F"]; M = bundle["M"]
    return np.concatenate([M, F], axis=1)   # 基础信息 + 手工特征

def resample(Xtr, ytr, strategy):
    if strategy == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            k = max(1, min(5, min(np.bincount(ytr))-1))
            if k < 1: return Xtr, ytr
            return SMOTE(k_neighbors=k, random_state=42).fit_resample(Xtr, ytr)
        except Exception as e:
            print("[warn] SMOTE 失败, 跳过:", e)
    return Xtr, ytr

def evaluate(cfg, task):
    bundle = load_dataset(cfg)
    groups = bundle["groups"]
    X = build_feature_matrix(bundle)
    y = bundle["y"][task]
    tr_idx, te_idx = outer_split(y, cfg["cv"]["test_ratio"], cfg["seed"], groups=groups)
    Xtr, Xte, ytr, yte = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

    models = build_models(task, cfg["imbalance"]["strategy"])
    all_res = {}
    for name, model in models.items():
        cv_scores = []
        for k, (a, b) in enumerate(stratified_folds(ytr, cfg["cv"]["n_folds"], cfg["seed"], groups=groups[tr_idx])):
            Xa, ya = Xtr[a], ytr[a]
            Xa, ya = resample(Xa, ya, cfg["imbalance"]["strategy"])
            model.fit(Xa, ya)
            pred = model.predict(Xtr[b])
            prob = model.predict_proba(Xtr[b]) if hasattr(model, "predict_proba") else None
            m = classification_metrics(ytr[b], pred, prob)
            cv_scores.append(m); print_metrics(m, f"{name} fold{k}")
        # 用全部训练集再拟一次, 报测试集
        Xa2, ya2 = resample(Xtr, ytr, cfg["imbalance"]["strategy"])
        model.fit(Xa2, ya2)
        pred_te = model.predict(Xte)
        prob_te = model.predict_proba(Xte) if hasattr(model, "predict_proba") else None
        m_te = classification_metrics(yte, pred_te, prob_te)
        print_metrics(m_te, f"{name} TEST")
        all_res[name] = {"cv": cv_scores, "test": m_te}
    out_dir = cfg["output"]["logs"]; os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"ml_{task}.json"), "w", encoding="utf-8") as f:
        json.dump(all_res, f, ensure_ascii=False, indent=2)
    return all_res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--task", choices=["binary","grade","syndrome"], default="binary")
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"])
    evaluate(cfg, args.task)
