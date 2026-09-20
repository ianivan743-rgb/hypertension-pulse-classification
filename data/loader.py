"""从 CSV 读取样本, 拆出: 波形 X(N,C,T), 手工特征 F(N,D), 基础信息 M(N,K), 三种标签.

支持两种输入 CSV:
    (a) 波形 CSV  (ingest_excel.py 产物, 含 wave_<c>_<t> 列)
    (b) 特征 CSV  (ingest_features.py 产物, 只有手工特征 + 标签)
loader 会自动检测有没有 wave 列; 缺失时 X 用零占位 (深度分支会失效, 但传统 ML 仍可跑).
"""
import os, yaml, numpy as np, pandas as pd


def load_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _has_wave_cols(df, prefix):
    for c in df.columns:
        if str(c).startswith(prefix): return True
    return False


def _extract_waves(df, prefix, n_ch, seq_len):
    cols = [f"{prefix}{c}_{t}" for c in range(n_ch) for t in range(seq_len)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少 {len(missing)} 个波形列, 例如: {missing[:3]}")
    arr = df[cols].to_numpy(dtype=np.float32)
    return arr.reshape(len(df), n_ch, seq_len)


def _safe_get(df, col, dtype=np.float32, fill=0.0):
    if col in df.columns:
        return df[col].to_numpy(dtype=dtype)
    return np.full(len(df), fill, dtype=dtype)


def load_dataset(cfg):
    p = cfg["data"]["csv_path"]
    if not os.path.exists(p):
        from data.synth import save_synth_csv
        print(f"[loader] CSV 不存在, 用合成 demo 数据: {p}")
        save_synth_csv(p, n=200,
                       n_ch=cfg["data"]["n_channels"],
                       seq_len=cfg["data"]["seq_len"])
    df = pd.read_csv(p)
    cols = cfg["data"]["cols"]
    prefix = cols["wave_prefix"]
    n_ch   = cfg["data"]["n_channels"]
    seq_len = cfg["data"]["seq_len"]

    # 波形: 可选. 没有就零填 (方便只有特征表时也能训练传统 ML)
    if _has_wave_cols(df, prefix):
        X = _extract_waves(df, prefix, n_ch, seq_len)
        has_wave = True
    else:
        print("[loader] 未检测到波形列, X 用零占位; 深度模型分支不可用")
        X = np.zeros((len(df), n_ch, seq_len), dtype=np.float32)
        has_wave = False

    # 手工特征: 支持 config 里给定的固定列表, 也支持 auto (取除标签/元数据/id/wave 外的所有数值列)
    if isinstance(cols.get("hand_feats"), list) and cols["hand_feats"]:
        hand = [c for c in cols["hand_feats"] if c in df.columns]
        missing = [c for c in cols["hand_feats"] if c not in df.columns]
        if missing:
            print(f"[loader] 缺失手工特征列 {len(missing)} 个, 例如 {missing[:5]} (填 0)")
        F = np.column_stack([_safe_get(df, c) for c in cols["hand_feats"]]).astype(np.float32)
    else:
        drop = set([cols["id"], cols["label_bin"], cols["label_grade"], cols["label_syndrome"]]
                    + list(cols.get("meta") or []))
        num = df.select_dtypes(include=[np.number])
        keep = [c for c in num.columns if c not in drop and not str(c).startswith(prefix)]
        F = num[keep].to_numpy(dtype=np.float32) if keep else np.zeros((len(df), 0), dtype=np.float32)
        print(f"[loader] auto 手工特征: {len(keep)} 维")

    M = np.column_stack([_safe_get(df, c) for c in (cols.get("meta") or [])]).astype(np.float32)         if cols.get("meta") else np.zeros((len(df), 0), dtype=np.float32)

    y = {
        "binary":   _safe_get(df, cols["label_bin"],      dtype=np.int64, fill=0),
        "grade":    _safe_get(df, cols["label_grade"],    dtype=np.int64, fill=0),
        "syndrome": _safe_get(df, cols["label_syndrome"], dtype=np.int64, fill=0),
    }
    sample_ids = df[cols["id"]].astype(str).to_numpy() if cols["id"] in df.columns else                  np.array([f"S{i}" for i in range(len(df))])
    groups = np.array([s.split("#", 1)[0] for s in sample_ids])
    return {"X": X, "F": F, "M": M, "y": y, "df": df,
            "sample_ids": sample_ids, "groups": groups, "has_wave": has_wave}


def zscore_fit(x):
    mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-8
    return mu, sd
def zscore_apply(x, mu, sd): return (x - mu) / sd
