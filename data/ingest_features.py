"""把三种特征 Excel (时域 / 频域 / APV) 汇成 features.csv, 每行一个受试者.

三种输入约定
------------
1) 时域 (time domain):
   - 每个 .xlsx = 1 位受试者, 24 sheet (对应 24 通道), 每 sheet 若干行 (每行 = 一个心搏周期)
   - 列: t1..t5, h1..h5, w31, w51, w31_t, w51_t, h1_t1, h3_h1, h4_h1, As, Ad, t,
          h1_OP..h5_OP, Beg
   - 聚合策略 (可配置):
       best_channel   选 h1 最大的通道, 该通道内先按行求均值 -> 每人 1 行 (默认, 与论文一致)
       mean_all       24 通道全部求均值 -> 每人 1 行
       per_channel    24 通道各自求均值后 concat -> 每人 24 x D 维 (信息最全, 但维度高)

2) 频域 (frequency domain):
   - 每个 .xlsx = 1 位受试者, 1 sheet, 每行 1 个通道 (name = 1..24)
   - 列: name, ApEn, E, TSEn, MSF, RMSF, frequency_variance, frequency_std_dev,
          频谱峰值_频率_1..5, 频谱峰值_幅度_1..5
   - 聚合: best_channel (E 最大) | mean_all | per_channel

3) APV:
   - 每个 .xlsx = 1 位受试者, 1 sheet, 通常 1 行 (若多行取首行)
   - 全部列直接采用

产物
----
features.csv, 每行:
    sample_id, <time_...>, <freq_...>, <apv_...>

用法
----
    python -m data.ingest_features \\
        --time_dir ../data/time_xlsx  \\
        --freq_dir ../data/freq_xlsx  \\
        --apv_dir  ../data/apv_xlsx   \\
        --out_csv  ../data/features.csv \\
        --time_strategy best_channel   \\
        --freq_strategy best_channel

三个 --*_dir 允许缺省, 缺失部分不会写入.
"""
import argparse, os, glob, sys
import numpy as np, pandas as pd


TIME_COLS = ["t1","t2","t3","t4","t5",
             "h1","h2","h3","h4","h5",
             "w31","w51","w31_t","w51_t",
             "h1_t1","h3_h1","h4_h1",
             "As","Ad","t",
             "h1_OP","h2_OP","h3_OP","h4_OP","h5_OP","Beg"]

FREQ_COLS = ["ApEn","E","TSEn","MSF","RMSF",
             "frequency_variance","frequency_std_dev",
             "频谱峰值_频率_1","频谱峰值_频率_2","频谱峰值_频率_3",
             "频谱峰值_频率_4","频谱峰值_频率_5",
             "频谱峰值_幅度_1","频谱峰值_幅度_2","频谱峰值_幅度_3",
             "频谱峰值_幅度_4","频谱峰值_幅度_5"]


def _sid_from_path(p): return os.path.splitext(os.path.basename(p))[0]


# ---------------- 时域 ----------------
def _time_one_subject(path, strategy="best_channel"):
    xls = pd.ExcelFile(path)
    sheets = xls.sheet_names[:24]
    per_ch = []
    for sh in sheets:
        df = pd.read_excel(xls, sheet_name=sh)
        # 只留约定列, 缺少的填 NaN
        for c in TIME_COLS:
            if c not in df.columns: df[c] = np.nan
        df = df[TIME_COLS].apply(pd.to_numeric, errors="coerce")
        # 通道内按行求均值 (每人每通道 1 行)
        per_ch.append(df.mean(axis=0, skipna=True))
    per_ch = pd.DataFrame(per_ch)                       # (24, D)

    if strategy == "best_channel":
        h1 = per_ch["h1"].fillna(-np.inf)
        ch = int(h1.idxmax())
        vec = per_ch.loc[ch].add_prefix("time_")
    elif strategy == "mean_all":
        vec = per_ch.mean(axis=0, skipna=True).add_prefix("time_")
    elif strategy == "per_channel":
        # concat: time_ch{c}_{col}
        parts = []
        for c in range(len(per_ch)):
            row = per_ch.loc[c].rename(lambda x: f"time_ch{c}_{x}")
            parts.append(row)
        vec = pd.concat(parts)
    else:
        raise ValueError(strategy)
    return vec.to_dict()


# ---------------- 频域 ----------------
def _freq_one_subject(path, strategy="best_channel"):
    df = pd.read_excel(path, sheet_name=0)
    # name 列可能叫 "通道名字name" 或类似, 找一列作为通道标识
    name_col = None
    for c in df.columns:
        cs = str(c)
        if "name" in cs.lower() or "通道" in cs or cs in ("name", "channel"):
            name_col = c; break
    if name_col is None:
        # 若找不到, 就按行序视作通道 1..N
        df = df.copy()
        df["name"] = np.arange(1, len(df)+1)
        name_col = "name"
    # 数值列
    for c in FREQ_COLS:
        if c not in df.columns: df[c] = np.nan
    dfv = df[FREQ_COLS].apply(pd.to_numeric, errors="coerce")

    if strategy == "best_channel":
        E = dfv["E"].fillna(-np.inf)
        idx = int(E.idxmax())
        vec = dfv.loc[idx].add_prefix("freq_")
    elif strategy == "mean_all":
        vec = dfv.mean(axis=0, skipna=True).add_prefix("freq_")
    elif strategy == "per_channel":
        parts = []
        for i in range(len(dfv)):
            row = dfv.loc[i].rename(lambda x: f"freq_ch{i}_{x}")
            parts.append(row)
        vec = pd.concat(parts)
    else:
        raise ValueError(strategy)
    return vec.to_dict()


# ---------------- APV ----------------
def _apv_one_subject(path):
    df = pd.read_excel(path, sheet_name=0)
    if len(df) == 0:
        return {}
    row = df.iloc[0].copy()
    # 强制数值化, 加前缀
    out = {}
    for k, v in row.items():
        try:
            out[f"apv_{k}"] = float(v)
        except Exception:
            pass
    return out


def _collect_dir(d, kind, strategy):
    if not d or not os.path.isdir(d): return {}
    files = sorted(glob.glob(os.path.join(d, "*.xls*")))
    result = {}
    for i, p in enumerate(files, 1):
        sid = _sid_from_path(p)
        try:
            if kind == "time":  feats = _time_one_subject(p, strategy)
            elif kind == "freq": feats = _freq_one_subject(p, strategy)
            else:                feats = _apv_one_subject(p)
        except Exception as e:
            print(f"[skip {kind}] {sid}: {e}"); continue
        result[sid] = feats
        if i % 50 == 0 or i == len(files):
            print(f"  {kind}: {i}/{len(files)}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time_dir", default="")
    ap.add_argument("--freq_dir", default="")
    ap.add_argument("--apv_dir",  default="")
    ap.add_argument("--out_csv",  required=True)
    ap.add_argument("--time_strategy", choices=["best_channel","mean_all","per_channel"],
                    default="best_channel")
    ap.add_argument("--freq_strategy", choices=["best_channel","mean_all","per_channel"],
                    default="best_channel")
    args = ap.parse_args()

    T = _collect_dir(args.time_dir, "time", args.time_strategy)
    F = _collect_dir(args.freq_dir, "freq", args.freq_strategy)
    A = _collect_dir(args.apv_dir,  "apv",  None)

    sids = sorted(set(T) | set(F) | set(A))
    if not sids:
        raise SystemExit("三个目录都为空, 没有可导出的受试者")

    rows = []
    for sid in sids:
        row = {"sample_id": sid}
        row.update(T.get(sid, {}))
        row.update(F.get(sid, {}))
        row.update(A.get(sid, {}))
        rows.append(row)
    out = pd.DataFrame(rows).fillna(0.0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[done] {args.out_csv}  shape={out.shape}")
    print(f"[done] 时域={len(T)} 频域={len(F)} APV={len(A)} 并集={len(sids)}")


if __name__ == "__main__":
    main()
