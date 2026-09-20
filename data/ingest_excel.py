"""把原始 Excel 数据转成本框架所需 CSV.

约定:
- 每个受试者 = 一个 .xlsx, sheet 名为 "0".."23" (或按顺序前 24 个 sheet 视作通道 0..23)
- 每个 sheet 是该通道 30 秒时序 (采样率 100 Hz -> ~3000 行), 可能有多列.
- 标签/元数据在另一张总表 (Excel 或 CSV), 通过 sample_id / 文件名关联.

用法:
    python -m data.ingest_excel \\
        --raw_dir  ../data/raw_xlsx        \\
        --labels   ../data/labels.xlsx     \\
        --out_csv  ../data/raw.csv         \\
        --fs 100 --segment_sec 10 --step_sec 10 \\
        --signal_col auto                  # 或 --signal_col_index 1 或 --signal_col amplitude

标签总表列约定 (至少):
    sample_id, label_hypertension, bp_grade, tcm_syndrome,
    age, sex, sbp, dbp, bmi
其中 sample_id 应与 Excel 文件名 (去掉 .xlsx) 一致.
"""
import argparse, os, sys, glob, re
import numpy as np, pandas as pd


def _list_channel_sheets(xls, expected=24):
    names = xls.sheet_names
    # 优先按 "0".."23" 命名匹配
    numeric = {}
    for s in names:
        m = re.fullmatch(r"\s*(\d+)\s*", str(s))
        if m: numeric[int(m.group(1))] = s
    if all(i in numeric for i in range(expected)):
        return [numeric[i] for i in range(expected)]
    # 否则按顺序取前 24
    if len(names) < expected:
        raise ValueError(f"sheet 数 {len(names)} < {expected}: {names}")
    print(f"[ingest] 未找到 0..{expected-1} 命名, 按顺序取前 {expected} 个 sheet: {names[:expected]}")
    return names[:expected]


def _pick_signal(df: pd.DataFrame, mode: str = "auto",
                 col_name: str = None, col_index: int = None) -> np.ndarray:
    """从 sheet 的 DataFrame 中挑出信号列."""
    # 只保留数值列
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise ValueError(f"sheet 里没有数值列: {list(df.columns)}")
    if mode == "name" and col_name and col_name in num.columns:
        col = col_name
    elif mode == "index" and col_index is not None and col_index < num.shape[1]:
        col = num.columns[col_index]
    else:
        # auto: 选方差最大的列 (信号 vs 时间戳/索引)
        variances = num.var(axis=0).replace([np.inf, -np.inf], np.nan).dropna()
        col = variances.idxmax()
    return num[col].to_numpy(dtype=np.float32), col


def _load_one_subject(path: str, signal_mode: str, col_name: str, col_idx: int,
                      n_channels: int = 24):
    xls = pd.ExcelFile(path)
    sheets = _list_channel_sheets(xls, expected=n_channels)
    signals, chosen_cols = [], []
    for ch, sh in enumerate(sheets):
        df = pd.read_excel(xls, sheet_name=sh)
        sig, col = _pick_signal(df, signal_mode, col_name, col_idx)
        signals.append(sig); chosen_cols.append(col)
    lens = [len(s) for s in signals]
    T = min(lens)
    signals = np.stack([s[:T] for s in signals], axis=0)   # (C, T)
    return signals, chosen_cols, T


def _segment(sig: np.ndarray, fs: int, seg_sec: float, step_sec: float):
    L = int(round(seg_sec*fs))
    S = int(round(step_sec*fs))
    T = sig.shape[1]
    segs, offsets = [], []
    off = 0
    while off + L <= T:
        segs.append(sig[:, off:off+L]); offsets.append(off/fs); off += S
    return segs, offsets


def _flatten_row(sample_id: str, segment_idx: int, seg: np.ndarray,
                 meta_row: dict, n_ch: int, seq_len: int) -> dict:
    row = {"sample_id": f"{sample_id}#seg{segment_idx}"}
    row.update(meta_row or {})
    # 手工特征交给下游 pipeline 计算; 这里先给零占位或者可留空, loader 端不强依赖
    for name in ["h1","h2","h3","h4","h5","t1","t2","t3","t4","t5",
                 "w1_t","w2_t","h3_h1","h4_h1",
                 "E","TSEn","ApEn",
                 "sAPVt1","sAPVt4","sAPVt5","APVh1","APVh3"]:
        row.setdefault(name, 0.0)
    for c in range(n_ch):
        for t in range(seq_len):
            row[f"wave_{c}_{t}"] = float(seg[c, t])
    return row


def load_labels(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        lab = pd.read_excel(path)
    else:
        lab = pd.read_csv(path)
    if "sample_id" not in lab.columns:
        raise ValueError("labels 表缺列 sample_id")
    lab["sample_id"] = lab["sample_id"].astype(str)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir",  required=True, help="存放 .xlsx 的目录")
    ap.add_argument("--labels",   required=False, default="", help="标签总表 (可选; 缺则只出波形)")
    ap.add_argument("--out_csv",  required=True)
    ap.add_argument("--fs", type=int, default=100, help="采样率 Hz")
    ap.add_argument("--segment_sec", type=float, default=10.0)
    ap.add_argument("--step_sec",    type=float, default=10.0, help="片段之间的步长")
    ap.add_argument("--n_channels",  type=int, default=24)
    ap.add_argument("--signal_col",       default="auto",
                    help="auto / <列名> ; 优先级低于 signal_col_index")
    ap.add_argument("--signal_col_index", type=int, default=-1,
                    help="按列序号取; -1 表示不使用")
    args = ap.parse_args()

    mode = "index" if args.signal_col_index >= 0 else \
           ("name" if args.signal_col != "auto" else "auto")

    files = sorted(glob.glob(os.path.join(args.raw_dir, "*.xls*")))
    if not files:
        raise SystemExit(f"未在 {args.raw_dir} 找到 xlsx 文件")
    labels = load_labels(args.labels) if args.labels else None

    seq_len = int(round(args.segment_sec * args.fs))
    print(f"[ingest] {len(files)} 个 Excel, 每个切 {args.segment_sec}s "
          f"(步长 {args.step_sec}s) -> 每片 {seq_len} 点")

    rows, log = [], []
    for i, fp in enumerate(files, 1):
        sid = os.path.splitext(os.path.basename(fp))[0]
        try:
            sig, cols_used, T = _load_one_subject(
                fp, mode, args.signal_col, args.signal_col_index, args.n_channels)
        except Exception as e:
            print(f"[skip] {fp}: {e}"); continue
        segs, offs = _segment(sig, args.fs, args.segment_sec, args.step_sec)
        if not segs:
            print(f"[skip] {sid}: 时序过短 T={T}"); continue

        meta = {}
        if labels is not None:
            hit = labels[labels["sample_id"] == sid]
            if len(hit) == 0:
                print(f"[warn] labels 里找不到 {sid}, 该文件跳过")
                continue
            meta = hit.iloc[0].to_dict()

        for k, seg in enumerate(segs):
            rows.append(_flatten_row(sid, k, seg, meta, args.n_channels, seq_len))
        log.append((sid, len(segs), cols_used[0]))
        if i % 20 == 0 or i == len(files):
            print(f"  processed {i}/{len(files)}  (last id={sid}, segs={len(segs)})")

    if not rows:
        raise SystemExit("没有可导出的行")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[done] wrote {args.out_csv}  shape={df.shape}")
    print(f"[done] 各文件所用信号列(第一通道示例): 前 5 = {log[:5]}")


if __name__ == "__main__":
    main()
