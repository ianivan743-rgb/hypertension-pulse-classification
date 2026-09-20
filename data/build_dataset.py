"""把 (波形 CSV) + (特征 CSV) + (标签总表) 合并成最终训练 CSV.

三个输入至少要有 features 或 waves 之一; labels 必需.

用法:
    python -m data.build_dataset \\
        --features ../data/features.csv        \\
        --waves    ../data/waves.csv           \\
        --labels   ../data/labels.xlsx         \\
        --out_csv  ../data/raw.csv
"""
import argparse, os
import numpy as np, pandas as pd


def _read_table(p):
    if p.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(p)
    return pd.read_csv(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="")
    ap.add_argument("--waves",    default="")
    ap.add_argument("--labels",   required=True)
    ap.add_argument("--out_csv",  required=True)
    ap.add_argument("--id_col",   default="sample_id")
    args = ap.parse_args()

    parts = []
    if args.features and os.path.exists(args.features):
        parts.append(_read_table(args.features))
    if args.waves and os.path.exists(args.waves):
        parts.append(_read_table(args.waves))
    if not parts:
        raise SystemExit("必须提供 --features 或 --waves 至少一个")

    labels = _read_table(args.labels)
    labels[args.id_col] = labels[args.id_col].astype(str)

    # 合并特征 + 波形 (按 sample_id; 波形表可能带 #seg 后缀)
    if len(parts) == 1:
        df = parts[0].copy()
    else:
        # 特征表(1 行/受试者) join 波形表(可能 3 行/受试者)
        feat, wave = parts
        feat[args.id_col] = feat[args.id_col].astype(str)
        wave[args.id_col] = wave[args.id_col].astype(str)
        wave["_base_id"] = wave[args.id_col].str.split("#", n=1).str[0]
        df = wave.merge(feat, left_on="_base_id", right_on=args.id_col,
                        how="left", suffixes=("", "_feat"))
        df = df.drop(columns=[c for c in df.columns if c.endswith("_feat")] + ["_base_id"])

    # 合并标签
    df["_base_id"] = df[args.id_col].astype(str).str.split("#", n=1).str[0]
    labels_slim = labels.rename(columns={args.id_col: "_base_id"})
    df = df.merge(labels_slim, on="_base_id", how="left", suffixes=("", "_lab"))
    df = df.drop(columns=["_base_id"])

    # 落盘
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[done] {args.out_csv}  shape={df.shape}")
    n_lab = df["label_hypertension"].notna().sum() if "label_hypertension" in df.columns else 0
    print(f"[done] 含标签样本: {n_lab}/{len(df)}")


if __name__ == "__main__":
    main()
