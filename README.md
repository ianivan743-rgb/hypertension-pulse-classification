# 高血压 24 通道脉搏波分类框架

面向硕士论文"模型"章节重写。原论文用 ImageNet 预训练 ViT + 手工特征 RF，本框架改为：
**① 领域内 SSL 预训练（1D-MAE） → ② 微调 + 多模态融合（波形 + 手工特征 + 人口学） → ③ 传统 ML 基线（RF/XGB/SVM/LR）**。

三类下游任务：
- `binary`   高血压 vs 对照
- `grade`    血压等级（4~5 类）
- `syndrome` 中医证型

---

## 目录结构

```
code/
  configs/default.yaml     全局配置
  data/
    ingest_excel.py        原始波形 xlsx → waves.csv（10s 切段）
    ingest_features.py     时/频/APV 特征 xlsx → features.csv
    build_dataset.py       features + waves + labels → raw.csv
    loader.py              读取 raw.csv, 拆分 X/F/M/y, 生成分组
  models/                  ssl_mae / encoder1d / fusion
  train/                   pretrain_ssl / finetune / train_ml
  utils/                   cv / metrics / io
```

---

## 数据约定

### A. 原始波形 xlsx（可选，用于 SSL / 深度模型）
- 1 个 xlsx = 1 位受试者
- 24 个 sheet = 24 通道（sheet0..sheet23）
- 每 sheet 约 3000 行 = 30 s @ 100 Hz
- `ingest_excel.py` 自动切成 3×10s 片段，`sample_id` 追加 `#seg0/1/2`

### B. 特征 xlsx（3 类，均为 1 xlsx = 1 位受试者）

| 类型 | sheet | 每 sheet 结构 | 列 |
|---|---|---|---|
| **time** | 24 个（对应通道） | 每 sheet 若干行（异常值已剔除） | `t1..t5 h1..h5 w31 w51 w31_t w51_t h1_t1 h3_h1 h4_h1 As Ad t h1_OP..h5_OP Beg`（26 列） |
| **freq** | 1 个 | 24 行 = 24 通道 | `name(1..24) ApEn E TSEn MSF RMSF frequency_variance frequency_std_dev 频谱峰值_频率_1..5 频谱峰值_幅度_1..5` |
| **APV**  | 1 个 | 1 行 | `APV_avrage APV_h1..h5 sAPVt1 sAPVt4 sAPVt5 sAPVt1_sAPVt sAPVt4_sAPVt5` + MiddleRing / InnerRing 变体 |

聚合策略（`--time_strategy` / `--freq_strategy`）：
- **`best_channel`**（默认，对齐原论文思路）：time 取 h1 最大的通道；freq 取 E 最大的通道
- **`mean_all`**：24 通道均值
- **`per_channel`**：24 通道全部拼接（列名前缀 `ch{i}_`）

### C. 标签总表
Excel 或 CSV，通过 `sample_id`（不含 `#segX` 后缀）与 xlsx 文件名关联。至少包含：
```
sample_id, label_hypertension, bp_grade, tcm_syndrome, age, sex, sbp, dbp, bmi
```

---

## 端到端流程

```bash
cd code
pip install -r requirements.txt

# ---- Step 1a: 特征 xlsx → features.csv ----
python -m data.ingest_features \
  --time_dir ../data/time_xlsx \
  --freq_dir ../data/freq_xlsx \
  --apv_dir  ../data/apv_xlsx \
  --out_csv  ../data/features.csv \
  --time_strategy best_channel \
  --freq_strategy best_channel

# ---- Step 1b: 波形 xlsx → waves.csv (可选) ----
python -m data.ingest_excel \
  --raw_dir ../data/raw_xlsx \
  --labels  ../data/labels.xlsx \
  --out_csv ../data/waves.csv \
  --fs 100 --segment_sec 10 --step_sec 10

# ---- Step 2: 合并 → raw.csv ----
python -m data.build_dataset \
  --features ../data/features.csv \
  --waves    ../data/waves.csv \
  --labels   ../data/labels.xlsx \
  --out_csv  ../data/raw.csv

# 只有特征、没有波形也可直接：
python -m data.build_dataset \
  --features ../data/features.csv \
  --labels   ../data/labels.xlsx \
  --out_csv  ../data/raw.csv

# ---- Step 3: 训练 ----
# (A) 传统 ML 基线
python -m train.train_ml --task binary
python -m train.train_ml --task grade
python -m train.train_ml --task syndrome

# (B) 领域 SSL 预训练（需要 waves 列）
python -m train.pretrain_ssl

# (C) 微调 + 多模态融合
python -m train.finetune --task binary
python -m train.finetune --task grade
python -m train.finetune --task syndrome
```

---

## 关键设计

- **防泄露**：`build_dataset` 保留原始 `sample_id` 及 `#seg` 后缀；`loader.py` 用 `sample_id.split("#")[0]` 生成 group，配合 `StratifiedGroupKFold`，保证同一受试者的 3 个片段不跨 train/val。
- **类别不平衡**：`class_weight` / SMOTE（每折内做）/ FocalLoss，可在配置切换。
- **两条轨道**：
  - **仅特征**（无波形）：loader 自动补零占位，可直接跑传统 ML。
  - **特征 + 波形**：SSL + 微调 + 融合三阶段流程完整可用。
- **可复现**：所有随机种子集中在 `configs/default.yaml.seed`。

