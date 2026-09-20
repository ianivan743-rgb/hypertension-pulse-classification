#!/usr/bin/env bash
# 冒烟测试: 用合成 200 样本, 短 epoch, 走完全 pipeline
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PYTHONPATH"
mkdir -p ../outputs/models ../outputs/logs

echo "==== [1/4] 传统 ML 基线 (binary/grade/syndrome) ===="
python -m train.train_ml --task binary
python -m train.train_ml --task grade
python -m train.train_ml --task syndrome

echo "==== [2/4] SSL 预训练 (短 epoch, 冒烟) ===="
python - <<'PY'
import yaml, sys
p = "configs/default.yaml"
c = yaml.safe_load(open(p))
c["train"]["epochs_ssl"] = 3
c["train"]["epochs_ft"]  = 3
yaml.safe_dump(c, open(p,"w"), sort_keys=False, allow_unicode=True)
print("cfg patched")
PY
python -m train.pretrain_ssl

echo "==== [3/4] 微调 (syndrome, 加载 ssl.pt) ===="
python -m train.finetune --task syndrome --pretrained ../outputs/models/ssl.pt

echo "==== [4/4] 多模态融合 (syndrome) ===="
python -m train.train_fusion --task syndrome --pretrained ../outputs/models/ssl.pt

echo "==== ALL DONE ===="
