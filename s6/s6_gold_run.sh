#!/bin/bash
# score a model folder on gold-58 with compact_infer (CPU, 8 threads), then report macro AUC (11 findings and 12)
M=$1; TAG=$2; cd ~/s6data
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
rm -rf gold/out_$TAG gold/tmp_$TAG
nice -n 10 venv312/bin/python -u s6_infer.py --data gold --split train --code code/heliosli/cloud_code --models $M --arm E_prior \
  --backbone convnext_small --size 320 --crop 288 --views v4 --slices 3 --phys-mm 3.5 --eval-stride 1 \
  --anchor-weights ~/rsna-knee/ext_models/helios/anatomy.pt --amp fp16 --chunk 192 --workers 2 --item-timeout 3600 \
  --output gold/out_$TAG --tmp gold/tmp_$TAG 2>&1 | grep -v Warning | tail -5
venv312/bin/python - $TAG <<'PY'
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
t = sys.argv[1]
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
p = pd.read_csv(f"gold/out_{t}/predictions.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
y = pd.read_csv("gold/train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[p.index, L].astype(float) >= 1
a = {l: roc_auc_score(y[l], p[l]) for l in L if 0 < y[l].sum() < len(y)}
print(f"GOLD {t}: n={len(p)} macro12 {np.mean(list(a.values())):.4f} | macro11 (no Syn) {np.mean([v for k, v in a.items() if k != 'Synovitis']):.4f}")
print({k: round(v, 3) for k, v in a.items()})
PY
