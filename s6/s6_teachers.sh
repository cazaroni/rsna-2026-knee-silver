#!/bin/bash
# 3-fold S6 teachers on our gen-2 targets (same recipe as S6-ours); each fold writes out-of-fold logits for its held-out third.
cd ~/s6data/code
venv=../venv312/bin/python
$venv - <<'PY'
import json, numpy as np
t = json.load(open("data/folds_v0.json"))["folds"]
uids = sorted(u for u, f in t.items() if f >= 0); rng = np.random.default_rng(20261017); perm = rng.permutation(len(uids))
f3 = {u: int(i % 3) for i, u in zip(range(len(uids)), [uids[j] for j in perm])}
f3.update({u: -1 for u, f in t.items() if f < 0})
json.dump({"folds": f3}, open("data/folds3.json", "w")); import collections; print("folds3", collections.Counter(f3.values()))
PY
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for k in 0 1 2; do
  A=(--pool data/pool_ok.json --folds-json data/folds3.json --targets data/targets_ours.csv --out ../runs_t3_f$k --arms E_prior
     --backbone convnext_small --init $HOME/rsna-knee/convnext_small_fb_in22k_ft_in1k.safetensors --seeds 1 --seed-base 42 --folds 3 --only-folds $k
     --epochs 16 --batch 2 --train-windows 64 --size 320 --views v4 --crop 288 --slices 3 --phys-mm 3.5
     --spatial-attn --lr 1e-4 --head-lr 1e-3 --wd 0.05 --aug crop --train-sampling uniform --ema 0
     --workers 16 --loader-timeout 600 --max-fail-frac 0.005 --log-every 200 --work-tmp /tmp/s6f0_work
     --resume --i-have-user-authorisation-to-finetune)
  for a in 1 2 3; do
    $venv -u train_s6f0_all.py "${A[@]}" 2>&1 | grep --line-buffered -v Warning >> ../teacher_f$k.log
    [ ${PIPESTATUS[0]} -eq 0 ] && break; echo "fold $k exit, resume attempt $a $(date)"; sleep 60
  done
  ls ../runs_t3_f$k/*oof* >/dev/null 2>&1 && echo "TEACHER-F$k-DONE $(date)" || { echo "TEACHER-F$k-NO-OOF $(date)"; ls ../runs_t3_f$k; }
done
echo "TEACHERS-DONE $(date)"
