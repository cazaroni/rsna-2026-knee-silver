#!/bin/bash
# Wave 2 on Ramona (CUDA). (1) cnxt_fmbal_s0: full fastMRI pool, same targets/steps as cnxt_fmall_s0, domain-balanced sampling.
# (2) anchored second fastMRI round: the fastMRI students re-predict fastMRI; fastMRI targets := 0.5 original teacher + 0.5 fastMRI
#     students; competition targets unchanged; then cnxt_fmr2_s0 (uniform sampling, same recipe).
DST=~/rsna-knee-fm; cd $DST
G="--data $DST --npy npy_fm --res 320 --bs 3 --workers 8 --epochs 12 --warmup_ep 1 --ema --all_data --all_sides --ext_weight 1.0"
C="--arch convnext_tiny.in12k_ft_in1k --weights $HOME/rsna-knee/convnext_tiny_in12k_ft_in1k.safetensors --lr 3e-5 --drop_path 0.1"
[ -f runs/cnxt_fmbal_s0/preds.npz ] || ~/rsna-knee/venv/bin/python ~/rsna-knee/train_mil2.py $C $G --ext_labels fm_all_targets.csv --domain_balance --seed 0 --tag cnxt_fmbal_s0 > ~/rsna-knee/train_mil_cnxt_fmbal_s0.log 2>&1
echo "FMBAL-DONE $(date)"
mkdir -p npy_fmx3; for u in $(ls npy_fm | grep "^fm_"); do [ -L npy_fmx3/$u ] || ln -s $(readlink -f npy_fm/$u) npy_fmx3/$u; done
for u in $(awk -F, '$2==1{print $1}' ~/rsna-knee/folds.csv | head -12); do [ -L npy_fmx3/$u ] || ln -s ~/rsna-knee/npy3/$u npy_fmx3/$u; done
X="--data $DST --npy npy_fmx3 --res 320 --bs 3 --workers 8 --fold 0"
for t in cnxt_fm_s0 cnxt_fm_s1 cnxt_fmall_s0; do [ -f runs/r2x_$t"_feats.npz" ] || ~/rsna-knee/venv/bin/python ~/rsna-knee/train_mil2.py $C $X --export_feats $DST/runs/$t/ema0.9995.pt --tag r2x_$t 2>&1 | grep export | tail -1; done
[ -f runs/r2x_e_fm_s0_feats.npz ] || ~/rsna-knee/venv/bin/python ~/rsna-knee/train_mil2.py --arch efficientnet_b0 --weights $HOME/rsna-knee/effb0_pretrained.pth $X --export_feats $DST/runs/e_fm_s0/ema0.9995.pt --tag r2x_e_fm_s0 2>&1 | grep export | tail -1
~/rsna-knee/venv/bin/python - <<'PY'
import numpy as np, pandas as pd
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
def ld(t):
    d = np.load(f"runs/{t}_feats.npz", allow_pickle=True)
    return pd.DataFrame(1 / (1 + np.exp(-d["logit"].astype(np.float64))), index=list(d["ids"]), columns=L)
S = (6 / 11) * (ld("r2x_cnxt_fm_s0") + ld("r2x_cnxt_fm_s1") + ld("r2x_cnxt_fmall_s0")) / 3 + (5 / 11) * ld("r2x_e_fm_s0")
cur = pd.read_csv("fm_all_targets.csv").set_index("StudyInstanceUID")[L]
fm = [u for u in cur.index if u.startswith("fm_") and u in S.index]
nxt = cur.copy(); nxt.loc[fm] = 0.5 * cur.loc[fm] + 0.5 * S.loc[fm]
nxt.to_csv("fm_r2_targets.csv"); print("round-2 targets: fastMRI rows updated", len(fm), "| mean |change|", float((nxt.loc[fm] - cur.loc[fm]).abs().mean().mean()))
PY
[ -f runs/cnxt_fmr2_s0/preds.npz ] || ~/rsna-knee/venv/bin/python ~/rsna-knee/train_mil2.py $C $G --ext_labels fm_r2_targets.csv --seed 0 --tag cnxt_fmr2_s0 > ~/rsna-knee/train_mil_cnxt_fmr2_s0.log 2>&1
echo "WAVE2-DONE $(date)"
