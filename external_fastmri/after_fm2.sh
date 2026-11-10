#!/bin/bash
# Full fastMRI pool: 5,046 known-side + 4,351 confidently classified studies. Waits for chain_fm (EfficientNet fastMRI student) and
# the extra copy; extends ~/rsna-knee-fm; pseudo-labels the extra studies with the same 8 CUDA gen-2 students; trains cnxt_fmall_s0.
cd ~/rsna-knee
until grep -q CHAIN-FM-DONE chain_fm.log 2>/dev/null && grep -q FM3-EXTRA-DONE fm3_extra.log 2>/dev/null; do sleep 120; done
echo "start $(date)"
venv/bin/python - <<'PY'
import pandas as pd
a = pd.read_csv("fm_known_side.csv"); b = pd.read_csv("fm_extra_side.csv")
pd.concat([a, b], ignore_index=True).drop_duplicates("study").to_csv("fm_all_side.csv", index=False); print("fm_all_side:", len(a) + len(b))
PY
sed -e 's|fm_known_side.csv|fm_all_side.csv|' build_fmdata.py > build_fmdata_all.py && venv/bin/python build_fmdata_all.py || exit 1
DST=~/rsna-knee-fm; cd $DST
mkdir -p npy_fmx2; for u in $(tail -n +2 ~/rsna-knee/fm_extra_side.csv | cut -d, -f1); do [ -d ~/rsna-knee/fm3/$u ] && [ ! -L npy_fmx2/$u ] && ln -s ~/rsna-knee/fm3/$u npy_fmx2/$u; done
for u in $(awk -F, '$2==1{print $1}' ~/rsna-knee/folds.csv | head -12); do [ -L npy_fmx2/$u ] || ln -s ~/rsna-knee/npy3/$u npy_fmx2/$u; done
X="--data $DST --npy npy_fmx2 --res 320 --bs 3 --workers 8 --fold 0"
for s in 0 1 2 3; do
  [ -f runs/fmx2_c$s"_feats.npz" ] || ~/rsna-knee/venv/bin/python ~/rsna-knee/train_mil2.py --arch convnext_tiny.in12k_ft_in1k --weights ~/rsna-knee/convnext_tiny_in12k_ft_in1k.safetensors --drop_path 0.1 $X --export_feats ~/rsna-knee/runs/cnxt_gen2_s$s/ema0.9995.pt --tag fmx2_c$s 2>&1 | grep export | tail -1
  [ -f runs/fmx2_e$s"_feats.npz" ] || ~/rsna-knee/venv/bin/python ~/rsna-knee/train_mil2.py --arch efficientnet_b0 --weights ~/rsna-knee/effb0_pretrained.pth $X --export_feats ~/rsna-knee/runs/e_gen2_s$s/ema0.9995.pt --tag fmx2_e$s 2>&1 | grep export | tail -1
done
~/rsna-knee/venv/bin/python - <<'PY'
import numpy as np, pandas as pd
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
def ld(t):
    d = np.load(f"runs/{t}_feats.npz", allow_pickle=True)
    return pd.DataFrame(1 / (1 + np.exp(-d["logit"].astype(np.float64))), index=list(d["ids"]), columns=L)
T = (6 / 11) * (sum(ld(f"fmx2_c{s}") for s in range(4)) / 4) + (5 / 11) * (sum(ld(f"fmx2_e{s}") for s in range(4)) / 4)
T = T.loc[[u for u in T.index if u.startswith("fm_")]]
old = pd.read_csv("fm_gen2_targets.csv").set_index("StudyInstanceUID")[L]
out = pd.concat([old, T.loc[[u for u in T.index if u not in old.index]]]); out.index.name = "StudyInstanceUID"
out.to_csv("fm_all_targets.csv"); print("targets rows:", len(out), "| new fastMRI pseudo-labelled:", len(T))
PY
echo "PSEUDO2-DONE $(date)"
G="--data $DST --npy npy_fm --res 320 --bs 3 --workers 8 --epochs 12 --warmup_ep 1 --ema --all_data --all_sides --ext_labels fm_all_targets.csv --ext_weight 1.0"
[ -f runs/cnxt_fmall_s0/preds.npz ] || ~/rsna-knee/venv/bin/python ~/rsna-knee/train_mil2.py --arch convnext_tiny.in12k_ft_in1k --weights ~/rsna-knee/convnext_tiny_in12k_ft_in1k.safetensors --lr 3e-5 --drop_path 0.1 $G --seed 0 --tag cnxt_fmall_s0 > ~/rsna-knee/train_mil_cnxt_fmall_s0.log 2>&1
echo "AFTER-FM2-DONE $(date)"
