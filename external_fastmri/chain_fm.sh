#!/bin/bash
# fastMRI chain on Ramona (CUDA). Waits for chain_cuda.sh (gen-2 seeds) and the fm3 copy; builds ~/rsna-knee-fm;
# pseudo-labels the 5,046 side-known fastMRI studies with the 8 CUDA gen-2 students; trains external students.
cd ~/rsna-knee
until grep -q CHAIN-CUDA-DONE chain_cuda.log 2>/dev/null && grep -q FM3-PULL-DONE fm3_pull.log 2>/dev/null; do sleep 120; done
echo "start $(date)"; venv/bin/python build_fmdata.py || exit 1
DST=~/rsna-knee-fm; cd $DST
mkdir -p npy_fmx; for u in $(ls ~/rsna-knee/fm3); do [ -L npy_fmx/$u ] || ln -s ~/rsna-knee/fm3/$u npy_fmx/$u; done
for u in $(awk -F, '$2==1{print $1}' ~/rsna-knee/folds.csv | head -12); do [ -L npy_fmx/$u ] || ln -s ~/rsna-knee/npy3/$u npy_fmx/$u; done
X="--data $DST --npy npy_fmx --res 320 --bs 3 --workers 8 --fold 0"
for s in 0 1 2 3; do
  [ -f runs/fmx_c$s"_feats.npz" ] || ~/rsna-knee/venv/bin/python ~/rsna-knee/train_mil2.py --arch convnext_tiny.in12k_ft_in1k --weights ~/rsna-knee/convnext_tiny_in12k_ft_in1k.safetensors --drop_path 0.1 $X --export_feats ~/rsna-knee/runs/cnxt_gen2_s$s/ema0.9995.pt --tag fmx_c$s 2>&1 | grep -E "export" | tail -1
  [ -f runs/fmx_e$s"_feats.npz" ] || ~/rsna-knee/venv/bin/python ~/rsna-knee/train_mil2.py --arch efficientnet_b0 --weights ~/rsna-knee/effb0_pretrained.pth $X --export_feats ~/rsna-knee/runs/e_gen2_s$s/ema0.9995.pt --tag fmx_e$s 2>&1 | grep -E "export" | tail -1
done
~/rsna-knee/venv/bin/python - <<'PY'
import numpy as np, pandas as pd
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
def ld(t):
    d = np.load(f"runs/{t}_feats.npz", allow_pickle=True)
    return pd.DataFrame(1 / (1 + np.exp(-d["logit"].astype(np.float64))), index=list(d["ids"]), columns=L)
C = sum(ld(f"fmx_c{s}") for s in range(4)) / 4; E = sum(ld(f"fmx_e{s}") for s in range(4)) / 4
T = (6 / 11) * C + (5 / 11) * E
fm = [u for u in T.index if u.startswith("fm_")]; T = T.loc[fm]
g2 = pd.read_csv("/workspace/rsna-knee/relabel/student2gen_opus50.csv").set_index("StudyInstanceUID")[L]
out = pd.concat([g2, T]); out.index.name = "StudyInstanceUID"; out.to_csv("fm_gen2_targets.csv")
print("pseudo-labelled fastMRI studies:", len(T), "| mean target fastMRI:", T.mean().round(3).to_dict())
print("mean target ours (gen-2):", g2.mean().round(3).to_dict())
PY
echo "PSEUDO-DONE $(date)"
G="--data $DST --npy npy_fm --res 320 --bs 3 --workers 8 --epochs 12 --warmup_ep 1 --ema --all_data --all_sides --ext_labels fm_gen2_targets.csv --ext_weight 1.0"
for s in 0 1; do [ -f runs/cnxt_fm_s$s/preds.npz ] || ~/rsna-knee/venv/bin/python ~/rsna-knee/train_mil2.py --arch convnext_tiny.in12k_ft_in1k --weights ~/rsna-knee/convnext_tiny_in12k_ft_in1k.safetensors --lr 3e-5 --drop_path 0.1 $G --seed $s --tag cnxt_fm_s$s > ~/rsna-knee/train_mil_cnxt_fm_s$s.log 2>&1; echo "cnxt_fm_s$s done $(date)"; done
[ -f runs/e_fm_s0/preds.npz ] || ~/rsna-knee/venv/bin/python ~/rsna-knee/train_mil2.py --arch efficientnet_b0 --weights ~/rsna-knee/effb0_pretrained.pth --lr 2e-4 $G --seed 0 --tag e_fm_s0 > ~/rsna-knee/train_mil_e_fm_s0.log 2>&1
echo "CHAIN-FM-DONE $(date)"
