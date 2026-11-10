#!/bin/bash
# S6 generation-3 + fastMRI chain (all automatic; every step checks the previous one).
set -o pipefail
cd ~/s6data; PY=$HOME/s6data/venv312/bin/python
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ---- 1. our-study targets: 0.5 Opus + 0.5 S6-teacher out-of-fold (3 folds) ----
$PY - <<'PY' || { echo "STEP1-FAILED"; exit 1; }
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
f3 = json.load(open("code/data/folds3.json"))["folds"]; parts = []
for k in range(3):
    ids = json.load(open(f"runs_t3_f{k}/pool_uids.json")); z = np.load(f"runs_t3_f{k}/partial_oof_E_prior_s42.npy")
    ids = ids if isinstance(ids, list) else ids["uids"]
    m = np.array([f3.get(u, -1) == k for u in ids]); assert z.shape == (len(ids), 12), z.shape
    parts.append(pd.DataFrame(1 / (1 + np.exp(-z[m].astype(np.float64))), index=np.array(ids)[m], columns=L))
oof = pd.concat(parts); assert oof.index.is_unique and np.isfinite(oof.values).all()
op = pd.read_csv("/workspace/rsna-knee/relabel/opus_p.csv").set_index("StudyInstanceUID")[L]
t = pd.read_csv("code/data/targets_ours.csv", dtype={"StudyInstanceUID": str}); ng = t.is_gold == 0
u = t.loc[ng, "StudyInstanceUID"]; miss = (~u.isin(oof.index)).sum(); print("OOF studies", len(oof), "| non-gold without OOF", int(miss))
for l in L:
    t.loc[ng, f"{l}|prior"] = (0.5 * op.loc[u, l].values + 0.5 * oof.reindex(u)[l].fillna(op.loc[u, l]).values).clip(0, 1)
t.to_csv("code/data/targets_s6gen.csv", index=False)
o = op.loc[oof.index]
print("teacher OOF AUC vs Opus>=0.5:", round(np.mean([roc_auc_score(o[l] >= 0.5, oof[l]) for l in L]), 4))
PY
echo "STEP1-DONE $(date)"
# ---- 2. fastMRI pseudo-labels from the 3 S6 teachers (compact_infer, eval stride 3) ----
R=~/s6data/fm9397; rm -rf $R; mkdir -p $R/train_series fm_teachers
$PY - <<'PY' || { echo "STEP2-PREP-FAILED"; exit 1; }
import pandas as pd
keep = set(pd.read_csv("/workspace/rsna-knee/fm_all_side.csv").study)
for f in ("train.csv", "train_series.csv"):
    d = pd.read_csv(f"/workspace/s6data/fmdata/{f}", dtype=str); d[d.StudyInstanceUID.isin(keep)].to_csv(f"/workspace/s6data/fm9397/{f}", index=False)
print("fm9397 studies", pd.read_csv("/workspace/s6data/fm9397/train.csv").shape[0])
PY
for u in $(tail -n +2 $R/train.csv); do cp -rs ~/s6data/fmdata/train_series/$u $R/train_series/$u; done
for k in 0 1 2; do cp runs_t3_f$k/models/E_prior_s42_f$k.pt fm_teachers/; done
rm -rf fm_pl_out fm_pl_tmp
$PY -u s6_infer.py --data $R --split train --code code/heliosli/cloud_code --models fm_teachers --arm E_prior \
  --backbone convnext_small --size 320 --crop 288 --views v4 --slices 3 --phys-mm 3.5 --eval-stride 3 \
  --anchor-weights ~/rsna-knee/ext_models/helios/anatomy.pt --amp fp16 --chunk 192 --workers 8 --item-timeout 1800 \
  --output fm_pl_out --tmp fm_pl_tmp 2>&1 | grep -v Warning | grep -E "WORKER_DONE|done\": [0-9]*000," > fm_pl.log
[ -f fm_pl_out/predictions.csv ] || { echo "STEP2-FAILED"; tail -5 fm_pl.log; exit 1; }
echo "STEP2-DONE $(date) $(tail -1 fm_pl.log | cut -c1-120)"
# ---- 3. fastMRI anchors for training (GPU) ----
cd code_fm
$PY - <<'PY'
import json, pandas as pd, os
ids = list(pd.read_csv("/workspace/s6data/fm9397/train.csv").StudyInstanceUID)
bad = set(json.load(open("/workspace/s6data/fm_pl_out/receipt.json")).get("fallback", {}) or {})
ids = [u for u in ids if u not in bad]; json.dump(ids, open("data/order_fm9397.json", "w")); print("fm anchors to build", len(ids), "| pseudo-label fallbacks excluded", len(bad))
PY
$PY -u precompute_anchors.py --order data/order_fm9397.json --out anchors --tmp /tmp/s6_fm_anch --workers 12 2>&1 | grep -v Warning | grep -E "SUMMARY" | tail -1
cd ~/s6data; echo "STEP3-DONE $(date)"
# ---- 4. combined root (ours + fastMRI), pool, folds, targets, anchors ----
C=~/s6data/s6comb; rm -rf $C code_comb; mkdir -p $C/train_series
$PY - <<'PY' || { echo "STEP4-FAILED"; exit 1; }
import json, numpy as np, pandas as pd, shutil, os
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
B = "/workspace/s6data"
okfm = [json.loads(l)["uid"] for l in open(f"{B}/code_fm/anchors/manifest.jsonl") if json.loads(l).get("error") is None]
order = set(json.load(open(f"{B}/code_fm/data/order_fm9397.json"))); okfm = sorted(set(okfm) & order)
pl = pd.read_csv(f"{B}/fm_pl_out/predictions.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[okfm, L]
assert np.isfinite(pl.values).all()
pool = json.load(open(f"{B}/code/data/pool_ok.json")); json.dump(pool + okfm, open(f"{B}/pool_comb.json", "w"))
f5 = json.load(open(f"{B}/code/data/folds_v0.json"))["folds"]; f5.update({u: i % 5 for i, u in enumerate(okfm)})
json.dump({"folds": f5}, open(f"{B}/folds_comb.json", "w"))
t = pd.read_csv(f"{B}/code/data/targets_s6gen.csv", dtype={"StudyInstanceUID": str})
fm = pd.DataFrame({"StudyInstanceUID": okfm, "is_gold": 0, "review_holdout": 0, "fold": [f5[u] for u in okfm]})
for l in L: fm[f"{l}|prior"] = pl[l].values.clip(0.001, 0.999)
for l in L: fm[f"{l}|strict"] = np.nan
pd.concat([t, fm[t.columns]], ignore_index=True).to_csv(f"{B}/targets_comb.csv", index=False)
for f in ("train.csv", "train_series.csv"):
    a = pd.read_csv(f"{B}/{f}", dtype=str); b = pd.read_csv(f"{B}/fm9397/{f}", dtype=str); b = b[b.StudyInstanceUID.isin(okfm)]
    pd.concat([a, b], ignore_index=True).to_csv(f"{B}/s6comb/{f}", index=False)
open(f"{B}/s6comb/expect.txt", "w").write(str(len(pool) - 58 + len(okfm)))
print("pool", len(pool), "+ fm", len(okfm), "| expect-train", len(pool) - 58 + len(okfm), "| fm target means", pl.mean().round(3).values)
PY
for u in $(ls ~/s6data/train_series); do cp -rs ~/s6data/train_series/$u $C/train_series/$u; done
for u in $(tail -n +2 $C/train.csv | cut -d, -f1 | grep "^fm_"); do cp -rs ~/s6data/fm9397/train_series/$u $C/train_series/$u 2>/dev/null; done
cp -r code code_comb && rm -rf code_comb/anchors_smoke && cp -n code_fm/anchors/*.json code_comb/anchors/ 2>/dev/null
sed -i 's|^DATA = .*|DATA = "/workspace/s6data/s6comb"|' code_comb/s6common.py
echo "STEP4-DONE $(date) | studies in root $(ls $C/train_series | wc -l) | anchors $(ls code_comb/anchors | wc -l)"
# ---- 5. check targets, then train the generation-3 S6 student on ours + fastMRI ----
cd code_comb; N=$(cat ../s6comb/expect.txt)
A=(--pool ../pool_comb.json --folds-json ../folds_comb.json --targets ../targets_comb.csv --out ../runs_comb --arms E_prior
   --backbone convnext_small --init $HOME/rsna-knee/convnext_small_fb_in22k_ft_in1k.safetensors --seeds 1 --seed-base 42 --folds 5 --only-folds 0
   --expect-train $N --expect-holdout 0 --all-data
   --epochs 16 --batch 2 --train-windows 64 --size 320 --views v4 --crop 288 --slices 3 --phys-mm 3.5
   --spatial-attn --lr 1e-4 --head-lr 1e-3 --wd 0.05 --aug crop --train-sampling uniform --ema 0
   --workers 16 --loader-timeout 600 --max-fail-frac 0.005 --log-every 200 --work-tmp /tmp/s6f0_work
   --resume --no-final-predict --i-have-user-authorisation-to-finetune)
CUDA_VISIBLE_DEVICES= $PY -u train_s6f0_all.py "${A[@]}" --check-targets 2>&1 | grep -v Warning | tail -4
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "STEP5-CHECK-FAILED"; exit 1; }
echo "STEP5-TRAIN-START $(date)"
for a in 1 2 3; do $PY -u train_s6f0_all.py "${A[@]}" 2>&1 | grep --line-buffered -v Warning >> ../train_comb.log; [ ${PIPESTATUS[0]} -eq 0 ] && break; echo "trainer exit, resume $a $(date)"; sleep 60; done
echo "CHAIN-S6FM-DONE $(date)"
