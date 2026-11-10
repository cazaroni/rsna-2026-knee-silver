#!/bin/bash
# Full S6 all-data run on our gen-2 student targets. Waits for the pull + anchors, checks anchors and targets, then trains.
# Same argv as nartaa's run_all_alldata.sh except: our targets, workers 16, no --grad-ckpt (memory-only option), our paths.
cd ~/s6data
until grep -q PULL-DONE pull.log && grep -q ANCHORS-LOOP-DONE anchors.log; do grep -q PULL-FAILED pull.log && { echo "PULL FAILED"; exit 1; }; sleep 60; done
cd code
python3 - <<'PY' || exit 1
import json, sys
pool = json.load(open("data/pool_ok.json"))
ok = {json.loads(l)["uid"] for l in open("anchors/manifest.jsonl") if json.loads(l).get("error") is None}
miss = [u for u in pool if u not in ok]
print(f"anchors ok {len(ok & set(pool))} of pool {len(pool)} | missing {len(miss)}")
sys.exit(0 if len(miss) <= 10 else 1)
PY
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=(--pool data/pool_ok.json --folds-json data/folds_v0.json --targets data/targets_ours.csv --out ../runs_ours --arms E_prior
   --backbone convnext_small --init $HOME/rsna-knee/convnext_small_fb_in22k_ft_in1k.safetensors --seeds 1 --seed-base 42 --folds 5 --only-folds 0
   --expect-train 4263 --expect-holdout 0 --all-data
   --epochs 16 --batch 2 --train-windows 64 --size 320 --views v4 --crop 288 --slices 3 --phys-mm 3.5
   --spatial-attn --lr 1e-4 --head-lr 1e-3 --wd 0.05 --aug crop --train-sampling uniform --ema 0
   --workers 16 --loader-timeout 600 --max-fail-frac 0.005 --log-every 200 --work-tmp /tmp/s6f0_work
   --resume --no-final-predict --i-have-user-authorisation-to-finetune)
CUDA_VISIBLE_DEVICES= ../venv312/bin/python -u train_s6f0_all.py "${A[@]}" --check-targets 2>&1 | grep -v Warning | tail -6
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "CHECK-TARGETS FAILED"; exit 1; }
echo "TRAIN-START $(date)"
for a in 1 2 3; do ../venv312/bin/python -u train_s6f0_all.py "${A[@]}" 2>&1 | grep --line-buffered -v Warning >> ../train_ours.log; [ ${PIPESTATUS[0]} -eq 0 ] && break; echo "trainer exit, resume attempt $a $(date)"; sleep 60; done
echo "S6-TRAIN-DONE $(date)"; ls -la ../runs_ours
