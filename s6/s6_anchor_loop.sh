#!/bin/bash
# Build S6 anchors on CPU for every pulled chunk (resumable; the precompute skips finished studies). Ends after the pull is done.
cd ~/s6data/code
while true; do
  PD=$(grep -c PULL-DONE ../pull.log)
  cat ../chunks/c??.done 2>/dev/null >/dev/null
  python3 -c "import glob,json; u=[l.strip() for f in sorted(glob.glob('../chunks/c??')) if __import__('os').path.exists(f+'.done') for l in open(f) if l.strip()]; json.dump(u, open('data/order_pulled.json','w')); print(len(u))"
  HF_HUB_OFFLINE=1 nice -n 5 ../venv312/bin/python -u precompute_anchors.py --order data/order_pulled.json --out anchors --tmp /tmp/s6_anch --workers 12 2>&1 | grep -v Warning | grep -E "SUMMARY|to do|error" | tail -3
  [ "$PD" -ge 1 ] && break
  grep -q PULL-FAILED ../pull.log && break
  sleep 300
done
echo "ANCHORS-LOOP-DONE $(date)"
