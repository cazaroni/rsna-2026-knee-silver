#!/bin/bash
# Pull the S6 pool DICOMs from the YUCA zip in chunks of 500 studies (resumable: chunks with a .done marker are skipped).
cd ~/s6data; mkdir -p chunks; split -l 500 -d pool_uids.txt chunks/c
Y="ssh -i $HOME/.ssh/cluster_key -o BatchMode=yes -o ServerAliveInterval=30 user@cluster.example.org"
for c in chunks/c??; do
  [ -f $c.done ] && continue
  for a in 1 2 3; do
    t0=$(date +%s)
    $Y "srun -A <account> -p cpu -c 2 --mem 6G -t 0-04:00:00 python3 ~/s6_stream.py" < $c 2> $c.err | tar -x -C ~/s6data
    if grep -q STREAM-DONE $c.err; then touch $c.done; echo "$c done $(date) $(tail -1 $c.err) in $(( $(date +%s) - t0 )) s"; break; fi
    echo "$c attempt $a failed: $(tail -2 $c.err | tr '\n' ' ')"; sleep 60
  done
  [ -f $c.done ] || { echo "PULL-FAILED at $c"; exit 1; }
done
echo "PULL-DONE $(date) | files $(find train_series -type f | wc -l) | $(du -sh . | cut -f1)"
