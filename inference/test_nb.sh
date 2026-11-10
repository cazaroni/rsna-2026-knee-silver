#!/bin/bash
# test_nb.sh <nb dir> <kernel id> <reference raw npz> <raw tolerance> <expected raw members> <views per model (1 = no TTA)>
# Hardened (Sol review 10-17): fails non-zero on ANY problem; requires a parsed push version and status COMPLETE; checks the final
# blend line, our coverage, raw parity, member count and (for TTA) that views differ. Prints TEST-PASS only when everything holds.
set -o pipefail
cd ~/rsna-knee; export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_token); K=venv/bin/kaggle
D=$1; KID=$2; REF=$3; TOL=$4; NM=$5; NV=$6
P=$($K kernels push -p $D 2>&1); echo "$P" | tail -1
V=$(echo "$P" | grep -oE "Kernel version [0-9]+ successfully pushed" | grep -oE "[0-9]+"); [ -n "$V" ] || { echo "TEST-FAIL push"; exit 1; }
echo "pushed version $V"
for i in $(seq 1 400); do s=$($K kernels status $KID 2>&1 | tail -1); echo "$s" | grep -qiE "complete|error|cancel" && break; sleep 45; done
echo "$s" | grep -qi "KernelWorkerStatus.COMPLETE" || { echo "TEST-FAIL status: $s"; exit 1; }
O=/tmp/par/kout_$(basename $KID)_v$V; rm -rf $O; mkdir -p $O
for i in 1 2 3 4; do $K kernels output $KID -p $O >/dev/null 2>&1; head -c 20 $O/submission.csv 2>/dev/null | grep -q StudyInstanceUID && head -c 20 $O/sub_ours.csv 2>/dev/null | grep -q StudyInstanceUID && break; sleep 30; done
venv/bin/python - "$O" "$REF" "$TOL" "$NM" "$NV" "$V" <<'PY'
import sys, glob, re, numpy as np, pandas as pd
O, REF, TOL, NM, NV, V = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
log = open(glob.glob(f"{O}/*.log")[0]).read().replace("\\u003e", ">")
m = re.search(r"BLOCK_PLUS_OURS share ([0-9.]+) \| rows (\d+) \| ours scored (\d+)/(\d+)", log)
sub = pd.read_csv(f"{O}/submission.csv"); ra = np.load(f"{O}/submission_raw.npz", allow_pickle=True); rb = np.load(REF, allow_pickle=True)
checks = {
  "final blend line": m is not None,
  "ours scored all rows": m is not None and m.group(3) == m.group(4) == str(len(sub)),
  "submission 3 rows, finite, in [0,1]": len(sub) == 3 and np.isfinite(sub.iloc[:, 1:].values).all() and ((sub.iloc[:, 1:].values >= 0) & (sub.iloc[:, 1:].values <= 1)).all(),
  "raw ids match reference": list(ra["ids"]) == list(rb["ids"]),
  "raw members": ra["A"].shape[1] == NM,
  "raw within tolerance": list(ra["ids"]) == list(rb["ids"]) and float(np.abs(ra["A"] - rb["A"]).max()) < TOL,
  "no Traceback in log": "Traceback" not in log,
}
if NV > 1:
    A = ra["A"]; checks["views differ"] = all(float(np.abs(A[:, NV*k] - A[:, NV*k+v]).max()) > 1e-4 for k in range(NM // NV) for v in range(1, NV))
    checks["tta-group line"] = "[tta-group]" in log
for k, v in checks.items(): print(("PASS " if v else "FAIL ") + k)
print("raw max diff", float(np.abs(ra["A"] - rb["A"]).max()) if checks["raw ids match reference"] else None)
ok = all(checks.values()); print(f"TEST-{'PASS' if ok else 'FAIL'} version {V}"); sys.exit(0 if ok else 1)
PY
