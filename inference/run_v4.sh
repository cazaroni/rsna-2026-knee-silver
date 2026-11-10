#!/bin/bash
# V1 = B4-fm-30 with our part = 1/3 our TTA students + 2/3 S6-fm (share 0.30). Check, submit on pass.
cd ~/rsna-knee; export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_token); K=venv/bin/kaggle
export S6_SOLO_SRC=$HOME/rsna-knee/kaggle_pkg/nb_s6fm_solo/rsna-knee-s6fm-solo.py
sed -e 's|0.5 \* _a\[_LAB\].rank(pct=True) + 0.5 \* _b\[_LAB\].rank(pct=True)|0.0 * _a[_LAB].rank(pct=True) + 1.0 * _b[_LAB].rank(pct=True)|' \
    -e "s|our part = 1/2 rank(our TTA students) + 1/2 rank(S6-ours)|our part = S6-fm only (students weight 0)|" patch_b4.py > patch_v2.py
grep -q "1.0 \* _b\[_LAB\]" patch_v2.py || { echo "PATCH-V2-FAILED"; exit 1; }
B=heliosli/rsna-knee-r6-frozen-fe055406-w1,nartaa/rsna-knee-publication-swa-weights-20261007,heliosli/rsna-orthofoundation-fold0-epoch7-20261004,yorchjais/rsna-knee-s6fm
venv/bin/python make_nb_block3.py kaggle_pkg/nb_v4fm config_FM_tta.json 0.4 yorchjais/rsna-knee-v4fm ext_nb/degbest/rnsa-best-public-score.ipynb $B yorchjais/rsna-knee-ext-pkg 2.0 | tail -1
python3 patch_v2.py kaggle_pkg/nb_v4fm/block-ours.ipynb || exit 1
grep -q "rsna-knee-s6fm" kaggle_pkg/nb_v4fm/block-ours.ipynb || { echo "V2 lacks s6fm"; exit 1; }
./test_nb.sh kaggle_pkg/nb_v4fm yorchjais/rsna-knee-v4fm /tmp/par/ref_FMtta2_cpu_raw.npz 0.03 9 3 > test_v4fm.log 2>&1; R=$?
tail -2 test_v4fm.log
ok=$R; grep -qh "S6-fm only" /tmp/par/kout_rsna-knee-v4fm_v*/*.log 2>/dev/null || { ok=1; echo "mix line missing"; }
if [ $ok -eq 0 ]; then V=$(grep -oE "TEST-PASS version [0-9]+" test_v4fm.log | grep -oE "[0-9]+$"); $K competitions submit rsna-knee-abnormality-detection -k yorchjais/rsna-knee-v4fm -v $V -f submission.csv -m "V4: R6 block (single view) 60% + S6-fm only 40%" 2>&1 | tail -1; else echo "V2 NOT SUBMITTED"; fi
echo RUN-V4-DONE
