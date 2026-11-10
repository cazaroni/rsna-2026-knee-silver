"""B3 = R6 block with R6's flip TTA off + S6-ours stage; our 30% part = rank mix 2/3 our TTA students + 1/3 S6-ours."""
import json, sys
p = sys.argv[1]; nb = json.load(open(p)); C = nb["cells"]
src = lambda i: "".join(C[i]["source"])
mk = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(True)}
fl = [i for i in range(len(C)) if "FLIP_TTA = True" in src(i)]; assert len(fl) == 1, fl
s = src(fl[0]); assert s.count("FLIP_TTA = True") == 1; C[fl[0]]["source"] = s.replace("FLIP_TTA = True", "FLIP_TTA = False   # [ours] B3: R6 single view to fit S6-ours in the 9 h budget").splitlines(True)
rel = [i for i in range(len(C)) if src(i).startswith("# [ours] release our GPU memory")]; assert len(rel) == 1, rel
import os
solo = open(os.environ.get("S6_SOLO_SRC", "/workspace/rsna-knee/kaggle_pkg/nb_s6ours_solo/rsna-knee-s6ours-solo.py")).read()
for old, new in [("A_TIMEOUT_S = 9000", "A_TIMEOUT_S = 5400"), ("GUARD_MAX_S = 480 * 60", "GUARD_MAX_S = 300 * 60"), ("B_TIMEOUT_S = 14400", "B_TIMEOUT_S = 9000")]:
    assert solo.count(old) == 1, old; solo = solo.replace(old, new)
c_w = mk("%%writefile /tmp/s6ours_solo.py\n" + solo)
c_r = mk("""# [ours] S6-ours stage (our retrain of S6 on our labels); its output joins OUR part, not the block
import subprocess, sys, os, shutil, time
_t = time.time(); _rc = subprocess.run([sys.executable, '-u', '/tmp/s6ours_solo.py']).returncode
assert _rc == 0 and os.path.isfile('/kaggle/working/submission.csv'), f'[ours] S6-ours stage failed rc={_rc}'
shutil.move('/kaggle/working/submission.csv', '/kaggle/working/sub_s6ours.csv'); shutil.rmtree('/kaggle/working/k1', ignore_errors=True)
print(f'S6OURS_STAGE_DONE {time.time() - _t:.0f}s', flush=True)
import pandas as pd
_LAB = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
_a = pd.read_csv('/kaggle/working/sub_ours.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
_b = pd.read_csv('/kaggle/working/sub_s6ours.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[_a.index]
assert _b[_LAB].notna().all().all()
_mix = (2 / 3) * _a[_LAB].rank(pct=True) + (1 / 3) * _b[_LAB].rank(pct=True)
shutil.copyfile('/kaggle/working/sub_ours.csv', '/kaggle/working/sub_ours_mil.csv')
_mix.reset_index().to_csv('/kaggle/working/sub_ours.csv', index=False)
print('[ours] our part = 2/3 rank(our TTA students) + 1/3 rank(S6-ours) | rows', len(_mix), flush=True)
""")
C[rel[0] + 1:rel[0] + 1] = [c_w, c_r]
json.dump(nb, open(p, "w"), indent=1); print("B3 patched: flip off in cell", fl[0], "| S6 stage after cell", rel[0], "| bytes", len(json.dumps(nb)))
