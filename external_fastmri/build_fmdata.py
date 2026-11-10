"""~/rsna-knee-fm: a separate data root = ~/rsna-knee (symlinked, untouched) + the 5,046 side-known fastMRI studies.
fastMRI rows: folds.csv fold=9 (training only, never validation/gold), laterality.csv side from tag/series name (source 'fm_tag'),
llm_labels_v3.jsonl all-zero grade placeholders (their training target always comes from --ext_labels at weight 1.0).
npy_fm/ = symlinks to every npy3 study dir plus every fm3 study dir. fastMRI data never leaves Ramona/YUCA (DUA)."""
import os, json, pandas as pd
SRC, DST = os.path.expanduser("~/rsna-knee"), os.path.expanduser("~/rsna-knee-fm")
os.makedirs(DST, exist_ok=True); os.makedirs(f"{DST}/runs", exist_ok=True)
SPECIAL = {"folds.csv", "laterality.csv", "llm_labels_v3.jsonl", "runs", "npy_fm"}
for n in os.listdir(SRC):
    if n in SPECIAL or n.startswith("npy_f0") or n.startswith("."): continue
    p = f"{DST}/{n}"
    if not os.path.lexists(p): os.symlink(f"{SRC}/{n}", p)
fm = pd.read_csv(f"{SRC}/fm_known_side.csv")
fm = fm[fm.study.map(lambda u: os.path.isdir(f"{SRC}/fm3/{u}"))]
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
f = pd.read_csv(f"{SRC}/folds.csv")
add = pd.DataFrame({"StudyInstanceUID": fm.study, "fold": 9, "gold": 0, "group": fm.study})
pd.concat([f, add[f.columns]], ignore_index=True).to_csv(f"{DST}/folds.csv", index=False)
lat = pd.read_csv(f"{SRC}/laterality.csv", keep_default_na=False)
la = pd.DataFrame({"StudyInstanceUID": fm.study, "tag": fm.side, "report": "", "side": fm.side, "source": "fm_tag", "p_right": ""})
pd.concat([lat, la[lat.columns]], ignore_index=True).to_csv(f"{DST}/laterality.csv", index=False)
with open(f"{DST}/llm_labels_v3.jsonl", "w") as out:
    for l in open(f"{SRC}/llm_labels_v3.jsonl"): out.write(l)
    for u in fm.study: out.write(json.dumps({"StudyInstanceUID": u, **{k: 0 for k in L}}) + "\n")
os.makedirs(f"{DST}/npy_fm", exist_ok=True)
for u in os.listdir(f"{SRC}/npy3"):
    if not os.path.lexists(f"{DST}/npy_fm/{u}"): os.symlink(f"{SRC}/npy3/{u}", f"{DST}/npy_fm/{u}")
for u in fm.study:
    if not os.path.lexists(f"{DST}/npy_fm/{u}"): os.symlink(f"{SRC}/fm3/{u}", f"{DST}/npy_fm/{u}")
print("fastMRI studies added:", len(fm), "| npy_fm entries:", len(os.listdir(f"{DST}/npy_fm")), "| folds rows:", len(pd.read_csv(f"{DST}/folds.csv")))
