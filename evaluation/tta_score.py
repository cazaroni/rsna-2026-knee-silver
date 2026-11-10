"""Crop-offset TTA screen (Codex Astra idea 3), rule fixed beforehand: candidate = mean probability over the centre crop and two
opposite diagonal offsets (+0.08,+0.08) and (-0.08,-0.08) [3-view], also all five views; baseline = centre crop. Per fold, the
fold's models are rank-averaged (as in the ensemble). Pass: ensemble Opus-ruler gain >= +0.0003 on BOTH folds, consensus not down."""
import glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
D = "/cluster/project/data"
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
opus = pd.read_csv(f"{D}/opus_p.csv").set_index("StudyInstanceUID")[L]
cons = pd.read_csv(f"{D}/consensus_labels.csv").set_index("StudyInstanceUID")[L]
folds = pd.read_csv(f"{D}/folds.csv").set_index("StudyInstanceUID")
SH = {"c": "0_0", "pp": "0p08_0p08", "mm": "m0p08_m0p08", "pm": "0p08_m0p08", "mp": "m0p08_0p08"}
def mac(P, Y):
    return np.nanmean([roc_auc_score(Y[:, j], P[:, j]) for j in range(12) if 0 < Y[:, j].sum() < len(Y)])
def ld(f):
    d = np.load(f, allow_pickle=True); return pd.DataFrame(1 / (1 + np.exp(-d["logit"].astype(np.float64))), index=list(d["ids"]), columns=L)
rk = lambda x: pd.DataFrame(x).rank(pct=True).values
for F in (0, 1):
    models = sorted({f.split(f"tta_f{F}_")[1].rsplit("_" + SH["c"], 1)[0] for f in glob.glob(f"{D}/runs/tta_f{F}_*_{SH['c']}_feats.npz")})
    ids = [u for u in folds.index[folds.fold == F]]
    Yo, Yc = (opus.loc[ids].values >= 0.5), (cons.loc[ids].values >= 0.5)
    E = {"centre": [], "3-view": [], "5-view": []}
    for m in models:
        V = {k: ld(f"{D}/runs/tta_f{F}_{m}_{v}_feats.npz").loc[ids].values for k, v in SH.items()}
        c, t3, t5 = V["c"], (V["c"] + V["pp"] + V["mm"]) / 3, sum(V.values()) / 5
        print(f"fold {F} {m:32s} Opus centre {mac(c, Yo):.4f} 3-view {mac(t3, Yo) - mac(c, Yo):+.4f} 5-view {mac(t5, Yo) - mac(c, Yo):+.4f} | consensus 3-view {mac(t3, Yc) - mac(c, Yc):+.4f}")
        E["centre"].append(rk(c)); E["3-view"].append(rk(t3)); E["5-view"].append(rk(t5))
    e = {k: sum(v) / len(v) for k, v in E.items()}
    for k in ("3-view", "5-view"):
        print(f"fold {F} ENSEMBLE {k}: Opus {mac(e[k], Yo) - mac(e['centre'], Yo):+.4f} | consensus {mac(e[k], Yc) - mac(e['centre'], Yc):+.4f}")
print("TTA-SCORE-DONE")
