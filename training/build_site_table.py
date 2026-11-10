"""Per-scanner Synovitis offsets for inference-time centring (code audit 2.2, test 1). Fixed beforehand: alpha 1, Synovitis only.
Scanner key = Manufacturer | Model | field strength | receive coil (study_meta.csv); scanners with < 15 training studies -> "rare".
Offset = scanner mean - overall mean of the out-of-fold Synovitis logit, per model family. No labels are used."""
import json, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]; SY = 8
m = pd.read_csv("study_meta.csv").set_index("StudyInstanceUID")
fs = lambda v: "nan" if pd.isna(v) or str(v).strip() in ("", "nan") else str(float(v))
key = (m.Manufacturer.fillna("?") + "|" + m.ManufacturerModelName.fillna("?") + "|" + m.MagneticFieldStrength.map(fs) + "|" + m.ReceiveCoilName.fillna("?"))
vc = key.value_counts(); common = sorted(vc[vc >= 15].index); site = key.where(key.isin(common), "rare")
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
fam = {"effb0": ["s2_lr2e4_12ep_warm1"] + [f"f{k}_s2recipe" for k in range(1, 5)],
       "cnxt": ["cnxt_tiny_in12k_320_lr3e5"] + [f"cnxt_tiny_lr3e5_f{k}" for k in range(1, 5)], "vit": ["vits_lejepa_ep20_bs12"]}
tables, gold = {}, {}
for f, runs in fam.items():
    zs = [np.load(f"runs/{r}/preds.npz", allow_pickle=True) for r in runs]
    oof = pd.concat([pd.Series(lg(z["Pv"][:, SY]), index=z["va_ids"]) for z in zs])
    s = site.loc[oof.index]; g = oof.groupby(s.values)
    off = (g.mean() - oof.mean()).where(g.size() >= 10, 0.0)          # thin groups (the ViT has only one fold) get no offset
    tables[f] = {k: round(float(v), 5) for k, v in off.items()}
    gold[f] = [lg(z["Pg"]) for z in zs]; go = list(zs[0]["go_ids"]); Yg = zs[0]["Ygo"] >= 0.5
    print(f"{f:6s} OOF studies {len(oof)} | scanners with an offset {int((off != 0).sum())} | offset range {off.min():+.2f} .. {off.max():+.2f}")
json.dump({"keys": common, "families": tables, "note": "Synovitis logit offset per scanner key; key = Manufacturer|Model|float(field)|coil, else 'rare'"}, open("kaggle_pkg/dataset/site_syn_offsets.json", "w"), indent=0)
# sanity on gold for the A ensemble structure (gold-informed idea: this only checks the code path, not a basis for tuning)
r = lambda P: np.apply_along_axis(rankdata, 0, P) / len(P)
og = np.array([tables_v for tables_v in [0]])
def ens(center):
    parts = []
    for f, w, n in (("effb0", 0.125, 5), ("vit", 0.125, 1), ("cnxt", 0.25 / 3, 3)):
        for G in gold[f][:n]:
            G = G.copy()
            if center: G[:, SY] -= site.loc[go].map(tables[f]).fillna(0).values
            parts.append(w * r(G))
    E = sum(parts); E[:, SY] = 0.75 * rankdata(E[:, SY]) + 0.25 * rankdata(E[:, 7]); return E
for c in (False, True):
    E = ens(c); a = [roc_auc_score(Yg[:, j], E[:, j]) for j in range(12)]
    print(f"A-structure on gold, centring {c}: macro {np.mean(a):.4f} | Synovitis {a[SY]:.3f}")
