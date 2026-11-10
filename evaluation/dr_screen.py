"""Reweighting screen (review-outlandish-fable.md idea 3): candidate runs vs the mean of o1_opus_f0 and o1_opus_f0_s1, fold 0.
Rulers: Opus p>=0.5, consensus, agreed cells, explicit cells (Opus status explicit_positive/explicit_negative only); gold SOFT."""
import sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
cands = sys.argv[1].split(","); bases = ["o1_opus_f0", "o1_opus_f0_s1"]
ld = lambda t: np.load(f"runs/{t}/preds.npz", allow_pickle=True)
B = [ld(t) for t in bases]; C = [ld(t) for t in cands]
va = list(B[0]["va_ids"]); assert all(list(x["va_ids"]) == va for x in B + C)
cons = pd.read_csv("public_labels/consensus_labels.csv").set_index("StudyInstanceUID").loc[va, L].values >= 0.5
opus = pd.read_csv("relabel/opus_p.csv").set_index("StudyInstanceUID").loc[va, L].values >= 0.5
st = {}
for l in open("relabel/labels_opus.jsonl"):
    r = json.loads(l); st[r["StudyInstanceUID"]] = [r[f]["status"] for f in L]
S = np.array([st[u] for u in va]); expl = np.isin(S, ["explicit_positive", "explicit_negative"])
R = {"Opus": (opus, None), "consensus": (cons, None), "agreed": (opus, cons == opus), "explicit": (opus, expl)}
def mac(P, Y, mk=None, idx=None, per=False):
    idx = np.arange(len(P)) if idx is None else idx; out = []
    for j in range(12):
        k = idx if mk is None else idx[mk[idx, j]]; y = Y[k, j]
        out.append(roc_auc_score(y, P[k, j]) if 0 < y.sum() < len(y) else np.nan)
    return np.array(out) if per else np.nanmean(out)
rng = np.random.default_rng(0); n = len(va); BS = [rng.choice(n, n) for _ in range(1000)]
for rn, (Y, mk) in R.items():
    bm = np.mean([mac(b["Pv"], Y, mk) for b in B])
    per = [mac(c["Pv"], Y, mk) - bm for c in C]
    bs = [np.mean([mac(c["Pv"], Y, mk, i) for c in C]) - np.mean([mac(b["Pv"], Y, mk, i) for b in B]) for i in BS]
    print(f"{rn:10s} " + "  ".join(f"{t} {d:+.4f}" for t, d in zip(cands, per)) + f"  | pooled {np.mean(per):+.4f} 90% CI [{np.percentile(bs,5):+.4f}, {np.percentile(bs,95):+.4f}]")
g = [mac(x["Pg"], x["Ygo"] >= 0.5) for x in B + C]
print("gold (SOFT): bases", [round(v, 4) for v in g[:2]], "candidates", [round(v, 4) for v in g[2:]])
pb = np.nanmean([mac(b["Pv"], opus, per=True) for b in B], 0); pc = np.nanmean([mac(c["Pv"], opus, per=True) for c in C], 0)
print("per finding, Opus ruler (candidate - base):", {l: round(float(d), 4) for l, d in zip(L, pc - pb)})
