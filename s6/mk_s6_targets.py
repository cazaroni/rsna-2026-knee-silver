"""targets_ours.csv: their targets_s6f0.csv with every non-gold L|prior replaced by our gen-2 student targets (same UIDs);
folds_v0.json rebuilt from their fold column (gold = -1). Everything else unchanged."""
import json, pandas as pd, numpy as np, sys
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
src = sys.argv[1]
t = pd.read_csv("data/targets_s6f0.csv", dtype={"StudyInstanceUID": str})
o = pd.read_csv(src).set_index("StudyInstanceUID")[L]
ng = t.is_gold == 0
miss = (~t.loc[ng, "StudyInstanceUID"].isin(o.index)).sum(); assert miss == 0, f"{miss} non-gold studies lack our targets"
for l in L:
    t.loc[ng, f"{l}|prior"] = o.loc[t.loc[ng, "StudyInstanceUID"], l].values
assert t.loc[ng, [f"{l}|prior" for l in L]].notna().all().all() and t.loc[~ng, [f"{l}|prior" for l in L]].isna().all().all()
t.to_csv("data/targets_ours.csv", index=False)
json.dump({"folds": {u: (-1 if g else int(f)) for u, g, f in zip(t.StudyInstanceUID, t.is_gold, t.fold)}}, open("data/folds_v0.json", "w"))
old = pd.read_csv("data/targets_s6f0.csv").loc[ng, [f"{l}|prior" for l in L]].mean().round(3).values
print("rows", len(t), "non-gold", int(ng.sum()), "| mean prior theirs", old, "| ours", t.loc[ng, [f"{l}|prior" for l in L]].mean().round(3).values)
