"""data/folds.csv: 5 patient-grouped folds over the LLM-labeled studies; gold studies get fold -1
(never trained on, kept as the final check). Stratified by how many findings a study has."""
import json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

D = "/cluster/project/data"
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
     "PF OA","Effusion","Synovitis","Baker\x27s","Contusion","Fracture"]
tr = pd.read_csv(f"{D}/train.csv")
meta = pd.read_csv(f"{D}/study_meta.csv", keep_default_na=False, dtype=str).set_index("StudyInstanceUID")
llm = pd.DataFrame([json.loads(l) for l in open(f"{D}/llm_labels_v3.jsonl")]).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
df = pd.DataFrame({"StudyInstanceUID": tr.StudyInstanceUID})
df["gold"] = tr[L].notna().all(axis=1).astype(int).values
pid = df.StudyInstanceUID.map(meta.PatientID).fillna("")
df["group"] = np.where(pid != "", pid, df.StudyInstanceUID)
npos = (llm.reindex(df.StudyInstanceUID)[L].values >= 2).sum(1)
df["strat"] = np.clip(npos, 0, 5)
df["fold"] = -1
gold_groups = set(df.group[df.gold == 1])
pool = df[(df.gold == 0) & ~df.group.isin(gold_groups)]
sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
for k, (_, va) in enumerate(sgk.split(pool, pool.strat, pool.group)):
    df.loc[pool.index[va], "fold"] = k
df[["StudyInstanceUID", "fold", "gold", "group"]].to_csv(f"{D}/folds.csv", index=False)
print(df.fold.value_counts().sort_index().to_dict())
print("gold:", int(df.gold.sum()), "| non-gold studies sharing a patient with gold (left out, fold -1):",
      int(((df.gold == 0) & df.group.isin(gold_groups)).sum()))
print("patients:", df.group.nunique(), "| patients with >1 study:", int((df.groupby("group").size() > 1).sum()))
