"""Self-training targets: 0.5 * Opus p + 0.5 * out-of-fold prediction of the Opus ConvNeXt fold model that never saw the study
(SWA weights, folds 0-4). Studies without an out-of-fold prediction keep their Opus p. Writes relabel/student_opus50.csv."""
import numpy as np, pandas as pd
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
opus = pd.read_csv("relabel/opus_p.csv").set_index("StudyInstanceUID")[L]
runs = ["o1_opus_f0"] + [f"cnxt_opus_f{k}" for k in range(1, 5)]
parts = []
for r in runs:
    z = np.load(f"runs/{r}/preds.npz", allow_pickle=True); parts.append(pd.DataFrame(z["Pv"], index=z["va_ids"], columns=L))
oof = pd.concat(parts); assert oof.index.is_unique, "a study has two out-of-fold predictions"
t = oof.reindex(opus.index)
miss = int(t.isna().any(axis=1).sum())
out = (0.5 * opus + 0.5 * t.fillna(opus)).clip(0, 1)
out.reset_index().to_csv("relabel/student_opus50.csv", index=False)
print(f"student targets: {len(out)} studies, {len(oof)} with an out-of-fold teacher, {miss} kept at Opus p only")
print("mean target  Opus:", opus.mean().round(3).to_dict()); print("mean target  student:", out.mean().round(3).to_dict())
