"""Generation-2 self-training targets: 0.5 * Opus p + 0.5 * out-of-fold prediction of the fold-STUDENT that never saw the study
(cnxt_fstud_f0..f4, fixed EMA 0.9995 weights; the fold-students were trained on the generation-1 targets). Studies without an
out-of-fold prediction keep their Opus p. Writes relabel/student2gen_opus50.csv."""
import numpy as np, pandas as pd
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
opus = pd.read_csv("relabel/opus_p.csv").set_index("StudyInstanceUID")[L]
parts = []
for k in range(5):
    z = np.load(f"runs/cnxt_fstud_f{k}/preds.npz", allow_pickle=True); parts.append(pd.DataFrame(z["Pv_ema0.9995"], index=z["va_ids"], columns=L))
oof = pd.concat(parts); assert oof.index.is_unique
t = oof.reindex(opus.index); miss = int(t.isna().any(axis=1).sum())
out = (0.5 * opus + 0.5 * t.fillna(opus)).clip(0, 1)
out.reset_index().to_csv("relabel/student2gen_opus50.csv", index=False)
g1 = pd.read_csv("relabel/student_opus50.csv").set_index("StudyInstanceUID")[L].loc[out.index]
print(f"gen-2 targets: {len(out)} studies, {len(oof)} with a fold-student teacher, {miss} kept at Opus p")
print("mean |gen2 - gen1| per finding:", (out - g1).abs().mean().round(4).to_dict())
