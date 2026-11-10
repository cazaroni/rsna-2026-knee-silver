import pandas as pd
d = pd.read_csv("/cluster/project/fastmri/fm_side_pred.csv", keep_default_na=False)
u = d[~d.known_side.isin(["L", "R"]) & ((d.p_right > 0.95) | (d.p_right < 0.05))].copy()
u["side"] = (u.p_right > 0.5).map({True: "R", False: "L"})
u[["study", "side"]].to_csv("/cluster/project/fastmri/fm_extra_side.csv", index=False)
open("/cluster/project/fastmri/fm_extra_dirs.txt", "w").write("\n".join(s + "/" for s in u.study) + "\n")
print(len(u), "extra confident-side studies:", u.side.value_counts().to_dict())
