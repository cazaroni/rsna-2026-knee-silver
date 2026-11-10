import pandas as pd
t = pd.read_csv("/cluster/project/fastmri/series_table.csv", keep_default_na=False)
s = t.groupby("study").side.agg(lambda x: "".join(sorted(set(v for v in x if v in ("L", "R")))))
k = s[s.isin(["L", "R"])]
k.rename("side").reset_index().to_csv("/cluster/project/fastmri/fm_known_side.csv", index=False)
open("/cluster/project/fastmri/fm_known_list.txt", "w").write("\n".join(k.index) + "\n")
print(len(k), "known-side studies:", k.value_counts().to_dict())
