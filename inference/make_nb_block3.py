"""make_nb_block3.py <out_dir> <our config> <our share> <kernel id> <source block notebook> <block datasets, comma> <our package dataset> <our stage hours>
As make_nb_block2, plus (Sol review 10-17): the package dataset and our stage's time budget are explicit; exactly one attached package
may contain knee_infer.py; the final blend uses the BLOCK's rank for any study our stage did not score (scored IDs from
submission_raw.npz) and prints coverage, instead of ranking our 0.5 fallbacks."""
import json, re, sys, os
out_dir, cfg, share, kid, src_nb, blk_ds, pkg_ds, hours = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6].split(","), sys.argv[7], float(sys.argv[8])
src = open("/workspace/rsna-knee/make_nb.py").read()
code1 = re.search(r"code1 = r'''(.*?)'''", src, re.S).group(1)
code2 = re.search(r"code2 = r'''(.*?)'''", src, re.S).group(1)
code2 = code2.replace("__CONFIG__", cfg).replace('"submission.csv"', '"sub_ours.csv"').replace("deadline=t0 + 8 * 3600", f"deadline=NB_T0 + {hours} * 3600")
assert '"sub_ours.csv"' in code2 and f"NB_T0 + {hours} * 3600" in code2 and 'm.get("crop")' in code2
guard = '# [ours] exactly one attached package may hold knee_infer.py (no ambiguous first-match discovery)\nimport glob as _g\n_pk = sorted(set(_g.glob("/kaggle/input/*/knee_infer.py") + _g.glob("/kaggle/input/*/*/knee_infer.py") + _g.glob("/kaggle/input/*/*/*/knee_infer.py")))\nprint("[ours] packages with knee_infer.py:", _pk)\nassert len(_pk) == 1, "expected exactly one package with knee_infer.py"\n'
blk = json.load(open(src_nb)); cells = blk["cells"]; n_t0 = 0
for c in cells:
    s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    if "NB_T0 = time.time()" in s:
        c["source"] = s.replace("NB_T0 = time.time()", "NB_T0 = globals().get('NB_T0', time.time())   # [ours] counts from the notebook start").splitlines(True); n_t0 += 1
assert n_t0 == 1
t0cell = "# [ours] notebook clock: every stage's time guard counts from here\nimport time\nNB_T0 = time.time()\n"
free = "# [ours] release our GPU memory before the block stages\nimport gc, torch\nfor _n in [k for k in list(globals()) if k in ('sub', 'stats', 'ki')]:\n    globals().pop(_n, None)\ngc.collect(); torch.cuda.empty_cache()\nprint('[ours] GPU memory after release (MB):', [round(torch.cuda.memory_reserved(i) / 2**20) for i in range(torch.cuda.device_count())])\n"
final = r'''# [ours] final: per-finding rank blend of the public block and our ensemble (share fixed before any run). Studies our stage did
# not score (time budget / failure) take the block's rank instead of a ranked 0.5 fallback.
import pandas as pd, numpy as np, shutil, time, os
SHARE = __SHARE__
shutil.copyfile("/kaggle/working/submission.csv", "/kaggle/working/sub_block.csv")
blk = pd.read_csv("/kaggle/working/sub_block.csv").set_index("StudyInstanceUID")
ours = pd.read_csv("sub_ours.csv").set_index("StudyInstanceUID")
assert set(blk.index) == set(ours.index) and list(blk.columns) == list(ours.columns), "row/column mismatch block vs ours"
ours = ours.loc[blk.index]
scored = set(np.load("submission_raw.npz", allow_pickle=True)["ids"].tolist()) if os.path.exists("submission_raw.npz") else set()
ok = blk.index.isin(list(scored))
rb = blk.rank(pct=True)
ro = pd.DataFrame(np.nan, index=blk.index, columns=blk.columns)
if ok.any(): ro.loc[ok] = ours.loc[ok].rank(pct=True).values
out = (1 - SHARE) * rb + SHARE * ro.fillna(rb)
assert np.isfinite(out.values).all()
out.reset_index().to_csv("/kaggle/working/submission.csv", index=False)
print(f"BLOCK_PLUS_OURS share {SHARE} | rows {len(out)} | ours scored {int(ok.sum())}/{len(out)} | total seconds {round(time.time() - NB_T0)}")
'''.replace("__SHARE__", repr(share))
mk = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(True)}
blk["cells"] = [mk(t0cell), mk(guard), mk("# [ours] our own-data ensemble first: " + cfg + "\n" + code1), mk(code2), mk(free)] + cells + [mk(final)]
os.makedirs(out_dir, exist_ok=True)
json.dump(blk, open(f"{out_dir}/block-ours.ipynb", "w"), indent=1)
meta = {"id": kid, "title": kid.split("/")[1], "code_file": "block-ours.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": "true", "enable_gpu": "true", "enable_internet": "false", "dataset_sources": blk_ds + [pkg_ds],
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "machine_shape": "NvidiaTeslaT4"}
assert "yorchjais/rsna-knee-infer-pkg" not in blk_ds or pkg_ds == "yorchjais/rsna-knee-infer-pkg"
json.dump(meta, open(f"{out_dir}/kernel-metadata.json", "w"), indent=1)
print("written", out_dir, "| cells", len(blk["cells"]), "| config", cfg, "| share", share, "| package", pkg_ds, "| our stage hours", hours)
