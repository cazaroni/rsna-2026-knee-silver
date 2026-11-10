"""fastMRI clinical knee DICOMs -> our six-slot npy3-style cache, with make_npy_v3.work UNCHANGED (same 384 px square resize,
canonical LPS orientation, 1-99 % clip, 48-slice cap). Only the file reader and the series table differ:
  - plane: from ImageOrientationPatient (work() also re-derives it geometrically);
  - fluid-sensitive: 'FS' in ScanOptions, or fat-sat / STIR / T2 named in SeriesDescription;
  - one series per (plane, fluid) slot: the one with most files.
Writes fastmri/series_table.csv, fastmri/fm3/<study_key>/<slot>.npy and fastmri/meta_fm3.jsonl. Data stays on YUCA (fastMRI DUA)."""
import os, re, sys, json, glob, csv
from multiprocessing import Pool
import numpy as np, pydicom
P = "/cluster/project"; FM = f"{P}/fastmri"
os.environ["OUT"] = f"{FM}/fm3"; os.environ["META"] = f"{FM}/meta_fm3.jsonl"
sys.path.insert(0, f"{P}/code")
import make_npy_v3 as m
assert m.OUT.endswith("/fm3")
class FileZ:                                   # stands in for the competition zip: read(path) -> bytes
    def read(self, p):
        with open(p, "rb") as f: return f.read()
FLUID = re.compile(r"(fat ?sat|fatsat|ftst|\bfs\b|stir|\bt2\b|t2)", re.I)
def scan_series(sdir):
    files = sorted(glob.glob(f"{sdir}/*.dcm"))
    if not files: return None
    try: d = pydicom.dcmread(files[0], stop_before_pixels=True)
    except Exception: return None
    try:
        o = np.array(d.ImageOrientationPatient, float); n = np.cross(o[:3], o[3:]); plane = ["Sagittal", "Coronal", "Axial"][int(np.argmax(np.abs(n)))]
    except Exception: plane = "?"
    desc = str(d.get("SeriesDescription", "")); so = str(d.get("ScanOptions", ""))
    fl = int(("FS" in re.findall(r"[A-Z0-9]+", so)) or bool(FLUID.search(desc)))
    side = str(d.get("Laterality", "")).strip().upper()
    if not side:
        side = "L" if re.match(r"^\s*lt", desc, re.I) else ("R" if re.match(r"^\s*rt", desc, re.I) else "")
    return dict(series_dir=sdir, plane=plane, fluid=fl, n=len(files), desc=desc, scanopt=so, side=side, files=files)
def study_key(sd):
    pat, st = sd.rstrip("/").split("/")[-2:]
    return "fm_" + re.sub(r"[^A-Za-z0-9]+", "_", pat).strip("_") + "_" + st
if __name__ == "__main__":
    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    studies = sorted(glob.glob(f"{FM}/raw/batch*/*/*/study_*"))
    if os.environ.get("LIMIT"): studies = studies[: int(os.environ["LIMIT"])]
    print(len(studies), "studies", flush=True)
    sdirs = [(study_key(s), d) for s in studies for d in sorted(glob.glob(f"{s}/*/"))]
    with Pool(nproc) as pool: info = pool.map(scan_series, [d for _, d in sdirs], chunksize=8)
    rows, jobs, best = [], [], {}
    for (k, _), r in zip(sdirs, info):
        if r is None or r["plane"] == "?": continue
        rows.append({"study": k, **{x: r[x] for x in ("series_dir", "plane", "fluid", "n", "desc", "scanopt", "side")}})
        key = (k, r["plane"], r["fluid"])
        if key not in best or r["n"] > best[key]["n"]: best[key] = r
    with open(f"{FM}/series_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(len(rows), "series |", len(best), "slots chosen", flush=True)
    jobs = [(k, pl, fl, r["series_dir"], r["files"]) for (k, pl, fl), r in best.items()]
    m._z = FileZ()
    with Pool(nproc) as pool, open(m.META, "w") as f:
        for i, meta in enumerate(pool.imap_unordered(m.work, jobs, chunksize=4), 1):
            f.write(json.dumps(meta) + "\n")
            if i % 2000 == 0: print(i, flush=True)
    import collections
    st = collections.Counter(json.loads(l)["status"] for l in open(m.META))
    print("status:", dict(st)); print("FM3-DONE", flush=True)
