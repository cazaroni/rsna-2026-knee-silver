"""make_npy v3: every series type at 384 px, straight from the competition zip.

Differences from v2 (which kept one fluid-sensitive series per plane at 256 px):
  - six slots per study: (Sagittal|Coronal|Axial) x (FL fluid-sensitive | NF not), largest series per slot
  - 384 x 384, slices padded to a physical square first so anatomy is not stretched
  - long 3D series are subsampled evenly to at most MAXS slices
Orientation is the same as v2: patient (LPS) axes, canonical order per geometric plane
  Sagittal (x, z, y)   Coronal (y, z, x)   Axial (z, y, x),  superior at the top, nothing mirrored.
Output: <out>/<study>/<Plane>_<FL|NF>.npy uint8, plus meta3.jsonl (one row per series).
"""
import io, json, os, pickle, sys, zipfile
from multiprocessing import Pool
import cv2, numpy as np, pandas as pd, pydicom

P = "/cluster/project"
ZIP = f"{P}/raw/rsna-knee-abnormality-detection.zip"
OUT = os.environ.get("OUT", f"{P}/data/npy3")
META = os.environ.get("META", f"{P}/data/meta3.jsonl")
SIZE, MAXS = 384, 48
CANON = {0: ("Sagittal", (0, 2, 1)), 1: ("Coronal", (1, 2, 0)), 2: ("Axial", (2, 1, 0))}
_z = None

def square(a, ps):
    """Zero-pad a slice to a physical square, then resize to SIZE."""
    h, w = a.shape
    hm, wm = h * ps[0], w * ps[1]
    if hm > wm * 1.01:
        pad = int(round((hm - wm) / ps[1])); a = np.pad(a, ((0, 0), (pad // 2, pad - pad // 2)))
    elif wm > hm * 1.01:
        pad = int(round((wm - hm) / ps[0])); a = np.pad(a, ((pad // 2, pad - pad // 2), (0, 0)))
    return cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA if a.shape[0] > SIZE else cv2.INTER_LINEAR)

def work(job):
    global _z
    study, csv_plane, fl, series, paths = job
    meta = {"study": study, "series": series, "csv_plane": csv_plane, "fluid": int(fl)}
    try:
        if _z is None:
            _z = zipfile.ZipFile(ZIP)
        dss, arrs = [], []
        for p in paths:
            ds = pydicom.dcmread(io.BytesIO(_z.read(p)))
            a = ds.pixel_array.astype(np.float32)
            if a.ndim != 2:
                meta["status"] = f"multiframe_{a.shape}"; return meta
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                a = a.max() - a
            dss.append(ds); arrs.append(a)
        meta["laterality_tag"] = sorted({str(d.get("Laterality", "")).strip().upper() for d in dss} - {""})
        shapes = {a.shape for a in arrs}
        if len(shapes) > 1:
            common = max(shapes, key=lambda s: sum(a.shape == s for a in arrs))
            keep = [i for i, a in enumerate(arrs) if a.shape == common]
            dss, arrs = [dss[i] for i in keep], [arrs[i] for i in keep]
        try:
            iop = np.array(dss[0].ImageOrientationPatient, float)
            r, c = iop[:3], iop[3:]; n = np.cross(r, c)
            keep = [i for i, d in enumerate(dss) if np.allclose(np.array(d.ImageOrientationPatient, float), iop, atol=1e-3)]
            pos = np.array([float(np.dot(np.array(dss[i].ImagePositionPatient, float), n)) for i in keep])
        except Exception:
            meta["status"] = "no_geometry"; return meta
        order = np.argsort(pos); keep = [keep[i] for i in order]; pos = pos[order]
        meta["n_orig"] = len(keep)
        meta["spacing_mm"] = round(float(np.median(np.diff(pos))), 3) if len(pos) > 1 else None
        if len(keep) > MAXS:
            sel = np.linspace(0, len(keep) - 1, MAXS).round().astype(int); keep = [keep[i] for i in sel]; pos = pos[sel]
        raw = np.stack([arrs[i] for i in keep])
        lo, hi = np.percentile(raw, [1, 99])
        raw = np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1)
        try:
            ps = [float(x) for x in dss[0].PixelSpacing]
        except Exception:
            ps = [1.0, 1.0]
        meta["pixel_mm"] = ps
        vol = np.stack([square(s, ps) for s in raw])              # axes: +n, +c (rows), +r (cols)
        dirs = [n, c, r]
        ax = [int(np.argmax(np.abs(d))) for d in dirs]
        if len(set(ax)) < 3:
            meta["status"] = "oblique_ambiguous"; return meta
        for k, d in enumerate(dirs):
            if d[ax[k]] < 0:
                vol = np.flip(vol, axis=k)
        plane, target = CANON[ax[0]]
        vol = vol.transpose([ax.index(t) for t in target])
        zaxis = target.index(2)
        if zaxis > 0:
            vol = np.flip(vol, axis=zaxis)
        slot = f"{plane}_{'FL' if fl else 'NF'}"
        meta.update(geo_plane=plane, slot=slot, status="ok")
        out = f"{OUT}/{study}/{slot}.npy"
        if os.path.exists(out):
            meta["status"] = "slot_taken"; return meta
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out, (np.ascontiguousarray(vol) * 255).astype(np.uint8))
        meta["shape"] = list(vol.shape)
        return meta
    except Exception as e:
        meta["status"] = f"error {e!r}"[:200]; return meta

if __name__ == "__main__":
    idx = pickle.load(open(f"{P}/raw/zip_index.pkl", "rb"))
    se = pd.read_csv(f"{P}/data/train_series.csv")
    se["n"] = [len(idx.get((a, b), [])) for a, b in zip(se.StudyInstanceUID, se.SeriesInstanceUID)]
    se = se[se.n > 0].sort_values("n", ascending=False)
    best = se.drop_duplicates(["StudyInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"])
    jobs = [(r.StudyInstanceUID, r.Anatomical_Plane, r.Fluid_Sensitive, r.SeriesInstanceUID,
             idx[(r.StudyInstanceUID, r.SeriesInstanceUID)]) for r in best.itertuples()]
    if os.environ.get("LIMIT"):
        jobs = jobs[:: max(1, len(jobs) // int(os.environ["LIMIT"]))]
    print(len(jobs), "series", flush=True)
    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    with Pool(nproc) as pool, open(META, "w") as f:
        for i, m in enumerate(pool.imap_unordered(work, jobs, chunksize=4), 1):
            f.write(json.dumps(m) + "\n")
            if i % 1000 == 0:
                print(i, flush=True)
    print("listo", flush=True)
