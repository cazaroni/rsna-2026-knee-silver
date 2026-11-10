"""Inference for the RSNA knee models: raw DICOM study -> 12 probabilities.

The preprocessing here must match the training caches exactly:
  prep="v2": one fluid-sensitive series per plane, 256 px direct resize   (make_npy_v2.py)
  prep="v3": six slots (plane x fluid), 384 px, physical-square padding,   (make_npy_v3.py)
             at most 48 slices per series
Both put volumes in patient (LPS) axes; right knees are then mirrored to left. The side comes
from the DICOM Laterality tag when present, else from the coronal left/right classifier.
Used by the Kaggle notebook and by the local parity check.

This is the speed-optimised version (see infer_opt/REPORT.md). Same predictions as the first version, less work:
  workers  v3 volumes are built by an exact shortcut (integer histogram for the 1/99 percentiles, a lookup table
           for the window, pixel data read only for the slices that survive the 48-slice cap); a study travels to
           the main process as one uint8 tensor in shared memory instead of pickled numpy volumes.
  GPU      the volume goes to the GPU once; the windows of all staggered bags are gathered there, every distinct
           window is embedded once, studies share backbone batches, and each bag only runs the small attention head.
With the default switches the probabilities are bit-identical to the first version's on the same machine.
"""
import math, os, time, traceback
import cv2, numpy as np, pandas as pd, pydicom, torch, torch.nn as nn, torch.nn.functional as F, timm

L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
     "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
SLOTS = ["Sagittal_FL", "Coronal_FL", "Axial_FL", "Sagittal_NF", "Coronal_NF", "Axial_NF"]
XAXIS = {"Sagittal": 0, "Coronal": 2, "Axial": 2}
CANON = {0: ("Sagittal", (0, 2, 1)), 1: ("Coronal", (1, 2, 0)), 2: ("Axial", (2, 1, 0))}
PREP = {"v2": dict(size=256, pad=False, maxs=0, nf=False), "v3": dict(size=384, pad=True, maxs=48, nf=True)}

# ---- switches of the optimised path (defaults = fastest setting that reproduces the first version bit for bit)
FAST_PREP = True      # v3: integer-histogram percentiles + lookup-table window; the uint8 volumes are bit-identical
LAZY_PIXELS = True    # v3: a series longer than the slice cap reads headers first and pixel data only for the kept
                      # slices. Same volume. The one behaviour change: a corrupt file among the DROPPED slices no
                      # longer discards the whole series (the first version failed the series while decoding it).
IO_THREADS = 16       # v3: per worker, the files of a series are read by a small thread pool. Same bytes, same order.
                      # Kaggle's input mount answers every file slowly (latency, not CPU), threads overlap the waiting.
DEDUP = True          # embed each distinct window of a study once (the staggered bags share about a quarter of them)
GROUP = 4             # studies per GPU batch
CHUNK = 0             # windows per backbone call; 0 = k, the batch shape of the first version
PAD_CHUNK = True      # pad the last chunk of a batch to that shape
BAG_HEAD = True       # attention head one bag at a time, as the first version ran it
                      # These three keep cuDNN/cuBLAS on the code paths of the first version, which is what makes the
                      # probabilities bit-identical. CHUNK=96, PAD_CHUNK=False, BAG_HEAD=False is ~5% faster and
                      # within 1.5e-3 (float16 rounding depends on the batch shape).
INT_CROP = False      # centre crop by slicing instead of grid_sample. OFF because it is NOT equivalent: under
                      # autocast affine_grid builds its grid in float16, the sample points sit up to ~0.1 px off the
                      # pixel centres, and the models were trained that way (probabilities move by up to 1.6e-2).
MULTI_GPU = True      # device="cuda" with several GPUs: the models are spread over them
SHM = True            # workers hand a study over as a shared-memory tensor (pickled numpy if /dev/shm is small)


def _square(a, ps, size):
    h, w = a.shape
    hm, wm = h * ps[0], w * ps[1]
    if hm > wm * 1.01:
        pad = int(round((hm - wm) / ps[1])); a = np.pad(a, ((0, 0), (pad // 2, pad - pad // 2)))
    elif wm > hm * 1.01:
        pad = int(round((wm - hm) / ps[0])); a = np.pad(a, ((pad // 2, pad - pad // 2), (0, 0)))
    return cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA if a.shape[0] > size else cv2.INTER_LINEAR)


def _load_series_ref(files, prep):
    """The first version of load_series, kept verbatim: reference, and fallback for anything unusual."""
    c = PREP[prep]
    dss, arrs = [], []
    for p in files:
        ds = pydicom.dcmread(p)
        a = ds.pixel_array.astype(np.float32)
        if a.ndim != 2:
            raise ValueError(f"multiframe {a.shape}")
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            a = a.max() - a
        dss.append(ds); arrs.append(a)
    tags = sorted({str(d.get("Laterality", "")).strip().upper() for d in dss} - {""})
    shapes = {a.shape for a in arrs}
    if len(shapes) > 1:
        common = max(shapes, key=lambda s: sum(a.shape == s for a in arrs))
        keep = [i for i, a in enumerate(arrs) if a.shape == common]
        dss, arrs = [dss[i] for i in keep], [arrs[i] for i in keep]
    iop = np.array(dss[0].ImageOrientationPatient, float)
    r, cc = iop[:3], iop[3:]; n = np.cross(r, cc)
    keep = [i for i, d in enumerate(dss) if np.allclose(np.array(d.ImageOrientationPatient, float), iop, atol=1e-3)]
    pos = np.array([float(np.dot(np.array(dss[i].ImagePositionPatient, float), n)) for i in keep])
    keep = [keep[i] for i in np.argsort(pos)]
    if c["maxs"] and len(keep) > c["maxs"]:
        keep = [keep[i] for i in np.linspace(0, len(keep) - 1, c["maxs"]).round().astype(int)]
    vol = np.stack([arrs[i] for i in keep])
    if c["pad"]:                                           # v3: window on raw, pad to physical square, resize
        lo, hi = np.percentile(vol, [1, 99])
        vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
        try:
            ps = [float(x) for x in dss[0].PixelSpacing]
        except Exception:
            ps = [1.0, 1.0]
        vol = np.stack([_square(s, ps, c["size"]) for s in vol])
    dirs = [n, cc, r]
    ax = [int(np.argmax(np.abs(d))) for d in dirs]
    if len(set(ax)) < 3:
        raise ValueError("oblique_ambiguous")
    for k, d in enumerate(dirs):
        if d[ax[k]] < 0:
            vol = np.flip(vol, axis=k)
    plane, target = CANON[ax[0]]
    vol = vol.transpose([ax.index(t) for t in target])
    zaxis = target.index(2)
    if zaxis > 0:
        vol = np.flip(vol, axis=zaxis)
    vol = np.ascontiguousarray(vol)
    if not c["pad"]:                                       # v2: orient, direct resize, then window
        vol = np.stack([cv2.resize(s, (c["size"], c["size"]), interpolation=cv2.INTER_AREA) for s in vol])
        lo, hi = np.percentile(vol, [1, 99])
        vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    return (vol * 255).astype(np.uint8), plane, tags



class _Slow(Exception):
    """Input outside what the shortcut handles exactly: let the reference path decide."""


_PCT = np.percentile(np.zeros(8, np.float32), [1, 99]).dtype.type      # scalar type np.percentile returns here


_IO = {}


def _read_all(files, stop):
    """[pydicom.dcmread(p, stop_before_pixels=stop) for p in files], with the file reads overlapping in threads."""
    if IO_THREADS <= 1 or len(files) < 2:
        return [pydicom.dcmread(p, stop_before_pixels=stop) for p in files]
    if _IO.get("pid") != os.getpid():                      # one pool per process, made after the fork
        from concurrent.futures import ThreadPoolExecutor
        _IO.update(pid=os.getpid(), pool=ThreadPoolExecutor(IO_THREADS))
    return list(_IO["pool"].map(lambda p: pydicom.dcmread(p, stop_before_pixels=stop), files))


def _pix(ds):
    a = ds.pixel_array
    if a.ndim != 2 or a.dtype.kind not in "iu" or a.dtype.itemsize > 2:   # 8/16-bit integers are exact in float32
        raise _Slow
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        a = a.astype(np.int32); a = a.max() - a
    return a


def _percentiles(ints, base, top):
    """np.percentile(volume, [1, 99]) from an integer histogram. Returns None unless the answer is certain: every
    order statistic the interpolation could touch must sit in one histogram bin (then it is that value exactly)."""
    h = np.zeros(top + 1, np.int64)
    for a in ints:
        h += np.bincount(a.ravel(), minlength=top + 1)
    cum = np.cumsum(h); n = int(cum[-1]); out = []
    for q in (0.01, 0.99):
        r = int(math.floor((n - 1) * q))
        v = int(np.searchsorted(cum, max(r - 1, 0), side="right"))
        if v != int(np.searchsorted(cum, min(r + 2, n - 1), side="right")):
            return None
        out.append(_PCT(v + base))
    return out


def _load_series_fast(files, c):
    """v3 volume, bit-identical to _load_series_ref, without the float copies of the raw stack."""
    lazy = bool(LAZY_PIXELS and c["maxs"] and len(files) > c["maxs"])
    dss = _read_all(files, lazy)
    if lazy:
        if any(int(d.get("NumberOfFrames", 1) or 1) != 1 or int(d.get("SamplesPerPixel", 1) or 1) != 1 for d in dss):
            raise _Slow
        arrs = None; shp = [(int(d.Rows), int(d.Columns)) for d in dss]
    else:
        arrs = [_pix(d) for d in dss]; shp = [a.shape for a in arrs]
    tags = sorted({str(d.get("Laterality", "")).strip().upper() for d in dss} - {""})
    src = list(range(len(files)))
    if len(set(shp)) > 1:
        common = max(set(shp), key=lambda s: sum(x == s for x in shp))
        src = [i for i in src if shp[i] == common]
        dss = [dss[i] for i in src]
    iop = np.array(dss[0].ImageOrientationPatient, float)
    r, cc = iop[:3], iop[3:]; n = np.cross(r, cc)
    keep = [i for i, d in enumerate(dss) if np.allclose(np.array(d.ImageOrientationPatient, float), iop, atol=1e-3)]
    pos = np.array([float(np.dot(np.array(dss[i].ImagePositionPatient, float), n)) for i in keep])
    keep = [keep[i] for i in np.argsort(pos)]
    if c["maxs"] and len(keep) > c["maxs"]:
        keep = [keep[i] for i in np.linspace(0, len(keep) - 1, c["maxs"]).round().astype(int)]
    dirs = [n, cc, r]
    ax = [int(np.argmax(np.abs(d))) for d in dirs]
    if len(set(ax)) < 3:
        raise _Slow                                        # the reference path raises "oblique_ambiguous"
    if lazy:
        ints = []
        for i, d in zip(keep, _read_all([files[src[i]] for i in keep], False)):
            a = _pix(d)
            if a.shape != shp[src[i]]:
                raise _Slow
            ints.append(a)
    else:
        ints = [arrs[src[i]] for i in keep]
    base = min(int(a.min()) for a in ints)
    if base < 0:
        ints = [a.astype(np.int32) - base for a in ints]
    else:
        base = 0
    top = max(int(a.max()) for a in ints)
    lohi = _percentiles(ints, base, top)
    if lohi is None:                                       # rare: interpolation between two values, ask numpy
        lohi = np.percentile(np.stack([(a + base).astype(np.float32) for a in ints]), [1, 99])
    lo, hi = lohi
    lut = np.clip((np.arange(base, base + top + 1).astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    try:
        ps = [float(x) for x in dss[0].PixelSpacing]
    except Exception:
        ps = [1.0, 1.0]
    h, w = ints[0].shape; hm, wm = h * ps[0], w * ps[1]; pr = pc = 0       # same padding as _square
    if hm > wm * 1.01:
        pc = int(round((hm - wm) / ps[1]))
    elif wm > hm * 1.01:
        pr = int(round((wm - hm) / ps[0]))
    buf = np.zeros((h + pr, w + pc), lut.dtype); view = buf[pr // 2: pr // 2 + h, pc // 2: pc // 2 + w]
    interp = cv2.INTER_AREA if buf.shape[0] > c["size"] else cv2.INTER_LINEAR
    vol = np.empty((len(ints), c["size"], c["size"]), np.uint8)
    for j, a in enumerate(ints):
        np.take(lut, a, out=view, mode="clip")           # indices are in range by construction
        vol[j] = (cv2.resize(buf, (c["size"], c["size"]), interpolation=interp) * 255).astype(np.uint8)
    for k, d in enumerate(dirs):
        if d[ax[k]] < 0:
            vol = np.flip(vol, axis=k)
    plane, target = CANON[ax[0]]
    vol = vol.transpose([ax.index(t) for t in target])
    zaxis = target.index(2)
    if zaxis > 0:
        vol = np.flip(vol, axis=zaxis)
    return np.ascontiguousarray(vol), plane, tags


def load_series(files, prep):
    """files: paths of one series' .dcm -> (uint8 volume in canonical LPS order, geometric plane, laterality tags)."""
    if FAST_PREP and PREP[prep]["pad"]:
        try:
            return _load_series_fast(files, PREP[prep])
        except Exception:
            pass                                           # the reference path returns the volume or raises its own error
    return _load_series_ref(files, prep)


def load_study(study, series_df, root, prep):
    """-> ({slot: volume}, laterality tags, errors). series_df: rows of *_series.csv for this study."""
    c = PREP[prep]
    se = series_df.copy()
    se["files"] = [sorted(os.path.join(root, study, s, f) for f in os.listdir(os.path.join(root, study, s)) if f.endswith(".dcm"))
                   if os.path.isdir(os.path.join(root, study, s)) else [] for s in se.SeriesInstanceUID]
    se["n"] = se.files.map(len)
    se = se[se.n > 0]
    if not c["nf"]:
        se = se[se.Fluid_Sensitive == 1]
    se = se.sort_values("n", ascending=False, kind="stable").drop_duplicates(["Anatomical_Plane", "Fluid_Sensitive"])
    vols, tags, errs = {}, [], []
    for r in se.itertuples():
        try:
            v, plane, t = load_series(r.files, prep)
            slot = f"{plane}_{'FL' if r.Fluid_Sensitive else 'NF'}"
            if slot not in vols:
                vols[slot] = v; tags += t
        except Exception as e:
            errs.append(f"{r.SeriesInstanceUID}: {e!r}"[:160])
    return vols, tags, errs


class KneeMIL(nn.Module):
    """Same network as train_mil.py (backbone + per-finding attention pooling)."""
    def __init__(s, arch="efficientnet_b0", res=256, d=512, full=False):
        super().__init__()
        s.res, s.vit, s.full = res, arch.startswith("vit"), full     # full: resize the whole field instead of a native crop
        s.crop = (0.0, 0.0)                                # optional test-time crop centre offset (normalised coordinates)
        s.b = timm.create_model(arch, pretrained=False, num_classes=0, **(dict(img_size=res) if s.vit else {}))
        fdim = s.b.num_features * (2 if s.vit else 1)
        s.register_buffer("mean", torch.zeros(1, 3, 1, 1)); s.register_buffer("std", torch.ones(1, 3, 1, 1))
        s.proj = nn.Sequential(nn.Linear(fdim, d), nn.GELU())
        s.slot = nn.Embedding(len(SLOTS), d)
        s.pos = nn.Sequential(nn.Linear(1, 128), nn.GELU(), nn.Linear(128, d))
        s.attn = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Linear(256, 12))
        s.cls = nn.Parameter(torch.zeros(12, d)); s.bias = nn.Parameter(torch.zeros(12))
        s.drop = nn.Dropout(0.2)
    def forward(s, x, slot, pos):                          # x uint8 (B, K, 3, H, W)
        B, K = x.shape[:2]
        x = x.flatten(0, 1).float().div(255)
        H = x.shape[-1]
        f0 = 1.0 if s.full else (min(1.0, s.res / H) if H > s.res else 1.0)
        th = torch.tensor([[f0, 0, s.crop[0]], [0, f0, s.crop[1]]], device=x.device).expand(len(x), 2, 3)
        x = F.grid_sample(x, F.affine_grid(th, (len(x), 3, s.res, s.res), align_corners=False),
                          mode="bilinear", padding_mode="zeros", align_corners=False)
        x = (x - s.mean) / s.std
        if s.vit:
            t = s.b.forward_features(x); f = torch.cat([t[:, 0], t[:, s.b.num_prefix_tokens:].mean(1)], 1)
        else:
            f = s.b(x.contiguous(memory_format=torch.channels_last))
        h = s.proj(f.float().view(B, K, -1)) + s.slot(slot) + s.pos(pos.unsqueeze(-1))
        w = s.attn(h).softmax(1)
        z = torch.einsum("bkj,bkd->bjd", w, h)
        return (z * s.cls).sum(-1) + s.bias

    # The same computation in three steps, so that run() can share work between bags, models and studies.
    def slices(s, v):
        """v uint8 (N, H, W) -> the network-resolution version of every slice: what forward()'s grid_sample makes
        of each channel. uint8 when the sampling grid is exactly a crop (or the identity), else float in 0..1."""
        H = v.shape[-1]
        f0 = 1.0 if s.full else (min(1.0, s.res / H) if H > s.res else 1.0)
        if INT_CROP and s.crop == (0.0, 0.0) and v.shape[-2] == H and (H - s.res) % 2 == 0 and ((f0 < 1.0 and not s.full) or H == s.res):
            o = (H - s.res) // 2                           # grid point i sits on input pixel centre i + o
            return v[:, o:o + s.res, o:o + s.res]
        out = []
        for i in range(0, len(v), 256):
            x = v[i:i + 256, None].float().div(255)
            th = torch.tensor([[f0, 0, s.crop[0]], [0, f0, s.crop[1]]], device=x.device).expand(len(x), 2, 3)
            out.append(F.grid_sample(x, F.affine_grid(th, (len(x), 1, s.res, s.res), align_corners=False),
                                     mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0])
        return torch.cat(out)

    def embed(s, x):
        """x (n, 3, res, res) windows gathered from slices() -> backbone features (n, fdim), float32."""
        x = x.float().div(255) if x.dtype == torch.uint8 else x
        x = (x - s.mean) / s.std
        if s.vit:
            t = s.b.forward_features(x); f = torch.cat([t[:, 0], t[:, s.b.num_prefix_tokens:].mean(1)], 1)
        else:
            f = s.b(x.contiguous(memory_format=torch.channels_last))
        return f.float()

    def head(s, f, slot, pos):
        """f (B, K, fdim) window features of B bags -> logits (B, 12)."""
        h = s.proj(f) + s.slot(slot) + s.pos(pos.unsqueeze(-1))
        w = s.attn(h).softmax(1)
        z = torch.einsum("bkj,bkd->bjd", w, h)
        return (z * s.cls).sum(-1) + s.bias


STAGGER = (-0.375, -0.125, 0.125, 0.375)                 # four offset bags; their predictions are averaged


def make_bag(vols, side, k, only=None, shift=0.0):
    """Evenly spaced 3-slice windows over every slot, mirrored for right knees. Same as Bags(train=False).
    shift != 0 moves every window centre by that fraction of the spacing (staggered bags)."""
    names = sorted(n for n in vols if only is None or n in only)
    S = len(names)
    counts = np.full(S, k // S); counts[: k - counts.sum()] += 1
    xs, sl, ps = [], [], []
    for name, m in zip(names, counts):
        v = vols[name]
        if side == "R":
            v = np.flip(v, axis=XAXIS[name.split("_")[0]])
        n = len(v)
        if shift == 0:
            c = np.linspace(0, n - 1, m).round().astype(int)
        else:
            c = np.clip(np.floor((np.arange(m) + 0.5 + shift) / m * n), 0, n - 1).astype(int)
        idx = np.clip(np.stack([c - 1, c, c + 1], 1), 0, n - 1)
        xs.append(np.ascontiguousarray(v[idx.ravel()]).reshape(m, 3, *v.shape[1:]))
        sl.append(np.full(m, SLOTS.index(name))); ps.append(c / max(n - 1, 1))
    return (torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(sl)),
            torch.from_numpy(np.concatenate(ps).astype(np.float32)))


def bag_plan(names, lens, k, only=None, shifts=(0.0,)):
    """make_bag as indices. names/lens: every slot of the study, sorted by name, with its slice count (the volumes are
    concatenated in that order). Returns (win, inv, slot, pos):
      win  (U, 3) int64   slice rows of the windows to embed (each distinct window once when DEDUP)
      inv  (B, k) int64   for bag b, position j: its row in win
      slot (B, k) int64, pos (B, k) float32   exactly make_bag's second and third outputs, per shift."""
    off = dict(zip(names, np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(int))); n_of = dict(zip(names, lens))
    use = sorted(n for n in names if only is None or n in only)
    S = len(use)
    counts = np.full(S, k // S); counts[: k - counts.sum()] += 1
    key, sl, ps = [], [], []
    for shift in shifts:
        kk, ss, pp = [], [], []
        for name, m in zip(use, counts):
            n = int(n_of[name])
            if shift == 0:
                c = np.linspace(0, n - 1, m).round().astype(int)
            else:
                c = np.clip(np.floor((np.arange(m) + 0.5 + shift) / m * n), 0, n - 1).astype(int)
            kk.append(off[name] + c); ss.append(np.full(m, SLOTS.index(name))); pp.append(c / max(n - 1, 1))
        key.append(np.concatenate(kk)); sl.append(np.concatenate(ss)); ps.append(np.concatenate(pp).astype(np.float32))
    key = np.stack(key).astype(np.int64)                    # (B, k) global row of every window centre
    lo = np.zeros(int(np.sum(lens)), np.int64); hi = np.zeros_like(lo)     # first / last row of the centre's slot
    for name in names:
        lo[off[name]: off[name] + n_of[name]] = off[name]; hi[off[name]: off[name] + n_of[name]] = off[name] + n_of[name] - 1
    if DEDUP:
        cen, inv = np.unique(key, return_inverse=True); inv = inv.reshape(key.shape)
    else:
        cen, inv = key.ravel(), np.arange(key.size).reshape(key.shape)
    win = np.clip(np.stack([cen - 1, cen, cen + 1], 1), lo[cen][:, None], hi[cen][:, None])
    return win, inv.astype(np.int64), np.stack(sl).astype(np.int64), np.stack(ps)


def mirror(v, names, lens):
    """Right knee -> left, as make_bag does it: v (N, H, W) is the concatenation of the slots in `names` order."""
    out, o = [], 0
    for name, n in zip(names, lens):
        out.append(torch.flip(v[o:o + n], dims=(XAXIS[name.split("_")[0]],))); o += n
    return torch.cat(out)


def side_input(vols, size=256):
    """Coronal fluid-sensitive slices at 50/65/80% of the stack, as latclf.py was trained."""
    v = vols.get("Coronal_FL")
    if v is None:
        return None
    n = len(v)
    x = v[[int(n * q) for q in (0.5, 0.65, 0.8)]].astype(np.float32) / 255
    if x.shape[-1] != size:
        x = np.stack([cv2.resize(s, (size, size), interpolation=cv2.INTER_AREA) for s in x])
    return torch.from_numpy((x - 0.45) / 0.225)


class StudySet(torch.utils.data.Dataset):
    """DICOM decoding happens in the DataLoader workers; one item = one study, never raises.
    A study is one uint8 tensor (all slots concatenated in name order): tensors cross to the main process through
    shared memory, numpy volumes would be pickled through a pipe."""
    def __init__(s, ids, series, root, prep, k, only=None, shm=True):
        s.ids, s.root, s.prep, s.k, s.only, s.shm = list(ids), root, prep, k, only, shm
        s.groups = {u: g for u, g in series.groupby("StudyInstanceUID")}
    def __len__(s): return len(s.ids)
    def __getitem__(s, i):
        u = s.ids[i]; t0 = time.perf_counter()
        try:
            vols, tags, errs = load_study(u, s.groups[u], s.root, s.prep)
            if not vols:
                return u, None, "no usable series: " + "; ".join(errs)
            t = {x[:1] for x in tags if x[:1] in "LR"}
            names = sorted(vols)
            vol = np.concatenate([vols[n] for n in names])
            return u, dict(vol=torch.from_numpy(vol) if s.shm else vol, names=names, lens=[len(vols[n]) for n in names],
                           tag=t.pop() if len(t) == 1 else "", lat=side_input(vols), sec=time.perf_counter() - t0), "; ".join(errs)
        except Exception as e:
            return u, None, repr(e)[:200] + " | " + traceback.format_exc()[-200:]


def _worker_init(_):
    cv2.setNumThreads(1)                                   # one worker per core already; same pixels, no oversubscription


def _identity(x):
    return x


def _shm_ok(need=2 << 30):
    """Shared-memory tensors need room in /dev/shm (about 0.5 GB at worst here); a full one kills workers."""
    try:
        st = os.statvfs("/dev/shm"); return st.f_bavail * st.f_frsize >= need
    except Exception:
        return False


def _studies(ids, series, root, prep, k, only, workers):
    """Yields (study, item, error) in order. If the worker pool itself dies (a worker killed for memory, shared memory
    full) the studies not yet delivered are decoded again: first by a new pool without shared memory, then in this process."""
    plans = [(workers, bool(SHM and workers and _shm_ok()))]
    if plans[0][1]: plans.append((workers, False))
    if workers: plans.append((0, False))
    done = 0
    for w, shm in plans:
        try:
            if w == 0 and workers:
                cv2.setNumThreads(1)                       # no OpenCV thread pool in a process that may fork again
            dl = torch.utils.data.DataLoader(StudySet(ids[done:], series, root, prep, k, only, shm), batch_size=None,
                                             num_workers=w, collate_fn=_identity, multiprocessing_context="fork" if w else None,
                                             worker_init_fn=_worker_init if w else None)
            for x in dl:
                done += 1; yield x
            return
        except Exception as e:
            print(f"LOADER FAILED after {done} studies ({w} workers, shared memory {shm}): {e!r}"[:300], flush=True)


def _devices(device, n_models):
    if MULTI_GPU and device == "cuda" and n_models > 1 and torch.cuda.device_count() > 1:
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return [device]


def _predict(group, nets, devs, k, only, shifts, chunk):
    """group: list of (item, side). -> probabilities (studies, models, 12), bags averaged as in the first version.
    Per device: volumes up once, mirrored there; per model: every distinct window through the backbone once, in chunks
    that mix studies; then the attention head on each bag. Nothing is read back before all models are queued, so with
    two GPUs both work at the same time."""
    B = len(shifts); plans, o, u = [], 0, 0
    for item, side in group:
        win, inv, sl, ps = bag_plan(item["names"], item["lens"], k, only, shifts)
        plans.append((win + o, inv + u, sl, ps)); o += int(sum(item["lens"])); u += len(win)
    win = np.concatenate([p[0] for p in plans]); inv = np.concatenate([p[1] for p in plans])
    sl = np.concatenate([p[2] for p in plans]); ps = np.concatenate([p[3] for p in plans])
    out = [None] * len(nets)
    for d in dict.fromkeys(devs):
        with torch.no_grad(), torch.autocast(torch.device(d).type, enabled=torch.device(d).type == "cuda"):
            vs = [torch.as_tensor(item["vol"]).to(d) for item, _ in group]
            v = torch.cat([mirror(x, item["names"], item["lens"]) if side == "R" else x for x, (item, side) in zip(vs, group)])
            win_d, inv_d = torch.from_numpy(win).to(d), torch.from_numpy(inv).to(d)
            sl_d, ps_d = torch.from_numpy(sl).to(d), torch.from_numpy(ps).to(d)
            for m, net in enumerate(nets):
                if devs[m] != d:
                    continue
                bank = net.slices(v); f = []
                for i in range(0, len(win_d), chunk):
                    w = win_d[i:i + chunk]; n = len(w)
                    if PAD_CHUNK and n < chunk:
                        w = torch.cat([w, w[:1].expand(chunk - n, -1)])
                    f.append(net.embed(bank[w])[:n])
                f = torch.cat(f)
                if BAG_HEAD:
                    z = torch.cat([net.head(f[inv_d[j:j + 1]], sl_d[j:j + 1], ps_d[j:j + 1]) for j in range(len(inv_d))])
                else:
                    z = net.head(f[inv_d], sl_d, ps_d)
                out[m] = torch.sigmoid(z.float())                                         # (studies * B, 12)
    P = np.stack([x.cpu().numpy().reshape(len(group), B, 12) for x in out], 1)       # (studies, models, B, 12)
    acc = 0
    for b in range(B):                                     # same summation order as the first version
        acc = acc + P[:, :, b]
    return acc / B


def scanner_keys(ids, series, root, known):
    """Scanner key per study as in study_meta.csv: per series the first file's header, per study the most common non-empty value."""
    import collections
    T = ["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "ReceiveCoilName"]; out = []
    for u in ids:
        vals = {t: [] for t in T}
        for s_ in series.SeriesInstanceUID[series.StudyInstanceUID == u]:
            d = os.path.join(root, u, str(s_))
            try:
                f = sorted(x for x in os.listdir(d) if x.endswith(".dcm"))[0]
                ds = pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True)
                for t in T: vals[t].append(str(ds.get(t, "")).strip())
            except Exception:
                pass
        v = {t: (collections.Counter(x for x in vals[t] if x).most_common(1)[0][0] if any(vals[t]) else "") for t in T}
        try: fs = str(float(v["MagneticFieldStrength"])) if v["MagneticFieldStrength"] else "nan"
        except ValueError: fs = "nan"
        k_ = "|".join([v["Manufacturer"] or "?", v["ManufacturerModelName"] or "?", fs, v["ReceiveCoilName"] or "?"])
        out.append(k_ if k_ in known else "rare")
    return out


def run(test_csv, series_csv, root, models, latclf_path, out_csv, prior, prep="v2", k=48, only=None,
        workers=4, budget_s=8 * 3600, device="cuda", stagger=False, syn_eff=0.0, site_table=None, families=None,
        deadline=None, raw_out=None):
    """models: list of (arch, res, weights_path[, full[, weight]]). Writes out_csv.
    One model: its probabilities. Several: per finding, each model's scores are ranked over the test set and the
    ranks averaged (AUC only depends on order). A failed study gets `prior` (one model) or the neutral 0.5 rank.
    stagger: every model sees four offset bags per study and their probabilities are averaged.
    syn_eff: final Synovitis = (1 - syn_eff) * rank(Synovitis) + syn_eff * rank(Effusion); other findings untouched."""
    t0 = time.time()
    end = t0 + budget_s if deadline is None else min(t0 + budget_s, float(deadline))   # stop reading studies at this time
    ids = pd.read_csv(test_csv).StudyInstanceUID.tolist()
    series = pd.read_csv(series_csv)
    devs = _devices(device, len(models))
    devs = [devs[i % len(devs)] for i in range(len(models))]
    nets = []
    for i, m in enumerate(models):
        arch, res, wpath = m[:3]
        net = KneeMIL(arch, res, full=bool(m[3]) if len(m) > 3 else False)
        net.load_state_dict(torch.load(wpath, map_location="cpu"))
        if len(m) > 5 and m[5]: net.crop = (float(m[5][0]), float(m[5][1]))   # optional 6th item: test-time crop centre
        nets.append(net.to(devs[i]).to(memory_format=torch.channels_last).eval())
    lat = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    lat.load_state_dict(torch.load(latclf_path, map_location="cpu")); lat = lat.to(devs[0]).eval()
    rows, stats = {u: None for u in ids}, dict(ok=0, failed=0, side_tag=0, side_clf=0, side_none=0, series_errors=0)
    tm = dict(decode=0.0, gpu=0.0)                         # seconds: in workers (summed over workers) / in the GPU stage
    kind = torch.device(device).type; amp = kind == "cuda"
    shifts = STAGGER if stagger else (0.0,)
    pend = []                                              # (study, item, side) waiting for the next GPU batch

    def flush():
        if not pend:
            return
        tg = time.time()
        try:
            for (u, _, _), p in zip(pend, _predict([x[1:] for x in pend], nets, devs, k, only, shifts, CHUNK or k)):
                rows[u] = p
            stats["ok"] += len(pend)
        except Exception:                                  # one bad study must not take its neighbours down:
            for u, item, side in pend:                     # each study alone, then once more with small chunks
                for chunk in (CHUNK or k, max(1, (CHUNK or k) // 4)):
                    try:
                        rows[u] = _predict([(item, side)], nets, devs, k, only, shifts, chunk)[0]
                        stats["ok"] += 1; break
                    except Exception as e:
                        err = e
                        if kind == "cuda": torch.cuda.empty_cache()
                else:
                    rows[u] = None; stats["failed"] += 1; print("FAILED", u, repr(err)[:200], flush=True)
        pend.clear(); tm["gpu"] += time.time() - tg

    gen = _studies(ids, series, root, prep, k, only, workers)
    for n, (u, item, err) in enumerate(gen):
        if time.time() >= end:                             # out of time: the rest keep the fallback; do not decode them
            stats["budget_cut"] = len(ids) - n
            print(f"TIME BUDGET reached after {n} studies: {len(ids) - n} studies keep the fallback", flush=True)
            break
        if err: stats["series_errors"] += 1
        if item is not None: tm["decode"] += item.get("sec", 0.0)
        if item is not None:
            try:
                with torch.no_grad(), torch.autocast(kind, enabled=amp):
                    side = item["tag"]
                    if side: stats["side_tag"] += 1
                    elif item["lat"] is not None:
                        side = "R" if torch.sigmoid(lat(item["lat"][None].to(devs[0])).float()).item() > 0.5 else "L"
                        stats["side_clf"] += 1
                    else:
                        side = "L"; stats["side_none"] += 1
                pend.append((u, item, side))
            except Exception as e:
                stats["failed"] += 1; print("FAILED", u, repr(e)[:200], flush=True)
            if len(pend) >= GROUP:
                flush()
        else:
            stats["failed"] += 1
            if n < 20 or item is not None: print("FALLBACK", u, (err or "time budget")[:200], flush=True)
        if n % 100 == 0:
            flush()
            print(f"{n + 1}/{len(ids)} studies, {time.time() - t0:.0f}s, {stats}", flush=True)
    gen.close()
    flush()
    M = len(nets)
    out = np.tile(np.asarray(prior, np.float32) if M == 1 else np.full(12, 0.5, np.float32), (len(ids), 1))
    okk = [i for i, u in enumerate(ids) if rows[u] is not None]
    if okk:
        A = np.stack([rows[ids[i]] for i in okk])                 # (studies, models, 12)
        if raw_out:                                          # raw per-model probabilities, for parity checks only
            np.savez(raw_out, ids=np.array([ids[i] for i in okk]), A=A)
        wts = np.array([float(m[4]) if len(m) > 4 else 1.0 for m in models]); wts = wts / wts.sum()   # optional model weights (5th item)
        _keys = [m[2] for m in models]                         # [tta-group] crop views of one checkpoint share its weights file
        if len(set(_keys)) < len(_keys):
            _uniq = list(dict.fromkeys(_keys)); _gi = [[i for i, k_ in enumerate(_keys) if k_ == u_] for u_ in _uniq]
            A = np.stack([A[:, g].mean(1) for g in _gi], 1)                          # probability average over views, before ranking
            wts = np.array([wts[g].sum() for g in _gi]); wts = wts / wts.sum()
            families = [families[g[0]] for g in _gi] if families else families
            M = len(_gi)
            print(f"[tta-group] {len(_keys)} members -> {M} models (views averaged before ranking)", flush=True)
        if site_table and time.time() > end + 1800:
            print("TIME BUDGET: no time left for the scanner-header pass; Synovitis left uncentred", flush=True)
        elif site_table:                                     # Synovitis centred per scanner (offsets from out-of-fold training predictions)
            A = A.copy(); keys = scanner_keys([ids[i] for i in okk], series, root, site_table["keys"])
            for m in range(M):
                off = np.array([site_table["families"][families[m]].get(k_, 0.0) for k_ in keys], np.float64)
                p = np.clip(A[:, m, L.index("Synovitis")].astype(np.float64), 1e-6, 1 - 1e-6)
                A[:, m, L.index("Synovitis")] = 1 / (1 + np.exp(-(np.log(p / (1 - p)) - off)))
            print("site centring: scanner keys", dict(pd.Series(keys).value_counts().head(8)), "| rare", keys.count("rare"), flush=True)
        out[okk] = A[:, 0] if M == 1 else sum(w * pd.DataFrame(A[:, m]).rank(pct=True).values for m, w in enumerate(wts))
    if syn_eff and okk:
        r = pd.DataFrame(out[okk]).rank(pct=True).values
        out[okk, L.index("Synovitis")] = (1 - syn_eff) * r[:, L.index("Synovitis")] + syn_eff * r[:, L.index("Effusion")]
    sub = pd.DataFrame(out, columns=L); sub.insert(0, "StudyInstanceUID", ids)
    sub.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {len(sub)} rows in {time.time() - t0:.0f}s | {stats}", flush=True)
    print(f"timing per study: decode {tm['decode'] / max(len(ids), 1):.2f}s of worker time ({workers} workers), GPU stage "
          f"{tm['gpu'] / max(len(ids), 1):.2f}s, wall {(time.time() - t0) / max(len(ids), 1):.2f}s | devices {sorted(set(devs))}", flush=True)
    return sub, stats
