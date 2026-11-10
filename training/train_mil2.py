"""Study-level trainer: 3-slice windows over every series, pooled per finding.

Each study becomes a bag of k windows (3 neighbouring slices -> 3 channels) spread over all
of its series. One shared backbone embeds every window; an attention layer with separate
weights for each of the 12 findings pools the bag, so a ligament tear can draw on two slices
while osteoarthritis draws on many. Soft targets = LLM grade / 4. Right knees are mirrored
to left at load (laterality.csv); no flips anywhere, side is signal.

Works on npy2 (Sagittal.npy ...) and npy3 (Sagittal_FL.npy, Sagittal_NF.npy ...).
Reports AUC on the held-out fold (silver, positive = grade >= mild) and on the gold 58,
keeps the best epoch and the weight average of the best three.
"""
import argparse, json, math, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=".")
ap.add_argument("--npy", default="npy2")
ap.add_argument("--arch", default="efficientnet_b0")
ap.add_argument("--weights", default="effb0_pretrained.pth")
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--k", type=int, default=48)
ap.add_argument("--k_eval", type=int, default=48)
ap.add_argument("--slots", default="")                    # comma list to restrict, e.g. Sagittal_FL,Coronal_FL
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--epochs", type=int, default=14)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lr_head", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.02)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--warmup_ep", type=float, default=0)       # 0 = 10% of training
ap.add_argument("--targets", default="grade", choices=["grade", "prob", "rubric", "binary"])
ap.add_argument("--full", action="store_true")              # resize the full field instead of a native crop
ap.add_argument("--fl_share", type=float, default=0)        # share of windows reserved for fluid-sensitive series (0 = even split)
ap.add_argument("--series_drop", type=float, default=0)     # training: chance of dropping each non-fluid series
ap.add_argument("--seed", type=int, default=0)
# head / loss experiments (hail-mary review, 10-10)
ap.add_argument("--branches", type=int, default=1)          # ACMIL: attention branches per finding
ap.add_argument("--div", type=float, default=0.0)           # ACMIL: penalty on similarity between a finding's branches
ap.add_argument("--smooth", type=float, default=0.0)        # AttriMIL-style: neighbouring windows of a series should score alike
ap.add_argument("--llr", type=float, default=0.0)           # large-loss rejection: final share of "absent" cells ignored per finding
ap.add_argument("--ext_silent", action="store_true")        # use --ext_labels only on cells the report is silent on (grade 0, no mention): there the target becomes the external value
ap.add_argument("--unm_weight", type=float, default=1.0)     # loss weight of cells the report does not mention (grade 0 and silent in --mention); 1 = as before
ap.add_argument("--mention", default="mention.csv")
ap.add_argument("--cell_weights", default="")                # CSV of per-cell loss weights (StudyInstanceUID + 12 findings), normalised per finding like --unm_weight         # 1 = the report asserts the finding (either way), 0 = silent
ap.add_argument("--aux_mention", default="")              # CSV (StudyInstanceUID + 12; 1 = report states the finding either way, 0 = silent): auxiliary head target
ap.add_argument("--aux_w", type=float, default=0.2)          # weight of the auxiliary mention loss
ap.add_argument("--domain_balance", action="store_true")   # equal total sampling weight for competition studies and fastMRI ("fm_") studies; same steps per epoch
ap.add_argument("--eval_shift", default="")                # eval only: "dx,dy" offset of the centre crop in normalised coordinates (training jitter is about +-0.17)
ap.add_argument("--no_mirror", action="store_true")         # do not mirror right knees to left
ap.add_argument("--fov_frac", type=float, default=0.0)      # global windows read this fraction of the cached image (e.g. 0.833 of a 512 cache = the 133 mm of the 384 cache); 0 = native crop as before
ap.add_argument("--local_k", type=int, default=0)           # extra windows per study read as native-resolution local crops (tiles) of the same slices
ap.add_argument("--local_frac", type=float, default=0.625)  # fraction of the cached image a local crop covers (0.625 of 512 = 320 px native, about 100 mm)
ap.add_argument("--hier", action="store_true")
ap.add_argument("--site_offset", action="store_true")       # training only: a learned per-scanner, per-finding logit offset soaks up each site's reporting habit (code audit 2.2)              # pool windows within each series first, then pool the series, per finding
ap.add_argument("--all_data", action="store_true")          # train on every fold incl. --fold; its "validation" is then in-sample (sanity only). Deploy the EMA weights.
ap.add_argument("--train_folds", default="")                # train only on these folds, e.g. "1,2" (default: every fold except --fold)
ap.add_argument("--pred_folds", default="")                 # after training, also save predictions for these folds (teacher for out-of-fold targets)
ap.add_argument("--all_sides", action="store_true")         # also train on studies whose side comes only from the image classifier (p_right)
ap.add_argument("--drop_path", type=float, default=0.0)     # stochastic depth in the backbone
ap.add_argument("--bagmix", type=float, default=0.0)        # chance per step of splicing window embeddings between studies (PseMix-style)
ap.add_argument("--amp", default="fp16", choices=["fp16", "bf16", "fp32"])   # autocast dtype; bf16/fp32 run without the GradScaler
ap.add_argument("--ema", action="store_true")               # keep three EMAs of the weights and score them at the end (no epoch selection)
ap.add_argument("--lr_floor", type=float, default=0.0)      # final learning rate as a share of the peak (0 = decay to ~zero)
ap.add_argument("--ctx", type=int, default=0)               # transformer layers over all window tokens before the pooling (public ConvNeXt reader)
ap.add_argument("--lr_ctx", type=float, default=2e-4)
ap.add_argument("--aug_series", action="store_true")        # one augmentation per series (not per window), and stronger: +-12 deg, +-12% scale
ap.add_argument("--spatial", action="store_true")           # CNN: keep a global vector + a 2x2 grid of local vectors per window
ap.add_argument("--ext_labels", default="")                 # CSV of external label probabilities (StudyInstanceUID + 12 findings)
ap.add_argument("--ext_weight", type=float, default=0.5)    # target = (1 - w) * ours + w * external
ap.add_argument("--export_feats", default="")            # path to trained weights: dump per-finding pooled features and exit
ap.add_argument("--head", default="attn", choices=["attn", "ctran"])   # ctran: label tokens + windows through a small transformer
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--tag", default="mil")
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda"
D = a.data
NPY = a.npy if os.path.isabs(a.npy) else f"{D}/{a.npy}"
OUT = f"{D}/runs/{a.tag}"; os.makedirs(OUT, exist_ok=True)
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
     "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
SLOTS = ["Sagittal_FL", "Coronal_FL", "Axial_FL", "Sagittal_NF", "Coronal_NF", "Axial_NF"]
XAXIS = {"Sagittal": 0, "Coronal": 2, "Axial": 2}        # patient-x axis of each volume
slot_of = lambda f: (f[:-4] if "_" in f else f[:-4] + "_FL")
only = set(a.slots.split(",")) if a.slots else None

tr = pd.read_csv(f"{D}/train.csv").set_index("StudyInstanceUID")
grade = pd.DataFrame([json.loads(l) for l in open(f"{D}/llm_labels_v3.jsonl")]).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[L].astype(int)
# training targets from the LLM grade (0 none, 1 minimal, 2 mild, 3 moderate, 4 severe)
GEN = [0.02, 0.25, 0.80, 0.95, 0.98]                       # probability that the finding is present
HIGH = [0.02, 0.05, 0.10, 0.90, 0.98]                      # host rubric: only high-grade ACL/MCL tears count
AREA = [0.02, 0.10, 0.45, 0.90, 0.98]                      # host rubric: moderate/large area or amount
MAPS = {"grade": {k: [0, .25, .5, .75, 1] for k in L}, "prob": {k: GEN for k in L}, "binary": {k: [0, 0, 1, 1, 1] for k in L},
        "rubric": {**{k: GEN for k in L}, "ACL": HIGH, "MCL": HIGH, "Medial OA": AREA, "Lateral OA": AREA, "PF OA": AREA, "Baker's": AREA}}
llm = pd.DataFrame({k: grade[k].map(dict(enumerate(MAPS[a.targets][k]))) for k in L})
if a.ext_labels:
    ext = pd.read_csv(a.ext_labels if os.path.isabs(a.ext_labels) else f"{D}/{a.ext_labels}").set_index("StudyInstanceUID")[L].reindex(llm.index)
    if a.ext_silent:
        _men = pd.read_csv(a.mention if os.path.isabs(a.mention) else f"{D}/{a.mention}").set_index("StudyInstanceUID")[L].reindex(llm.index).fillna(1)
        _sil = (grade.reindex(llm.index)[L] == 0) & (_men == 0)
        llm = llm.where(~_sil, (1 - a.ext_weight) * llm + a.ext_weight * ext.fillna(llm))
        print(f"targets: {a.ext_labels} at weight {a.ext_weight:.2f} on the {100 * _sil.values.mean():.0f}% of cells the report is silent on; ours elsewhere", flush=True)
    else:
        llm = (1 - a.ext_weight) * llm + a.ext_weight * ext.fillna(llm)
        print(f"targets: {1 - a.ext_weight:.2f} ours + {a.ext_weight:.2f} {a.ext_labels}", flush=True)
RUB = {k: 3 for k in ("ACL", "MCL", "Medial OA", "Lateral OA", "PF OA", "Baker's", "Effusion")}   # host-rubric positives for a second silver view
folds = pd.read_csv(f"{D}/folds.csv").set_index("StudyInstanceUID")
_lat = pd.read_csv(f"{D}/laterality.csv", keep_default_na=False).set_index("StudyInstanceUID")
side = _lat.side.to_dict()
if a.all_sides:
    n0 = sum(v in ("L", "R") for v in side.values())
    for u, p in _lat.p_right.items():
        if side[u] not in ("L", "R") and p != "": side[u] = "R" if float(p) > 0.5 else "L"
    print(f"all_sides: side filled from the classifier for {sum(v in ('L', 'R') for v in side.values()) - n0} studies", flush=True)
ok = lambda u: os.path.isdir(f"{NPY}/{u}")
known = lambda u: side.get(u) in ("L", "R")
tr_ids = [u for u in folds.index[(folds.fold >= 0) & ((folds.fold != a.fold) | a.all_data)] if ok(u) and known(u)]
if a.all_data: print("all_data: training includes the validation fold; its scores below are in-sample", flush=True)
if a.train_folds:
    _tf = {int(x) for x in a.train_folds.split(",")}; assert a.fold not in _tf
    tr_ids = [u for u in tr_ids if folds.fold[u] in _tf]
va_ids = [u for u in folds.index[folds.fold == a.fold] if ok(u)]
go_ids = [u for u in folds.index[folds.gold == 1] if ok(u)]
if a.limit: tr_ids, va_ids = tr_ids[: a.limit], va_ids[: a.limit // 2]
Ytr = llm.loc[tr_ids].values.astype(np.float32)
if a.aux_mention:                                                    # targets (12) + mention targets (12)
    assert not a.cell_weights and a.unm_weight == 1.0 and not a.site_offset and a.head == "attn" and not a.hier, "--aux_mention: unsupported combination"
    _mt = pd.read_csv(a.aux_mention if os.path.isabs(a.aux_mention) else f"{D}/{a.aux_mention}").set_index("StudyInstanceUID")[L].reindex(tr_ids)
    print(f"aux_mention: {a.aux_mention} weight {a.aux_w} | missing rows {int(_mt.isna().any(axis=1).sum())} (filled 1) | mention rate", dict(zip(L, _mt.fillna(1).values.mean(0).round(2))), flush=True)
    Ytr = np.concatenate([Ytr, _mt.fillna(1).values.astype(np.float32)], 1)
RATE = None
if a.cell_weights:                                                   # targets (12) + per-cell loss weights (12) from a file
    assert a.unm_weight == 1.0, "--cell_weights and --unm_weight are exclusive"
    Wtr = pd.read_csv(a.cell_weights if os.path.isabs(a.cell_weights) else f"{D}/{a.cell_weights}").set_index("StudyInstanceUID")[L].reindex(tr_ids).fillna(1).values.astype(np.float32)
    RATE = torch.tensor(Wtr.mean(0)).clamp(min=1e-3)
    print("cell_weights: mean weight per finding", dict(zip(L, Wtr.mean(0).round(3))), flush=True)
    Ytr = np.concatenate([Ytr, Wtr], 1)
elif a.unm_weight != 1.0:                                              # targets (12) + per-cell loss weights (12)
    men = pd.read_csv(a.mention if os.path.isabs(a.mention) else f"{D}/{a.mention}").set_index("StudyInstanceUID")[L].reindex(tr_ids).fillna(1).values
    Wtr = np.where((grade.loc[tr_ids].values == 0) & (men == 0), a.unm_weight, 1.0).astype(np.float32)
    RATE = torch.tensor(Wtr.mean(0)).clamp(min=1e-3)
    print(f"unm_weight {a.unm_weight}: share of unmentioned cells per finding", dict(zip(L, (Wtr != 1).mean(0).round(2))), flush=True)
    Ytr = np.concatenate([Ytr, Wtr], 1)
N_SITES = 0
if a.site_offset:                                                    # scanner key as in study_meta.csv; < 15 training studies -> one shared "rare" key
    _m = pd.read_csv(f"{D}/study_meta.csv").set_index("StudyInstanceUID")
    _fs = lambda v: "nan" if pd.isna(v) or str(v).strip() in ("", "nan") else str(float(v))
    _key = (_m.Manufacturer.fillna("?") + "|" + _m.ManufacturerModelName.fillna("?") + "|" + _m.MagneticFieldStrength.map(_fs) + "|" + _m.ReceiveCoilName.fillna("?"))
    _vc = _key.loc[tr_ids].value_counts(); _common = sorted(_vc[_vc >= 15].index)
    _idx = {k: i + 1 for i, k in enumerate(_common)}; N_SITES = len(_common) + 1
    _sid = np.array([_idx.get(_key.get(u, ""), 0) for u in tr_ids], np.float32)
    print(f"site_offset: {N_SITES - 1} scanners with >= 15 training studies + 1 rare group", flush=True)
    Ytr = np.concatenate([Ytr, _sid[:, None]], 1)                    # rides along as the last target column
Yva = (grade.loc[va_ids].values >= 2).astype(np.float32)                 # silver: mild or worse (kept for continuity)
Yvr = np.stack([(grade.loc[va_ids, k].values >= RUB.get(k, 2)) for k in L], 1).astype(np.float32)   # silver, host-rubric cut-offs
Ygo = (tr.loc[go_ids, L].values > 0.5).astype(np.float32)
print(f"{a.tag}: arch={a.arch} res={a.res} k={a.k}/{a.k_eval} npy={NPY} fold={a.fold} "
      f"train={len(tr_ids)} silver-val={len(va_ids)} gold={len(go_ids)}", flush=True)

class Bags(Dataset):
    def __init__(s, ids, Y, k, train): s.ids, s.Y, s.k, s.train = ids, Y, k, train
    def __len__(s): return len(s.ids)
    def __getitem__(s, i):
        u = s.ids[i]
        files = sorted(f for f in os.listdir(f"{NPY}/{u}") if f.endswith(".npy") and (only is None or slot_of(f) in only))
        isfl = np.array([slot_of(f).endswith("_FL") for f in files])
        if s.train and a.series_drop and isfl.any():
            keep = isfl | (np.random.rand(len(files)) >= a.series_drop)
            files, isfl = [f for f, k_ in zip(files, keep) if k_], isfl[keep]
        S = len(files)
        if a.fl_share and isfl.any() and (~isfl).any():
            nfl = int(round(s.k * a.fl_share)); counts = np.zeros(S, int)
            counts[isfl] = nfl // isfl.sum(); counts[np.where(isfl)[0][: nfl - counts[isfl].sum()]] += 1
            counts[~isfl] = (s.k - nfl) // (~isfl).sum(); counts[np.where(~isfl)[0][: s.k - counts.sum()]] += 1
        else:
            counts = np.full(S, s.k // S); extra = np.arange(S)
            if s.train: np.random.shuffle(extra)
            counts[extra[: s.k - counts.sum()]] += 1
        xs, sl, ps = [], [], []
        for f, m in zip(files, counts):
            v = np.load(f"{NPY}/{u}/{f}", mmap_mode="r")
            if side.get(u) == "R" and not a.no_mirror:
                v = np.flip(v, axis=XAXIS[f.split("_")[0].split(".")[0]])
            n = len(v)
            if s.train:
                e = np.linspace(0, n, m + 1)
                c = np.minimum(np.floor(np.random.uniform(e[:-1], e[1:])).astype(int), n - 1)
            else:
                c = np.linspace(0, n - 1, m).round().astype(int)
            idx = np.clip(np.stack([c - 1, c, c + 1], 1), 0, n - 1)           # (m, 3)
            xs.append(np.ascontiguousarray(v[idx.ravel()]).reshape(m, 3, *v.shape[1:]))
            sl.append(np.full(m, SLOTS.index(slot_of(f)))); ps.append(c / max(n - 1, 1))
        return (torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(sl)),
                torch.from_numpy(np.concatenate(ps).astype(np.float32)), torch.from_numpy(s.Y[i]))

class Net(nn.Module):
    def __init__(s, d=512):
        super().__init__()
        vit = a.arch.startswith("vit")
        kw = dict(img_size=a.res) if vit else {}
        if a.drop_path: kw["drop_path_rate"] = a.drop_path
        if a.weights.endswith(".safetensors"):
            s.b = timm.create_model(a.arch, pretrained=True, pretrained_cfg_overlay=dict(file=f"{D}/{a.weights}" if not os.path.isabs(a.weights) else a.weights), num_classes=0, **kw)
        else:
            s.b = timm.create_model(a.arch, pretrained=False, num_classes=0, **kw)
            sd = torch.load(a.weights if os.path.isabs(a.weights) else f"{D}/{a.weights}", map_location="cpu")
            sd = {k: v for k, v in sd.items() if not k.startswith(("classifier.", "head.fc.", "fc.", "head.weight", "head.bias"))}   # keep head.norm (ConvNeXt's final LayerNorm)
            r = s.b.load_state_dict(sd, strict=False)
            print("weights:", a.weights, "| missing", len(r.missing_keys), "unexpected", len(r.unexpected_keys), flush=True)
        s.vit = vit
        fdim = s.b.num_features * (2 if vit else 1)
        cfg = s.b.pretrained_cfg
        s.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1))
        s.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1))
        s.proj = nn.Sequential(nn.Linear(fdim, d), nn.GELU())
        s.slot = nn.Embedding(len(SLOTS), d)
        s.pos = nn.Sequential(nn.Linear(1, 128), nn.GELU(), nn.Linear(128, d))
        s.sp = nn.Embedding(5, d) if a.spatial else None           # which part of the window a token covers
        s.ctx = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, d * 2, dropout=0.1, batch_first=True, norm_first=True), a.ctx, enable_nested_tensor=False) if a.ctx else None
        s.ctx_gate = nn.Parameter(torch.zeros(1)) if a.ctx else None   # starts at 0: the model begins as the plain pooling head
        s.R = a.branches
        s.attn = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Linear(256, 12 * s.R))   # attention maps: per finding, per branch
        if a.head == "ctran":
            s.lab = nn.Parameter(torch.randn(12, d) * 0.02)
            s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, d * 2, dropout=0.1, batch_first=True, norm_first=True), 2, enable_nested_tensor=False)
            s.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        s.cls = nn.Parameter(torch.randn(12, d) * 0.02); s.bias = nn.Parameter(torch.zeros(12))
        if a.aux_mention: s.cls_m = nn.Parameter(torch.randn(12, d) * 0.02); s.bias_m = nn.Parameter(torch.zeros(12))
        s.loc = nn.Embedding(2, d) if a.local_k else None          # global vs local window
        s.site_bias = nn.Embedding(N_SITES, 12) if a.site_offset else None
        if a.site_offset: nn.init.zeros_(s.site_bias.weight)
        if a.hier:                                                 # series-level attention, one score per finding
            s.attn2 = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Linear(256, 12)); s.register_buffer("sel", torch.eye(12))
        s.drop = nn.Dropout(0.2)
    def head_params(s):
        return [p for n, p in s.named_parameters() if not n.startswith(("b.", "ctx"))]
    def ctx_params(s):
        return [p for n, p in s.named_parameters() if n.startswith("ctx")]
    def view(s, x, train, grp=None, f0=None, ctr=None, shift=None):
        """uint8 (N,3,H,W) -> float (N,3,res,res): native center crop at eval, jittered crop + small rotation in training.
        f0: fraction of the image to read (default: native crop / --fov_frac); ctr: fixed (N,2) centres at eval; shift: max |offset| in training."""
        N, _, H, _ = x.shape
        x = x.float().div(255)
        if f0 is None and a.fov_frac and not a.full:              # global windows from a finer cache: anti-aliased downsize first (like INTER_AREA in the cache), then the usual native crop
            S_ = int(round(a.res / a.fov_frac))
            if H > S_:
                x = F.interpolate(x, size=(S_, S_), mode="bilinear", antialias=True, align_corners=False); H = S_
            f0 = min(1.0, a.res / H)
        if f0 is None:
            f0 = 1.0 if a.full else (min(1.0, a.res / H) if H > a.res else 1.0)
        if train:
            M = N if grp is None else int(grp.max()) + 1                     # one draw per window, or per series
            pick = (lambda t: t) if grp is None else (lambda t: t[grp])
            U_ = lambda lo, hi: torch.empty(M, device=x.device).uniform_(lo, hi)
            if grp is None:
                f = U_(f0 * 0.9, min(1.0, f0 * 1.1) if not a.full else 1.0); ang = U_(-7, 7) * math.pi / 180
                sh = (1 - f) if shift is None else torch.full_like(f, shift)
            else:
                f = (f0 * U_(0.88, 1.12)).clamp(max=1.0); ang = U_(-12, 12) * math.pi / 180
                sh = (1 - f).clamp(min=0.06)
            cx, cy = [pick((torch.rand(M, device=x.device) * 2 - 1) * sh) for _ in range(2)]
            f, ang = pick(f), pick(ang)
            th = torch.stack([torch.stack([f * ang.cos(), -f * ang.sin(), cx], 1), torch.stack([f * ang.sin(), f * ang.cos(), cy], 1)], 1)
        else:
            th = torch.tensor([[f0, 0, 0], [0, f0, 0]], device=x.device).expand(N, 2, 3)
            if ctr is not None:
                th = th.clone(); th[:, 0, 2] = ctr[:, 0]; th[:, 1, 2] = ctr[:, 1]
            elif a.eval_shift:
                _dx, _dy = (float(v) for v in a.eval_shift.split(","))
                th = th.clone(); th[:, 0, 2] = _dx; th[:, 1, 2] = _dy
        g = F.affine_grid(th, (N, 3, a.res, a.res), align_corners=False)
        x = F.grid_sample(x, g, mode="bilinear", padding_mode="zeros", align_corners=False)
        if train:
            c = pick(U_(0.85, 1.15)).view(N, 1, 1, 1); b = pick(U_(-0.05, 0.05)).view(N, 1, 1, 1)
            x = ((x - 0.5) * c + 0.5 + b).clamp(0, 1)
        return (x - s.mean) / s.std
    def forward(s, x, slot, pos):
        B, K = x.shape[:2]
        grp = (torch.arange(B, device=x.device).unsqueeze(1) * len(SLOTS) + slot).flatten() if (a.aug_series and s.training) else None
        if a.local_k:                                              # pick local windows spread over the bag (random in training)
            if s.training: li = torch.stack([torch.randperm(K, device=x.device)[: a.local_k] for _ in range(B)])
            else: li = torch.linspace(0, K - 1, a.local_k, device=x.device).round().long().unsqueeze(0).expand(B, -1)
            xl = torch.gather(x, 1, li.view(B, -1, 1, 1, 1).expand(-1, -1, *x.shape[2:]))
            sl_l, ps_l = torch.gather(slot, 1, li), torch.gather(pos, 1, li)
            T = torch.tensor([[-.25, -.25], [.25, -.25], [-.25, .25], [.25, .25]], device=x.device)   # 2x2 tiles over the central ~140 mm
            ctr = T[torch.arange(a.local_k, device=x.device) % 4].repeat(B, 1)
            xl = s.view(xl.flatten(0, 1), s.training, None, f0=a.local_frac, ctr=ctr, shift=0.3)
        x = s.view(x.flatten(0, 1), s.training, grp)
        if a.local_k:
            x = torch.cat([x.view(B, K, *x.shape[1:]), xl.view(B, a.local_k, *xl.shape[1:])], 1).flatten(0, 1)
            slot = torch.cat([slot, sl_l], 1); pos = torch.cat([pos, ps_l], 1); K = K + a.local_k
        if s.vit:
            t = s.b.forward_features(x); f = torch.cat([t[:, 0], t[:, s.b.num_prefix_tokens:].mean(1)], 1)
        elif a.spatial:
            fm = s.b.forward_features(x.contiguous(memory_format=torch.channels_last)).float()       # (B*K, F, h, w)
            tok = torch.cat([fm.mean((2, 3)).unsqueeze(1), F.adaptive_avg_pool2d(fm, 2).flatten(2).transpose(1, 2)], 1)   # (B*K, 5, F)
            h = s.proj(tok).view(B, K, 5, -1) + (s.slot(slot) + s.pos(pos.unsqueeze(-1))).unsqueeze(2) + s.sp.weight
            h = h.flatten(1, 2); K = K * 5; slot = slot.repeat_interleave(5, 1)
        else:
            f = s.b(x.contiguous(memory_format=torch.channels_last))
        if a.export_feats == "backbone" and not s.vit: s.f_out = f.float().view(B, K, -1).mean(1).detach()
        if not (a.spatial and not s.vit):
            h = s.proj(f.float().view(B, K, -1)) + s.slot(slot) + s.pos(pos.unsqueeze(-1))
            if a.local_k:
                h = h + s.loc((torch.arange(K, device=h.device) >= K - a.local_k).long()).unsqueeze(0)
        mix = None
        if s.training and a.bagmix and B > 1 and torch.rand(()) < a.bagmix:
            perm = torch.randperm(B, device=h.device)
            m = torch.rand(B, h.shape[1], device=h.device) < torch.rand(B, 1, device=h.device)   # windows a study keeps
            h = torch.where(m.unsqueeze(-1), h, h[perm])
            mix = (perm, m.float().mean(1))
        if s.ctx is not None:
            h = h + s.ctx_gate * (s.ctx(h) - h)
        if a.head == "ctran":
            t = s.tr(torch.cat([s.lab.unsqueeze(0).expand(B, -1, -1), h], 1))
            logit = s.out(t[:, :12]).squeeze(-1)
            return logit, {"bl": logit.unsqueeze(1), "w": None, "inst": None, "mix": mix}
        if a.hier:                                                 # windows -> one summary per series -> study, separately per finding
            Mk = F.one_hot(slot, len(SLOTS)).bool()                # (B, K, S)
            lg = s.attn(h).view(B, K, s.R, 12)[:, :, 0]            # (B, K, 12)
            lg = lg.unsqueeze(2).masked_fill(~Mk.unsqueeze(-1), float("-inf"))   # (B, K, S, 12)
            w1 = torch.nan_to_num(lg.float().softmax(1), nan=0.0)  # empty series -> zero weights
            zs = torch.einsum("bksj,bkd->bsjd", w1, s.drop(h).float())           # (B, S, 12, d)
            has = Mk.any(1)                                        # (B, S)
            g2 = (s.attn2(zs) * s.sel).sum(-1).masked_fill(~has.unsqueeze(-1), float("-inf"))   # (B, S, 12)
            w2 = g2.softmax(1)
            z = torch.einsum("bsj,bsjd->bjd", w2, zs)
            logit = (z * s.cls).sum(-1) + s.bias
            inst = torch.einsum("bkd,jd->bkj", h, s.cls)
            return logit, {"bl": logit.unsqueeze(1), "w": None, "inst": inst, "mix": mix}
        w = s.attn(h).view(B, K, s.R, 12).softmax(1)               # (B, K, R, 12)
        z = torch.einsum("bkrj,bkd->brjd", w, s.drop(h))           # (B, R, 12, d)
        bl = (z * s.cls).sum(-1) + s.bias                          # (B, R, 12) one logit per branch
        if a.export_feats: s.z_out = z.mean(1).detach()             # (B, 12, d) the vector each finding reads
        inst = torch.einsum("bkd,jd->bkj", h, s.cls)               # signed evidence of each window for each finding
        return bl.mean(1), {"bl": bl, "w": w, "inst": inst, "mix": mix, "z": z}

if a.domain_balance:
    _fm = np.array([u.startswith("fm_") for u in tr_ids]); _n_fm, _n_c = int(_fm.sum()), int((~_fm).sum())
    assert _n_fm > 0 and _n_c > 0, "domain_balance needs both competition and fm_ studies"
    _w = np.where(_fm, 0.5 / _n_fm, 0.5 / _n_c)
    from torch.utils.data import WeightedRandomSampler
    _smp = WeightedRandomSampler(torch.as_tensor(_w, dtype=torch.double), num_samples=len(tr_ids), replacement=True)
    print(f"domain_balance: {_n_c} competition + {_n_fm} fastMRI studies, each domain 50% of draws", flush=True)
    dl = DataLoader(Bags(tr_ids, Ytr, a.k, True), a.bs, sampler=_smp, num_workers=a.workers, drop_last=True, persistent_workers=True)
else:
    dl = DataLoader(Bags(tr_ids, Ytr, a.k, True), a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True)
dva = DataLoader(Bags(va_ids, Yva, a.k_eval, False), a.bs * 2, num_workers=a.workers)
dgo = DataLoader(Bags(go_ids, Ygo, a.k_eval, False), a.bs * 2, num_workers=a.workers)
net = Net().to(dev).to(memory_format=torch.channels_last)
if a.export_feats:                                                 # feature export only (no training)
    if a.export_feats != "backbone":
        sd = torch.load(a.export_feats if os.path.isabs(a.export_feats) else f"{D}/{a.export_feats}", map_location="cpu")
        print("export: load", net.load_state_dict(sd, strict=True), flush=True)
    ex_ids = [u for u in folds.index[folds.fold >= 0] if ok(u)]
    if a.limit: ex_ids = ex_ids[: a.limit]
    dex = DataLoader(Bags(ex_ids, np.zeros((len(ex_ids), 12), np.float32), a.k_eval, False), a.bs * 2, num_workers=a.workers)
    net.eval(); Z, LG = [], []
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16 if a.amp == "bf16" else torch.float16, enabled=a.amp != "fp32"):
        for x, sl, ps, _ in dex:
            lg, _aux = net(x.to(dev), sl.to(dev), ps.to(dev)); Z.append((net.f_out if a.export_feats == "backbone" else net.z_out).float().cpu().numpy().astype(np.float16)); LG.append(lg.float().cpu().numpy())
    out = f"{D}/runs/{a.tag}_feats.npz"
    np.savez(out, ids=np.array(ex_ids), fold=folds.fold.loc[ex_ids].values, Z=np.concatenate(Z), logit=np.concatenate(LG), source=a.export_feats, args=json.dumps(vars(a)))
    print(f"export: {len(ex_ids)} studies -> {out}", flush=True); raise SystemExit(0)
groups = [{"params": net.b.parameters(), "lr": a.lr}, {"params": net.head_params(), "lr": a.lr_head}]
if a.ctx: groups.append({"params": net.ctx_params(), "lr": a.lr_ctx})
opt = torch.optim.AdamW(groups, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=a.epochs * len(dl),
                                            pct_start=(a.warmup_ep / a.epochs) if a.warmup_ep else 0.1,
                                            final_div_factor=(1 / (25 * a.lr_floor)) if a.lr_floor else 1e4)
scaler = torch.amp.GradScaler(enabled=a.amp == "fp16")
lossf = nn.BCEWithLogitsLoss(reduction="none")
import collections
llbuf = [collections.deque(maxlen=3000) for _ in range(12)]     # recent losses of "absent" cells, per finding

EMA_D = (0.998, 0.9995, 0.9998)
emas = [{k: v.detach().clone() for k, v in net.state_dict().items()} for _ in EMA_D] if a.ema else []
def ema_update(t):
    with torch.no_grad():
        sd_ = net.state_dict()
        for d, e in zip(EMA_D, emas):
            dt = d if t > len(dl) else 0.0                         # the average just follows the weights during the first epoch
            for k, v in sd_.items():
                if v.is_floating_point(): e[k].lerp_(v, 1 - dt)
                else: e[k].copy_(v)

def total_loss(out, aux, y, sl, ep):
    if a.site_offset:                                            # training only; evaluation never adds it
        out = out + net.site_bias(y[:, -1].long()); y = y[:, :-1]
    if aux.get("mix") is not None:                               # spliced bags: mix the targets by window share
        perm, share = aux["mix"]; y = share.unsqueeze(1) * y + (1 - share.unsqueeze(1)) * y[perm]
    cw = None
    ym = None
    if a.aux_mention: y, ym = y[:, :12], y[:, 12:24]
    if y.shape[1] == 24: y, cw = y[:, :12], y[:, 12:]            # per-cell weights ride along with the targets
    L = lossf(out.float(), y)                                    # (B, 12)
    wgt = torch.ones_like(L) if cw is None else cw
    if a.llr:
        absent = y <= 0.03                                       # LLM grade "none": absent OR simply not mentioned
        r = a.llr * min(1.0, ep / max(1.0, 0.5 * a.epochs)) if ep >= 1 else 0.0
        Ld = L.detach()
        for j in range(12):
            v = Ld[:, j][absent[:, j]]
            if r > 0 and len(llbuf[j]) > 200 and len(v):
                thr = float(np.quantile(np.fromiter(llbuf[j], float), 1 - r))
                wgt[:, j] = torch.where(absent[:, j] & (Ld[:, j] > thr), torch.zeros_like(L[:, j]), wgt[:, j])
            llbuf[j].extend(v.tolist())
    loss = (L * wgt).sum() / wgt.sum().clamp(min=1) if cw is None else ((L * wgt).mean(0) / RATE.to(L.device)).mean()   # fixed per-finding normalisation
    if a.branches > 1 and aux["w"] is not None:
        bl = aux["bl"].float()
        loss = loss + (lossf(bl, y.unsqueeze(1).expand_as(bl)) * wgt.unsqueeze(1)).mean()
        if a.div:
            w0, w1 = aux["w"][:, :, 0].float(), aux["w"][:, :, 1].float()            # (B, K, 12)
            loss = loss + a.div * F.cosine_similarity(w0, w1, dim=1).mean()
    if a.smooth and aux["inst"] is not None:
        d_ = aux["inst"][:, 1:].float() - aux["inst"][:, :-1].float()               # neighbours in the bag
        same = (sl[:, 1:] == sl[:, :-1]).unsqueeze(-1).float()                      # only within one series
        loss = loss + a.smooth * (F.huber_loss(d_, torch.zeros_like(d_), reduction="none", delta=1.0) * same).sum() / same.sum().clamp(min=1) / 12
    if ym is not None:
        zm = aux["z"].float().mean(1)                           # (B, 12, d)
        al = (zm * net.cls_m).sum(-1) + net.bias_m
        loss = loss + a.aux_w * lossf(al, ym).mean()
    return loss

def predict(loader):
    net.eval(); P = []
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16 if a.amp == "bf16" else torch.float16, enabled=a.amp != "fp32"):
        for x, sl, ps, _ in loader:
            P.append(torch.sigmoid(net(x.to(dev), sl.to(dev), ps.to(dev))[0].float()).cpu().numpy())
    return np.concatenate(P)

def aucs(P, Y):
    r = [roc_auc_score(Y[:, j], P[:, j]) if 0 < Y[:, j].sum() < len(Y) else float("nan") for j in range(12)]
    return float(np.nanmean(r)), r

top = []                                                           # (silver auc, epoch, state_dict on cpu)
gstep = 0
for ep in range(a.epochs):
    net.train(); tot = 0; t0 = time.time()
    for x, sl, ps, y in dl:
        x, sl, ps, y = x.to(dev, non_blocking=True), sl.to(dev), ps.to(dev), y.to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16 if a.amp == "bf16" else torch.float16, enabled=a.amp != "fp32"):
            out, aux = net(x, sl, ps)
        loss = total_loss(out, aux, y, sl, ep)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
        gstep += 1
        if a.ema: ema_update(gstep)
    Pv, Pg = predict(dva), predict(dgo)
    sv, svl = aucs(Pv, Yva); gv, gvl = aucs(Pg, Ygo)
    print(f"EPOCH {ep} loss={tot / len(dl):.4f} silver_val_AUC={sv:.4f} rubric_silver={aucs(Pv, Yvr)[0]:.4f} gold_AUC={gv:.4f} ({time.time() - t0:.0f}s)", flush=True)
    top.append((sv, ep, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, gv, svl, gvl, Pv, Pg))
    top = sorted(top, key=lambda t: -t[0])[:3]

sv, ep, sd, gv, svl, gvl, Pv, Pg = top[0]
torch.save(sd, f"{OUT}/best.pt")
print(f"BEST {a.tag} fold={a.fold}: silver_val_AUC={sv:.4f} at ep{ep} (gold_AUC there={gv:.4f})")
print("  silver:", {k: round(v, 3) for k, v in zip(L, svl)})
print("  gold:  ", {k: round(v, 3) for k, v in zip(L, gvl)})
avg = {k: (sum(t[2][k].float() for t in top) / len(top)).to(top[0][2][k].dtype) if top[0][2][k].is_floating_point() else top[0][2][k] for k in sd}
net.load_state_dict(avg); torch.save(avg, f"{OUT}/swa.pt")
Pv, Pg = predict(dva), predict(dgo)
_px = {}
if a.pred_folds:
    px_ids = [u for u in folds.index[folds.fold.isin([int(x) for x in a.pred_folds.split(",")])] if ok(u)]
    _px = dict(px_ids=np.array(px_ids), Px=predict(DataLoader(Bags(px_ids, np.zeros((len(px_ids), 12), np.float32), a.k_eval, False), a.bs * 2, num_workers=a.workers)))
    print(f"pred_folds {a.pred_folds}: {len(px_ids)} studies predicted", flush=True)
sv2, svl2 = aucs(Pv, Yva); gv2, gvl2 = aucs(Pg, Ygo)
print(f"SWA  {a.tag} epochs {sorted(t[1] for t in top)}: silver_val_AUC={sv2:.4f} rubric_silver={aucs(Pv, Yvr)[0]:.4f} gold_AUC={gv2:.4f}")
print("  silver:", {k: round(v, 3) for k, v in zip(L, svl2)})
print("  gold:  ", {k: round(v, 3) for k, v in zip(L, gvl2)})
extra = dict(_px)
for d, e in zip(EMA_D, emas):                                      # EMAs: no epoch selection involved
    net.load_state_dict(e); Pe, Ge = predict(dva), predict(dgo)
    print(f"EMA {d}  {a.tag}: silver_val_AUC={aucs(Pe, Yva)[0]:.4f} rubric_silver={aucs(Pe, Yvr)[0]:.4f} gold_AUC={aucs(Ge, Ygo)[0]:.4f}", flush=True)
    extra[f"Pv_ema{d}"], extra[f"Pg_ema{d}"] = Pe, Ge
    torch.save({k: v.cpu() for k, v in e.items()}, f"{OUT}/ema{d}.pt")
np.savez(f"{OUT}/preds.npz", va_ids=np.array(va_ids), go_ids=np.array(go_ids), Pv=Pv, Pg=Pg, Yva=Yva, Yvr=Yvr, Ygo=Ygo, args=json.dumps(vars(a)), **extra)
