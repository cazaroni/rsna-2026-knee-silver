"""LeJEPA v3 = v2 with --data/--npy, safetensors CNN weights, head.norm kept (ConvNeXt). v2 docstring follows.
LeJEPA v2: same objective as ssl_lejepa.py, but pretrained on what the trainer actually sees:
3-adjacent-slice windows (as the 3 channels) from ALL SIX series types of the npy3 cache.
(v1 used single slices from the fluid-sensitive series of npy2 only.)

Original notes: LeJEPA continued pretraining of a DINOv2 ViT on knee MRI slices (arXiv 2511.08544).

Loss = (1 - lamb) * invariance + lamb * SIGReg. No teacher, no EMA, no stop-gradient:
every view's projection is pulled to the mean of the global views, and SIGReg keeps the
projections an isotropic Gaussian so they cannot collapse.

Data: every npy2 slice (right knees mirrored to left) held in GPU memory as uint8; all
augmentation runs on the GPU. No left/right flips, they would undo the side normalization.
The held-out fold and the gold studies are excluded so the later evaluation stays honest.
"""
import argparse, json, math, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
ap.add_argument("--weights", default="/cluster/project/data/weights/dinov2_vits14_model.safetensors")
ap.add_argument("--out", default="/cluster/project/data/ssl/lejepa_v2")
ap.add_argument("--exclude_fold", type=int, default=0)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--n_local", type=int, default=4)
ap.add_argument("--lamb", type=float, default=0.02)
ap.add_argument("--k_local", type=int, default=8)
ap.add_argument("--proj_dim", type=int, default=256)
ap.add_argument("--lr", type=float, default=3e-5)          # backbone
ap.add_argument("--lr_proj", type=float, default=3e-4)
ap.add_argument("--warmup", type=int, default=0)             # 0 = one epoch
ap.add_argument("--g_scale", type=float, nargs=2, default=(0.8, 1.0))   # conservative: views must share the finding
ap.add_argument("--l_scale", type=float, nargs=2, default=(0.5, 0.8))
ap.add_argument("--l_size", type=int, default=112)
ap.add_argument("--rot", type=float, default=7.0)
ap.add_argument("--max_studies", type=int, default=0)      # smoke tests
ap.add_argument("--max_steps", type=int, default=0)
ap.add_argument("--resume", default="")
ap.add_argument("--data", default="/cluster/project/data")
ap.add_argument("--npy", default="npy3")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)
dev = torch.device("cuda")
D = a.data
XAXIS = {"Sagittal": 0, "Coronal": 2, "Axial": 2}
import cv2
SRC = 256                                                  # windows are stored at 256 px (downsized from the 384 px cache)

# ── data ────────────────────────────────────────────────────────────────────────
folds = pd.read_csv(f"{D}/folds.csv").set_index("StudyInstanceUID")
side = pd.read_csv(f"{D}/laterality.csv", keep_default_na=False).set_index("StudyInstanceUID").side.to_dict()
keep = set(folds.index[(folds.fold >= 0) & (folds.fold != a.exclude_fold)])
meta = [json.loads(l) for l in open(f"{D}/meta3.jsonl")]
vols = [(m["study"], m["slot"], m["shape"][0]) for m in meta if "shape" in m and m["study"] in keep]
if a.max_studies:
    ok = set(sorted({v[0] for v in vols})[: a.max_studies]); vols = [v for v in vols if v[0] in ok]
offs = np.concatenate([[0], np.cumsum([v[2] for v in vols])])
lens_t = torch.tensor([v[2] for v in vols])
N = int(offs[-1])
t0 = time.time()
data = torch.empty((N, 256, 256), dtype=torch.uint8)

def load(i):
    u, p, n = vols[i]
    v = np.load(f"{D}/{a.npy}/{u}/{p}.npy")
    if side.get(u) == "R":
        v = np.flip(v, axis=XAXIS[p.split("_")[0]])
    v = np.stack([cv2.resize(s_, (SRC, SRC), interpolation=cv2.INTER_AREA) for s_ in np.ascontiguousarray(v)])
    data[offs[i]:offs[i + 1]] = torch.from_numpy(v)

with ThreadPoolExecutor(8) as ex:
    list(ex.map(load, range(len(vols))))
# v2 has 510k slices (31 GiB): keep them in host memory and move one batch at a time (v1's 22 GiB fitted on the GPU)
print(f"{len(vols)} volumes, {N} slices in RAM ({data.numel() / 2**30:.1f} GiB) in {time.time() - t0:.0f}s", flush=True)

MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
U = lambda b, lo, hi: torch.empty(b, device=dev).uniform_(lo, hi)

def view(x, size, scale):
    """x: (B,3,256,256) float in [0,1] (3 adjacent slices) -> one augmented view (B,3,size,size), ImageNet-normalized."""
    B = x.shape[0]
    s, r = U(B, *scale), torch.exp(U(B, math.log(3 / 4), math.log(4 / 3)))
    w, h = (s * r).sqrt().clamp(max=1), (s / r).sqrt().clamp(max=1)
    cx, cy = U(B, -1, 1) * (1 - w), U(B, -1, 1) * (1 - h)
    ang = U(B, -a.rot, a.rot) * math.pi / 180
    cos, sin = ang.cos(), ang.sin()
    theta = torch.stack([torch.stack([w * cos, -h * sin, cx], 1), torch.stack([w * sin, h * cos, cy], 1)], 1)
    grid = F.affine_grid(theta, (B, 3, size, size), align_corners=False)
    o = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False).clamp(0, 1)
    o = o.pow(torch.exp(U(B, -0.25, 0.25)).view(B, 1, 1, 1))                                    # gamma
    o = (o - 0.5) * U(B, 0.85, 1.15).view(B, 1, 1, 1) + 0.5 + U(B, -0.07, 0.07).view(B, 1, 1, 1)    # contrast, brightness
    blur = (torch.rand(B, device=dev) < 0.15).view(B, 1, 1, 1)
    o = torch.where(blur, F.avg_pool2d(o, 3, 1, 1), o)
    o = (o + torch.randn_like(o) * U(B, 0, 0.03).view(B, 1, 1, 1)).clamp(0, 1)
    return (o - MEAN) / STD

# ── model ───────────────────────────────────────────────────────────────────────
VIT = a.model.startswith("vit")
if VIT:
    net = timm.create_model(a.model, pretrained=True, pretrained_cfg_overlay=dict(file=a.weights),
                            img_size=224, num_classes=0, dynamic_img_size=True).to(dev)
    edim = net.num_features * 2
else:                                                      # CNN backbone (e.g. efficientnet_b0): pooled features
    net = timm.create_model(a.model, pretrained=False, num_classes=0)
    if a.weights.endswith(".safetensors"):
        from safetensors.torch import load_file; sd0 = load_file(a.weights)
    else:
        sd0 = torch.load(a.weights, map_location="cpu")
    sd = {k: v for k, v in sd0.items() if not k.startswith(("classifier.", "head.fc.", "fc.", "head.weight", "head.bias"))}   # keep ConvNeXt head.norm
    r = net.load_state_dict(sd, strict=False); print("weights:", a.weights, "| missing", len(r.missing_keys), "unexpected", len(r.unexpected_keys), flush=True)
    net = net.to(dev).to(memory_format=torch.channels_last)
    edim = net.num_features
proj = nn.Sequential(nn.Linear(edim, 2048), nn.BatchNorm1d(2048), nn.GELU(),
                     nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.GELU(),
                     nn.Linear(2048, a.proj_dim)).to(dev)

def embed(x):
    if not VIT:
        return net(x.contiguous(memory_format=torch.channels_last))
    t = net.forward_features(x)
    return torch.cat([t[:, 0], t[:, net.num_prefix_tokens:].mean(1)], 1)

class LocaLeReg(nn.Module):
    """Local Epps-Pulley test of a max-entropy (uniform-ball, identity-covariance) target per neighbourhood."""
    def __init__(s, dim, k=8, knots=17, slices=256, iters=10):
        super().__init__()
        t = torch.linspace(0, 3, knots); dt = 3 / (knots - 1)
        wts = torch.full((knots,), 2 * dt); wts[[0, -1]] = dt
        g = torch.Generator().manual_seed(0); n = 200000
        u = torch.randn(n, dim, generator=g); u = u / u.norm(dim=1, keepdim=True)
        r = (dim + 2) ** 0.5 * torch.rand(n, 1, generator=g).pow(1.0 / dim)
        x1 = (u * r)[:, 0]                                    # 1-D marginal of the uniform ball (isotropic: any direction)
        phi = torch.cos(x1.unsqueeze(1) * t).mean(0)          # target CF (real; the sine part is 0 by symmetry)
        s.k, s.iters, s.slices = k, iters, slices
        s.register_buffer("t", t); s.register_buffer("phi", phi); s.register_buffer("w", wts * torch.exp(-t.square() / 2))
    @torch.no_grad()
    def groups(s, m):                                         # k-means on the batch (detached)
        m = F.normalize(m, dim=1)                             # as the authors: k-means on normalised points
        c = m[torch.randperm(m.size(0), device=m.device)[: s.k]].clone()
        for _ in range(s.iters):
            a = torch.cdist(m, c).argmin(1)
            for j in range(s.k):
                if (a == j).any(): c[j] = m[a == j].mean(0)
        return torch.cdist(m, c).argmin(1)
    def forward(s, p):                                        # p: (V, B, D)
        a = s.groups(p.detach().mean(0))
        A = torch.randn(p.size(-1), s.slices, device=p.device); A = A / A.norm(dim=0)
        tot, n = 0.0, 0
        for j in range(s.k):
            idx = (a == j).nonzero().squeeze(1)
            if idx.numel() < 4: continue
            q = p[:, idx]; q = q - q.mean(dim=(0, 1), keepdim=True)     # centre the neighbourhood on its own mean
            xt = (q @ A).unsqueeze(-1) * s.t                            # (V, b, slices, knots)
            err = (xt.cos().mean(-3) - s.phi).square() + xt.sin().mean(-3).square()
            tot = tot + (err @ s.w).mean() * idx.numel() * idx.numel(); n += idx.numel()
        return tot / max(n, 1)

sigreg = LocaLeReg(a.proj_dim, k=a.k_local).to(dev)
steps_per_epoch = N // a.bs
a.warmup = a.warmup or steps_per_epoch
offs_t, lens_t = torch.from_numpy(offs[:-1]).to(dev), lens_t.to(dev)
total = a.max_steps or a.epochs * steps_per_epoch
opt = torch.optim.AdamW([{"params": net.parameters(), "lr": a.lr}, {"params": proj.parameters(), "lr": a.lr_proj}],
                        weight_decay=0.05)
lr_at = lambda k: min(1, (k + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, k / total))))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
scaler = torch.amp.GradScaler()
step = 0
if a.resume and os.path.exists(a.resume):
    ck = torch.load(a.resume, map_location="cpu")
    net.load_state_dict(ck["net"]); proj.load_state_dict(ck["proj"]); opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"]); step = ck["step"]
    print("resumed at step", step, flush=True)
print(f"steps/epoch={steps_per_epoch} total={total} bs={a.bs} views=2x224{tuple(a.g_scale)}+{a.n_local}x{a.l_size}{tuple(a.l_scale)} "
      f"lamb={a.lamb} lr={a.lr}/{a.lr_proj} proj={a.proj_dim} warmup={a.warmup}", flush=True)

# ── train ───────────────────────────────────────────────────────────────────────
V = 2 + a.n_local
net.train(); proj.train()
t0, seen, log = time.time(), 0, {"inv": 0.0, "sig": 0.0, "n": 0}
while step < total:
    si = torch.randint(0, len(vols), (a.bs,), device=dev)       # series-balanced: long stacks do not dominate
    c = (torch.rand(a.bs, device=dev) * lens_t[si]).long()                       # window centre within the series
    nb = torch.stack([c - 1, c, c + 1], 1).clamp(min=0).minimum((lens_t[si] - 1).unsqueeze(1))
    x = data[(offs_t[si].unsqueeze(1) + nb).reshape(-1).cpu()].to(dev, non_blocking=True).view(a.bs, 3, SRC, SRC).float().div(255)
    with torch.no_grad():
        g = torch.cat([view(x, 224, a.g_scale) for _ in range(2)])
        l = torch.cat([view(x, a.l_size, a.l_scale) for _ in range(a.n_local)])
    with torch.autocast("cuda"):
        e = torch.cat([embed(g), embed(l)])
    p = proj(e.float()).view(V, a.bs, -1)
    inv = (p[:2].mean(0) - p).square().mean()
    sig = sigreg(p)
    loss = (1 - a.lamb) * inv + a.lamb * sig
    if not torch.isfinite(loss):
        raise SystemExit(f"non-finite loss at step {step}: inv={inv.item()} sig={sig.item()}")
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    gn = nn.utils.clip_grad_norm_(list(net.parameters()) + list(proj.parameters()), 3.0)
    scaler.step(opt); scaler.update(); sched.step()
    step += 1; seen += a.bs
    log["inv"] += inv.item(); log["sig"] += sig.item(); log["n"] += 1
    if step % 50 == 0:
        with torch.no_grad():
            es = e.float()[: a.bs]
            estd = F.normalize(es, dim=1).std(0).mean().item()      # -> 0 means the embedding collapsed
        print(f"step {step}/{total} ep {step / steps_per_epoch:.2f} inv={log['inv'] / log['n']:.4f} "
              f"sig={log['sig'] / log['n']:.3f} gnorm={gn.item():.2f} emb_std={estd:.4f} "
              f"lr={sched.get_last_lr()[0]:.2e} {seen / (time.time() - t0):.0f} slices/s", flush=True)
        log = {"inv": 0.0, "sig": 0.0, "n": 0}
    if step % steps_per_epoch == 0 or step == total:
        ep = step // steps_per_epoch if step % steps_per_epoch == 0 else "final"
        torch.save(net.state_dict(), f"{a.out}/backbone_ep{ep}.pt")
        torch.save({"net": net.state_dict(), "proj": proj.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": step}, f"{a.out}/last.pt")
        print(f"saved backbone_ep{ep}.pt at step {step} ({(time.time() - t0) / 60:.1f} min)", flush=True)
print("done", flush=True)
