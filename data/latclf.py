"""Left/right knee classifier on npy2 coronal slices (NOT mirrored).

Trains on studies whose side is known (tag or report), reports held-out accuracy,
predicts the unknown ones, and fills laterality.csv where it is confident
(source="image"). The weights (latclf.pt) are reused at test time, where there is
no report and the tag is often blank.
"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, timm
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
NPY, CONF = "npy2", float(os.environ.get("CONF", 0.95))
lat = pd.read_csv("laterality.csv", keep_default_na=False)
has_cor = lambda u: os.path.exists(f"{NPY}/{u}/Coronal.npy")
lat = lat[lat.StudyInstanceUID.map(has_cor)]
known = lat[lat.side.isin(["L", "R"])].sample(frac=1, random_state=0)
unknown = lat[~lat.side.isin(["L", "R"])]
cut = int(len(known) * 0.8)
tr_ids, va_ids = known.StudyInstanceUID[:cut].tolist(), known.StudyInstanceUID[cut:].tolist()
y = dict(zip(known.StudyInstanceUID, (known.side == "R").astype(float)))
print(f"coronal available: known={len(known)} unknown={len(unknown)} train={len(tr_ids)} val={len(va_ids)}", flush=True)

class DS(Dataset):
    def __init__(s, ids, train): s.ids, s.train = ids, train
    def __len__(s): return len(s.ids)
    def __getitem__(s, i):
        u = s.ids[i]; v = np.load(f"{NPY}/{u}/Coronal.npy", mmap_mode="r"); n = len(v)
        idx = [int(n * q) for q in (0.5, 0.65, 0.8)]
        if s.train: idx = [min(n - 1, max(0, j + np.random.randint(-2, 3))) for j in idx]
        x = np.ascontiguousarray(v[idx]).astype(np.float32) / 255
        if s.train: x = np.clip(x * np.random.uniform(.85, 1.15), 0, 1)
        return torch.from_numpy((x - 0.45) / 0.225), torch.tensor(y.get(u, -1.0))

net = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
sd = {k: v for k, v in torch.load("effb0_pretrained.pth", map_location="cpu").items() if not k.startswith("classifier.")}
net.load_state_dict(sd, strict=False)
net = net.to(dev)
dl = DataLoader(DS(tr_ids, True), 32, shuffle=True, num_workers=6, drop_last=True)
EP = 4
opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=EP * len(dl))
lossf = nn.BCEWithLogitsLoss()

def predict(ids):
    net.eval(); P = []
    with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
        for x, _ in DataLoader(DS(ids, False), 64, num_workers=6):
            P.append(torch.sigmoid(net(x.to(dev)).float()).squeeze(1).cpu().numpy())
    return np.concatenate(P)

for ep in range(EP):
    net.train()
    for x, t in dl:
        with torch.autocast(dev, enabled=dev == "cuda"):
            loss = lossf(net(x.to(dev)).float().squeeze(1), t.to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    p = predict(va_ids); t = np.array([y[u] for u in va_ids])
    conf = (p > CONF) | (p < 1 - CONF)
    print(f"ep{ep} val acc={((p > .5) == t).mean():.4f} | confident {conf.mean():.3f} of val, "
          f"acc there {((p[conf] > .5) == t[conf]).mean():.4f}", flush=True)
torch.save(net.state_dict(), "latclf.pt")

pu = predict(unknown.StudyInstanceUID.tolist())
full = pd.read_csv("laterality.csv", keep_default_na=False)
pred = dict(zip(unknown.StudyInstanceUID, pu))
full["p_right"] = full.StudyInstanceUID.map(pred)
fill = full.side.isin(["", "both"]) & full.p_right.notna() & ((full.p_right > CONF) | (full.p_right < 1 - CONF))
full.loc[fill, "side"] = np.where(full.loc[fill, "p_right"] > .5, "R", "L")
full.loc[fill, "source"] = "image"
full.to_csv("laterality.csv", index=False)
print("filled from image:", int(fill.sum()), "| still unknown:", int((~full.side.isin(["L", "R"])).sum()))
print(full.source.value_counts().to_string())
