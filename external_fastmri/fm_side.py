"""Knee side for fastMRI studies with our image side classifier (latclf.pt, EfficientNet-B0 trained on competition npy2 coronal
fluid slices at 256 px). fm3 Coronal_FL volumes (same canonical orientation, 384 px) are resized to 256 and read at the same
relative slice positions (0.5, 0.65, 0.8). Accuracy is measured on the 5,046 studies whose side is known from tags/series names
before anything is applied to the unknown ones. Writes fastmri/fm_side_pred.csv (study, p_right, known_side)."""
import os, glob, numpy as np, pandas as pd, torch, timm, cv2
P = "/cluster/project"; FM = f"{P}/fastmri"
known = pd.read_csv(f"{FM}/fm_known_side.csv").set_index("study").side.to_dict()
studies = sorted(os.listdir(f"{FM}/fm3"))
net = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
net.load_state_dict(torch.load(f"{P}/data/latclf.pt", map_location="cpu")); net = net.cuda().eval()
def x_of(u):
    f = f"{FM}/fm3/{u}/Coronal_FL.npy"
    if not os.path.exists(f):
        f = f"{FM}/fm3/{u}/Coronal_NF.npy"
        if not os.path.exists(f): return None
    v = np.load(f, mmap_mode="r"); n = len(v)
    sl = [cv2.resize(np.asarray(v[int(n * q)], np.float32), (256, 256), interpolation=cv2.INTER_AREA) for q in (0.5, 0.65, 0.8)]
    x = np.stack(sl) / 255.0
    return torch.from_numpy(((x - 0.45) / 0.225).astype(np.float32))
rows, batch, ids = [], [], []
def flush():
    if not batch: return
    with torch.no_grad(), torch.autocast("cuda"):
        p = torch.sigmoid(net(torch.stack(batch).cuda()).float()).squeeze(1).cpu().numpy()
    rows.extend(zip(ids, p)); batch.clear(); ids.clear()
for u in studies:
    x = x_of(u)
    if x is None: continue
    batch.append(x); ids.append(u)
    if len(batch) == 64: flush()
flush()
d = pd.DataFrame(rows, columns=["study", "p_right"]); d["known_side"] = d.study.map(known).fillna("")
d.to_csv(f"{FM}/fm_side_pred.csv", index=False)
k = d[d.known_side.isin(["L", "R"])]
acc = ((k.p_right > 0.5) == (k.known_side == "R")).mean()
conf = (k.p_right > 0.95) | (k.p_right < 0.05)
print(f"scored {len(d)} | known {len(k)}: accuracy {acc:.4f}; confident (p<0.05 or >0.95) {conf.mean():.3f} with accuracy {((k.p_right[conf] > 0.5) == (k.known_side[conf] == 'R')).mean():.4f}")
u = d[~d.known_side.isin(['L', 'R'])]; uc = (u.p_right > 0.95) | (u.p_right < 0.05)
print(f"unknown {len(u)}: confident {uc.sum()} ({uc.mean():.3f})"); print("FM-SIDE-DONE")
