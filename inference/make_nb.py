import json, sys
code1 = r'''# RSNA knee: inference only. Preprocessing and model code live in knee_infer.py (attached dataset).
import glob, json, os, subprocess, sys, time
t0 = time.time()
PKG = os.path.dirname((glob.glob("/kaggle/input/*/knee_infer.py") + glob.glob("/kaggle/input/*/*/knee_infer.py") + glob.glob("/kaggle/input/*/*/*/knee_infer.py"))[0])
COMP = os.path.dirname((glob.glob("/kaggle/input/*/test_series.csv") + glob.glob("/kaggle/input/*/*/test_series.csv"))[0])
print("package:", PKG, "| competition data:", COMP)
if glob.glob(os.path.join(PKG, "*.whl")):   # DICOM decoders for JPEG-Lossless / JPEG 2000, installed offline
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", PKG,
                        "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], capture_output=True, text=True)
    print("decoder wheels:", "ok" if r.returncode == 0 else r.stderr[-400:])
import pydicom, timm, torch
print("python", sys.version.split()[0], "| pydicom", pydicom.__version__, "| torch", torch.__version__, "| timm", timm.__version__,
      "| gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", "| cpus", os.cpu_count())
from pydicom.pixels import get_decoder
for name, uid in [("JPEG Baseline", "1.2.840.10008.1.2.4.50"), ("JPEG Lossless", "1.2.840.10008.1.2.4.57"), ("JPEG Lossless SV1", "1.2.840.10008.1.2.4.70"),
                  ("JPEG-LS Lossless", "1.2.840.10008.1.2.4.80"), ("JPEG 2000 Lossless", "1.2.840.10008.1.2.4.90"), ("JPEG 2000", "1.2.840.10008.1.2.4.91"), ("RLE", "1.2.840.10008.1.2.5")]:
    try:
        d = get_decoder(uid); print(f"decoder {name:20s} available={d.is_available} plugins={[k for k, v in d._available.items()] if hasattr(d, '_available') else '?'}")
    except Exception as e:
        print(f"decoder {name:20s} ERROR {e!r}"[:160])
'''
code2 = r'''sys.path.insert(0, PKG)
import knee_infer as ki
cfg = json.load(open(f"{PKG}/__CONFIG__"))
print(cfg["name"], "| prep", cfg["prep"], "| models", [m["file"] for m in cfg["models"]])
sub, stats = ki.run(f"{COMP}/test.csv", f"{COMP}/test_series.csv", f"{COMP}/test_series",
                    [(m["arch"], m["res"], f"{PKG}/{m['file']}", m.get("full", False), m.get("weight", 1.0), m.get("crop")) for m in cfg["models"]],
                    f"{PKG}/latclf.pt", "submission.csv", cfg["prior"], prep=cfg["prep"], k=cfg["k"],
                    only=cfg.get("only"), workers=4, budget_s=8 * 3600, stagger=cfg.get("stagger", False), syn_eff=cfg.get("syn_eff", 0.0), site_table=(json.load(open(f"{PKG}/{cfg['site_table']}")) if cfg.get("site_table") else None), families=[m.get("family") for m in cfg["models"]],
                    device="cuda" if torch.cuda.is_available() else "cpu", deadline=t0 + 8 * 3600, raw_out="submission_raw.npz")
print(sub.describe().T[["mean", "min", "max"]])
print("total seconds:", round(time.time() - t0))
'''
code2 = code2.replace("__CONFIG__", sys.argv[2] if len(sys.argv) > 2 else "config.json")
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c.splitlines(True)} for c in (code1, code2)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
