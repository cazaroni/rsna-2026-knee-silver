"""Minimal launcher: Heliosli's compact_infer.main() unchanged, with nartaa's path setup (heliosli + cloud_code on sys.path,
download_data stub). Args after the script name go to compact_infer."""
import os, sys, types, runpy
C = "/workspace/s6data/code/heliosli"
for p in (C, C + "/cloud_code"):
    sys.path.insert(0, p)
m = types.ModuleType("training.download_data"); m.ChunkedArchive = None; sys.modules["training.download_data"] = m
import distutils.util  # setuptools shim (vendored MONAI needs it)
sys.argv = [C + "/compact_infer.py"] + sys.argv[1:]
runpy.run_path(C + "/compact_infer.py", run_name="__main__")
