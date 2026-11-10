"""Wrap the S6-ours solo script as a 'block' notebook for make_nb_block3 (Pick A: our part + S6-ours, no public block).
The script runs as a subprocess; its hang guard is tightened because our stage runs first."""
import json, sys
import os
s = open(os.environ.get("S6_SOLO_SRC", "/workspace/rsna-knee/kaggle_pkg/nb_s6ours_solo/rsna-knee-s6ours-solo.py")).read()
for old, new in [("A_TIMEOUT_S = 9000", "A_TIMEOUT_S = 5400"), ("GUARD_MAX_S = 480 * 60", "GUARD_MAX_S = 390 * 60")]:
    assert s.count(old) == 1, old; s = s.replace(old, new)
cells = ["%%writefile /tmp/s6ours_solo.py\n" + s,
         "# S6-ours (our retrain of heliosli's S6 on our labels); its CoAtNet component is only a per-study fallback (W_A = 0)\n"
         "import subprocess, sys, os, time\nNB_T0 = time.time()\n"
         "_rc = subprocess.run([sys.executable, '-u', '/tmp/s6ours_solo.py']).returncode\n"
         "assert _rc == 0 and os.path.isfile('/kaggle/working/submission.csv'), f'S6-ours stage failed rc={_rc}'\nprint('S6OURS_STAGE_DONE', flush=True)\n"]
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c.splitlines(True)} for c in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w"), indent=1); print("written", sys.argv[1])
