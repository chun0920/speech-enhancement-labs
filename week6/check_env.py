"""Week 6 environment check.   Run (in the `se` conda environment):  python check_env.py

You want every line to start with [ OK ]. Run it again after each step in README.md.
"""
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
SETRAIN = HOME / "baselines" / "SEtrain"
DATA = HOME / "data" / "VCTK-DEMAND"
EXP = HOME / "exp"
problems = []


def line(ok, text):
    print(("[ OK ] " if ok else "[FAIL] ") + text)
    if not ok:
        problems.append(text)


for name in ("torch", "numpy", "soundfile", "einops", "omegaconf", "tensorboard", "joblib", "tqdm", "librosa", "pesq"):
    try:
        line(True, f"import {name:<12}{getattr(importlib.import_module(name), '__version__', '')}")
    except Exception as e:
        line(False, f"import {name:<12}-> {e}  (README, Step 0)")
try:
    import torch
    line(torch.cuda.is_available(), f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none visible'}")
except Exception as e:
    line(False, f"torch -> {e}")

for f in ("my_model.py", "infer_dir.py", "dataloader_vctk.py", "setrain_model.py"):
    line(Path(f).exists(), f"{f} present in this folder (Step 0)")

for kind, n in (("val", 1572), ("test", 824)):
    for side in ("clean", "noisy"):
        have = len(list((DATA / kind / side).glob("*.wav")))
        line(have == n, f"{DATA / kind / side}: {have} files (expected {n})" + ("" if have == n else "  (Step 0)"))

line((SETRAIN / "train.py").exists(), f"SEtrain: {SETRAIN}")
if (SETRAIN / ".git").exists() and shutil.which("git"):
    head = subprocess.run(["git", "-C", str(SETRAIN), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    line(head.startswith("3ec437d"), f"SEtrain version {head}")

runs = sorted(EXP.glob("gtcrn_vctk_full_*")) if EXP.exists() else []
if runs:
    line(True, f"Found training run: {runs[-1].name}")
else:
    line(True, "Training run in ~/exp/gtcrn_vctk_full_* (or running on lab112)")

print()
if problems:
    print(f"{len(problems)} problem(s) - fine if they belong to a step you have not done yet.")
    sys.exit(1)
print("Ready.")
