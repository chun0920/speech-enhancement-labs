"""Week 7 environment check.   Run (in the `se` conda environment):  python check_env.py

You want every line to start with [ OK ]. Run it again after each step in README.md.
"""
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
DATA = HOME / "data" / "VCTK-DEMAND"
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

for f in ("my_model.py", "infer_dir.py", "setrain_model.py", "paper_numbers.csv"):
    line(Path(f).exists(), f"{f} present in this folder (Step 0)")

for side in ("clean", "noisy"):
    have = len(list((DATA / "test" / side).glob("*.wav")))
    line(have == 824, f"{DATA / 'test' / side}: {have} files (expected 824)")

print()
if problems:
    print(f"{len(problems)} problem(s) - fine if they belong to a step you have not done yet.")
    sys.exit(1)
print("Ready.")
