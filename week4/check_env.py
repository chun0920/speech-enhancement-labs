"""Week 4 environment check.   Run (in the `se` conda environment):  python check_env.py

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

for f in ("my_model.py", "infer_dir.py"):
    line(Path(f).exists(), f"{f} copied from your Week 3 folder (Step 0)")

for kind, n in (("train", 10000), ("val", 1572), ("test", 824)):
    for side in ("clean", "noisy"):
        have = len(list((DATA / kind / side).glob("*.wav")))
        line(have == n, f"{DATA / kind / side}: {have} files (expected {n})" + ("" if have == n else "  (Step 1)"))

line((SETRAIN / "train.py").exists(), f"SEtrain: {SETRAIN}  (Step 2)")
if (SETRAIN / ".git").exists() and shutil.which("git"):
    head = subprocess.run(["git", "-C", str(SETRAIN), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    line(head.startswith("3ec437d"), f"SEtrain version {head} (the course uses 3ec437d)")
if (SETRAIN / "train.py").exists():
    text = (SETRAIN / "train.py").read_text()
    line("from dataloader_vctk import VCTKDataset as Dataset" in text, "train.py imports VCTKDataset (Step 4)")
    line((SETRAIN / "dataloader_vctk.py").exists(), "dataloader_vctk.py is in SEtrain (Step 4)")
    line((SETRAIN / "configs" / "cfg_train_vctk.yaml").exists(), "cfg_train_vctk.yaml is in SEtrain/configs (Step 4)")

print()
if problems:
    print(f"{len(problems)} problem(s) - fine if they belong to a step you have not done yet.")
    sys.exit(1)
print("Ready.")
