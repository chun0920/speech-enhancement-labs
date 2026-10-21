"""Week 8 environment check.   Run (in the `se` conda environment):  python check_env.py

You want every line to start with [ OK ]. Run it again after each step in README.md.
"""
import importlib
import sys
from pathlib import Path

problems = []


def line(ok, text):
    print(("[ OK ] " if ok else "[FAIL] ") + text)
    if not ok:
        problems.append(text)


for name in ("torch", "numpy", "soundfile", "einops", "pesq", "thop"):
    try:
        line(True, f"import {name:<12}{getattr(importlib.import_module(name), '__version__', '')}")
    except Exception as e:
        hint = "pip install thop==0.1.1.post2209072238" if name == "thop" else "README, Step 0"
        line(False, f"import {name:<12}-> {e}  ({hint})")
try:
    import torch
    line(torch.cuda.is_available(), f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none visible'}")
except Exception as e:
    line(False, f"torch -> {e}")

for f in ("my_model.py", "infer_dir.py"):
    line(Path(f).exists(), f"{f} present in this folder (Step 0)")

print()
if problems:
    print(f"{len(problems)} problem(s) - fine if they belong to a step you have not done yet.")
    sys.exit(1)
print("Ready.")
