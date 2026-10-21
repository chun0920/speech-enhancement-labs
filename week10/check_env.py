"""Week 10 environment check.   Run (in the `se` conda environment):  python check_env.py

You want every line to start with [ OK ]. Run it again after each step in README.md.
"""
import importlib
import shutil
import sys
from pathlib import Path

problems = []


def line(ok, text):
    print(("[ OK ] " if ok else "[FAIL] ") + text)
    if not ok:
        problems.append(text)


for name in ("torch", "numpy", "soundfile", "einops", "pesq"):
    try:
        line(True, f"import {name:<12}{getattr(importlib.import_module(name), '__version__', '')}")
    except Exception as e:
        line(False, f"import {name:<12}-> {e}  (README, Step 0)")

git_path = shutil.which("git")
line(bool(git_path), f"git executable: {git_path or 'NOT FOUND'}")

for f in ("my_model.py", "infer_dir.py"):
    line(Path(f).exists(), f"{f} present in this folder (Step 0)")

print()
if problems:
    print(f"{len(problems)} problem(s) - fine if they belong to a step you have not done yet.")
    sys.exit(1)
print("Ready.")
