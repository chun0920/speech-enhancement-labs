"""Week 9 environment check.   Run (in the `se` conda environment):  python check_env.py

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


for name in ("torch", "numpy", "soundfile", "einops", "pesq", "scipy"):
    try:
        line(True, f"import {name:<12}{getattr(importlib.import_module(name), '__version__', '')}")
    except Exception as e:
        line(False, f"import {name:<12}-> {e}  (README, Step 0)")
for f in ("my_model.py", "setrain_model.py"):
    line(Path(f).exists(), f"{f} present in this folder (Step 0)")

data = Path.home() / "data" / "VCTK-DEMAND"
line(len(list((data / "test" / "clean").glob("*.wav"))) == 824, f"{data / 'test'}: the test set (Week 3)")
line((data / "logfiles" / "log_testset.txt").exists(), "noise type and SNR of every test file (logfiles/log_testset.txt)")
listen = Path.home() / "speech-enhancement-labs" / "tools" / "se_eval" / "listen_lists" / "vctk_demand_test.txt"
line(listen.exists(), "the listening list (tools/se_eval/listen_lists/vctk_demand_test.txt)")

print()
if problems:
    print(f"{len(problems)} problem(s) - fine if they belong to a step you have not done yet.")
    sys.exit(1)
print("Ready.")
