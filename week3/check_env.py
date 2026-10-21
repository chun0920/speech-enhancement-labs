"""Week 3 environment check.   Run (in the `se` conda environment):  python check_env.py

You want every line to start with [ OK ]. Run it again after each setup step in README.md.
If one says [FAIL], see README.md > Troubleshooting, or send a screenshot to your mentor.
"""
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
GTCRN = HOME / "baselines" / "gtcrn"
SE = HOME / "speech-enhancement-labs" / "tools" / "se_eval"
DATA = {"VCTK-DEMAND test": (HOME / "data" / "VCTK-DEMAND" / "test", ["clean", "noisy"], 824),
        "DNS3 blind test": (HOME / "data" / "DNS3-blind", ["noisy"], 700)}
problems = []


def line(ok, text):
    print(("[ OK ] " if ok else "[FAIL] ") + text)
    if not ok:
        problems.append(text)


def info(text):
    print("[INFO] " + text)


line(sys.version_info >= (3, 9), f"Python {sys.version.split()[0]} ({sys.prefix})")
for name in ("numpy", "soundfile", "torch", "einops"):
    try:
        line(True, f"import {name:<10}{getattr(importlib.import_module(name), '__version__', '')}")
    except Exception as e:
        line(False, f"import {name:<10}-> {e}")

try:
    import torch
    if torch.cuda.is_available():
        x = torch.ones(1000, device="cuda")
        line(float((x * 2).sum()) == 2000, f"torch works on the GPU: {torch.cuda.get_device_name(0)} (CUDA build {torch.version.cuda})")
    else:
        line(False, "no CUDA GPU visible to torch (nvidia-smi works? the right PyTorch build? README Step 0)")
except Exception as e:
    msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    line(False, f"torch on the GPU -> {msg}" + ("  (PyTorch too old for this GPU: README Step 0)" if "kernel image" in str(e) else ""))

line(shutil.which("git") is not None, "git is installed" + ("" if shutil.which("git") else "  -> ask your mentor"))
uv = shutil.which("uv") or (str(HOME / ".local" / "bin" / "uv") if (HOME / ".local" / "bin" / "uv").exists() else None)
line(uv is not None, f"uv is installed ({uv})" if uv else "uv is installed  -> README Step 0, then open a new terminal")
line((SE / "se_eval.py").exists(), f"the shared evaluation tool: {SE}")
line((SE / ".venv").exists(), "its environment is built (cd $SE && uv sync)")

line((GTCRN / "gtcrn.py").exists(), f"GTCRN code: {GTCRN}  (README Step 2)")
for ck in ("model_trained_on_vctk.tar", "model_trained_on_dns3.tar"):
    line((GTCRN / "checkpoints" / ck).exists(), f"checkpoint {ck}")
if (GTCRN / ".git").exists() and shutil.which("git"):
    head = subprocess.run(["git", "-C", str(GTCRN), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    line(head.startswith("502ebfa"), f"GTCRN version {head} (the course was tested with 502ebfa)")

found = 0
for name, (root, kinds, n) in DATA.items():
    if root.exists():
        found += 1
        for k in kinds:
            have = len(list((root / k).glob("*.wav")))
            line(have == n, f"{name}: {root / k} has {have} files (expected {n})")
        src = root.parent / "SOURCE.txt" if name.startswith("VCTK") else root / "SOURCE.txt"
        if src.exists():
            info(f"fingerprints in {src}: compare them with README Step 3")
line(found > 0, "test data found  (README Step 3)" if found else "no test data yet  -> README Step 3")

try:
    import my_model
    unset = [n for n in ("REPO", "CHECKPOINT", "MODEL_NAME") if getattr(my_model, n, None) is None]
    if unset:
        info(f"my_model.py: {', '.join(unset)} not set yet (Exercise 1 - fine before Step 5)")
    else:
        line(Path(my_model.CHECKPOINT).exists(), f"my_model.py: CHECKPOINT {my_model.CHECKPOINT}")
except Exception as e:
    info(f"my_model.py does not import yet ({type(e).__name__}) - fine before Step 5")

print()
if problems:
    print(f"{len(problems)} problem(s). See README.md > Troubleshooting.")
    sys.exit(1)
print("Ready.")
