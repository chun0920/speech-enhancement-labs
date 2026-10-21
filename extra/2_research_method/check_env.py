"""Week 4 environment check.   Run:  uv run check_env.py

You want every line to start with [ OK ]. If one says [FAIL], see README.md > Troubleshooting,
or send a screenshot of this output to your mentor.
"""
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PACKAGES = ["numpy", "scipy", "matplotlib", "soundfile", "torch", "pesq", "pystoi"]
problems = []


def line(ok, text):
    print(("[ OK ] " if ok else "[FAIL] ") + text)
    if not ok:
        problems.append(text)


in_project = Path(sys.prefix).resolve() == (Path.cwd() / ".venv").resolve()
line(in_project, f"Python {sys.version.split()[0]} from this folder's .venv  ({sys.prefix})"
     + ("" if in_project else "  -> start it with  uv run check_env.py  (not python check_env.py)"))
line(sys.version_info[:2] == (3, 12), "Python 3.12, as pinned in .python-version")

for name in PACKAGES:
    try:
        version = getattr(importlib.import_module(name), "__version__", "")
        line(True, f"import {name:<11}{version}")
    except Exception as e:
        line(False, f"import {name:<11}-> {e}")

try:
    import torch

    batch = torch.zeros(2, 1, 16000)
    line(batch.shape == (2, 1, 16000), f"torch works on the CPU (CUDA build {torch.version.cuda})")
    if torch.cuda.is_available():
        on_gpu = (batch + 1).cuda()
        line(on_gpu.is_cuda and on_gpu.sum().item() == 32000, f"torch works on the GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("[INFO] no CUDA GPU visible - everything runs, but training takes 15-20 minutes per run on a CPU")
except Exception as e:
    first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    line(False, f"torch test -> {first}")

try:
    import numpy as np
    import soundfile as sf

    beep = 0.3 * np.sin(2 * np.pi * 1000 * np.arange(8000) / 16000)
    tmp = Path(tempfile.gettempdir()) / f"check_env_beep_{os.getpid()}.wav"
    sf.write(tmp, beep, 16000, subtype="PCM_16")
    back, sr = sf.read(tmp, dtype="float32")
    tmp.unlink()
    line(sr == 16000 and np.abs(back - beep).max() < 1e-3, "write + read a WAV file with soundfile")
except Exception as e:
    line(False, f"WAV write/read -> {e}")

try:
    data = Path("data")
    clips = sorted((data / "train").glob("*.flac"))
    line(len(clips) == 117, f"data/train: {len(clips)} training clips (117 expected)")
    tests = sorted((data / "test").glob("*.flac"))
    line(len(tests) == 20, f"data/test: {len(tests)} test sentences (20 expected)")
    babble = sf.info(data / "noise_babble.flac")
    line(babble.samplerate == 16000 and babble.duration >= 10, f"data/noise_babble.flac found ({babble.duration:.1f} s)")
    from pesq import pesq
    from pystoi import stoi

    speech, sr = sf.read(data / "speech_sample.wav", dtype="float32")
    p, s = pesq(16000, speech, speech, "wb"), stoi(speech, speech, 16000)
    line(p > 4.5 and s > 0.99, f"data/speech_sample.wav, PESQ and STOI run (clean vs itself: PESQ {p:.2f}, STOI {s:.2f})")
except Exception as e:
    line(False, f"data / metrics -> {e}  (copy the whole week4 folder, including data/)")

if shutil.which("git") is None:
    print("[WARN] git is not installed on this computer. The labs still run and stamp every number with a")
    print("       fingerprint of your code instead of a commit. Tell your mentor.")
else:
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    own = top.returncode == 0 and Path(top.stdout.strip()).resolve() == Path.cwd().resolve()
    line(own, "this folder is its own git repository" + ("" if own else "  -> README > Setup: git init"))
    if own:
        name = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True).stdout.strip()
        line(bool(name), f"git knows who you are: {name or '(not set)'}" + ("" if name else '  -> git config user.name "Your Name"'))
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        line(head.returncode == 0, f"first commit made: {head.stdout.strip() or '(none yet)'}"
             + ("" if head.returncode == 0 else "  -> git add -A && git commit -m 'Week 4 starting files'"))

print()
if problems:
    print(f"{len(problems)} problem(s). See README.md > Troubleshooting.")
    sys.exit(1)
print("Ready. Start with:  uv run lab1_reproduce.py --steps 0")
