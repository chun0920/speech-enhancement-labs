"""Week 2 environment check.   Run:  python check_env.py

You want every line to start with [ OK ]. If one says [FAIL], see README.md > Troubleshooting,
or send a screenshot of this output to your mentor.
"""
import importlib
import sys
import tempfile
from pathlib import Path

PACKAGES = ["numpy", "scipy", "matplotlib", "soundfile", "torch", "pesq", "pystoi"]
problems = []


def line(ok, text):
    print(("[ OK ] " if ok else "[FAIL] ") + text)
    if not ok:
        problems.append(text)


line(sys.version_info >= (3, 9), f"Python {sys.version.split()[0]}  (3.9 or newer)")

for name in PACKAGES:
    try:
        version = getattr(importlib.import_module(name), "__version__", "")
        line(True, f"import {name:<11}{version}")
    except Exception as e:
        line(False, f"import {name:<11}-> {e}")

try:
    import torch

    batch = torch.zeros(2, 1, 16000)
    line(batch.shape == (2, 1, 16000), "torch works on the CPU")
    if torch.cuda.is_available():
        on_gpu = (batch + 1).cuda()
        line(on_gpu.is_cuda and on_gpu.sum().item() == 32000, f"torch works on the GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("[INFO] no CUDA GPU visible - fine for this week, everything runs on the CPU")
except Exception as e:
    first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    fix = "  -> PyTorch is too old for this GPU: see README > Troubleshooting (cu130)" if "no kernel image" in str(e) else ""
    line(False, f"torch test -> {first}{fix}")

try:
    import numpy as np
    import soundfile as sf

    beep = 0.3 * np.sin(2 * np.pi * 1000 * np.arange(8000) / 16000)
    tmp = Path(tempfile.gettempdir()) / "check_env_beep.wav"
    sf.write(tmp, beep, 16000, subtype="PCM_16")
    back, sr = sf.read(tmp, dtype="float32")
    tmp.unlink()
    line(sr == 16000 and np.abs(back - beep).max() < 1e-3, "write + read a WAV file with soundfile")
except Exception as e:
    line(False, f"WAV write/read -> {e}")

try:
    from pesq import pesq
    from pystoi import stoi

    speech, sr = sf.read(Path("data") / "speech_sample.wav", dtype="float32")
    line(sr == 16000 and len(speech) > 16000, f"data/speech_sample.wav found ({len(speech) / sr:.2f} s)")
    p, s = pesq(16000, speech, speech, "wb"), stoi(speech, speech, 16000)
    line(p > 4.5 and s > 0.99, f"PESQ and STOI run (clean vs itself: PESQ {p:.2f}, STOI {s:.2f})")
except Exception as e:
    line(False, f"PESQ / STOI test -> {e}")

print()
if problems:
    print(f"{len(problems)} problem(s). See README.md > Troubleshooting.")
    sys.exit(1)
print("Ready. Start with:  python lab1_frames_fft.py")
