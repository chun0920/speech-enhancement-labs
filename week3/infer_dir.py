"""
Run your model (my_model.py) on every WAV file in a folder.                      (Week 3, given - no TODOs)

    python infer_dir.py --noisy <noisy test folder> --out out/enhanced            all files, on the GPU if there is one
    python infer_dir.py --noisy <noisy test folder> --out out/enhanced --limit 5  a quick test on 5 files
    python infer_dir.py ... --checkpoint <other .tar>                             another checkpoint than my_model.CHECKPOINT
    python infer_dir.py ... --adapter setrain_model                               another wrapper file than my_model.py (Week 4)

Every enhanced file is saved under the SAME name as its noisy file (se_eval matches files by name), as
32-bit float WAV so nothing is clipped or rounded. Each output is checked: same length as the input,
no NaN or Inf, not silent. At the end it prints the real-time factor (RTF = processing time / audio
duration) and writes out/enhanced/inference_log.json: which checkpoint (with its SHA256), which device
and library versions produced these files - so the numbers can be traced back later.
"""
import argparse
import hashlib
import importlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisy", required=True, help="folder with the noisy test WAV files (16 kHz mono)")
    ap.add_argument("--out", default="out/enhanced", help="where the enhanced files go")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="only the first N files (0 = all)")
    ap.add_argument("--checkpoint", help="use this checkpoint instead of my_model.CHECKPOINT")
    ap.add_argument("--adapter", default="my_model", help="the wrapper module (a .py file in this folder, without .py)")
    args = ap.parse_args()
    my_model = importlib.import_module(args.adapter)

    files = sorted(Path(args.noisy).glob("*.wav"))[: args.limit or None]
    if not files:
        raise SystemExit(f"no .wav files in {args.noisy}")
    if args.checkpoint:
        my_model.CHECKPOINT = Path(args.checkpoint)
    for name in ("REPO", "CHECKPOINT", "MODEL_NAME"):
        if getattr(my_model, name, None) is None:
            raise SystemExit(f"{args.adapter}.py: {name} is not set (Week 3 Exercise 1, or give --checkpoint)")
    if not Path(my_model.CHECKPOINT).exists():
        raise SystemExit(f"checkpoint not found: {my_model.CHECKPOINT}")

    model = my_model.build_model(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{my_model.MODEL_NAME}: {n_params:,} parameters | device {args.device} | {len(files)} files")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    problems, proc_time, audio_time = [], 0.0, 0.0
    with torch.inference_mode():
        my_model.enhance(model, np.zeros(my_model.SAMPLE_RATE, dtype=np.float32), args.device)   # warm-up, not timed
        for k, f in enumerate(files, 1):
            noisy, sr = sf.read(f, dtype="float32")
            if sr != my_model.SAMPLE_RATE or noisy.ndim != 1:
                raise SystemExit(f"{f.name}: {sr} Hz, shape {noisy.shape} - expected {my_model.SAMPLE_RATE} Hz mono")
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            enh = my_model.enhance(model, noisy, args.device)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            proc_time += time.perf_counter() - t0
            audio_time += len(noisy) / sr
            enh = np.asarray(enh, dtype=np.float32)
            if enh.shape != noisy.shape:
                problems.append(f"{f.name}: output shape {enh.shape}, input {noisy.shape}")
            elif not np.all(np.isfinite(enh)):
                problems.append(f"{f.name}: NaN or Inf in the output")
            elif np.abs(enh).max() < 1e-4:
                problems.append(f"{f.name}: the output is (almost) silent")
            sf.write(out / f.name, enh, sr, subtype="FLOAT")
            if k % 100 == 0 or k == len(files):
                print(f"  {k}/{len(files)}   RTF so far {proc_time / audio_time:.4f}", flush=True)

    rtf = proc_time / audio_time
    log = {"model": my_model.MODEL_NAME, "checkpoint": str(Path(my_model.CHECKPOINT).resolve()),
           "checkpoint_sha256": sha256(my_model.CHECKPOINT), "parameters": n_params, "noisy_dir": str(Path(args.noisy).resolve()),
           "n_files": len(files), "device": args.device,
           "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "",
           "cpu": platform.processor() or platform.machine(), "torch": torch.__version__,
           "audio_seconds": round(audio_time, 2), "processing_seconds": round(proc_time, 3), "rtf": round(rtf, 5),
           "date": time.strftime("%Y-%m-%d %H:%M"), "problems": problems}
    (out / "inference_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nsaved {len(files)} files in {out}/  |  RTF {rtf:.4f} on {args.device} "
          "(file by file, whole utterances - Week 8 measures RTF properly)")
    if problems:
        print(f"\n[PROBLEM] {len(problems)} files:")
        for p in problems[:10]:
            print("  " + p)
        raise SystemExit(1)
    print("all outputs: same length as the input, finite, not silent")


if __name__ == "__main__":
    main()
