"""
se_eval - the evaluation script both groups share, so that every number can go into one comparison table.

It knows nothing about models: it reads WAV files. Give it a folder of enhanced files and, if the test set
has them, a folder of clean references with the SAME file names.

    uv run --project <this folder> <this folder>/se_eval.py --enhanced out/enhanced --clean <clean dir> \\
        --name "GTCRN official ckpt" --dataset vctk-demand-test

Metrics (one fixed protocol - do not change it inside a group's copy):
  with --clean    PESQ   ITU-T P.862.2 wide-band, `pesq` package, 16 kHz input
                  STOI   `pystoi`, 16 kHz input (pystoi resamples to 10 kHz inside), extended=False
                  ESTOI  `pystoi`, extended=True
                  SI-SDR in dB, both signals made zero-mean first (= the "SISNR" of GTCRN / UL-UNAS papers)
  with --dnsmos   DNSMOS OVRL, SIG, BAK (P.835) and P808 - Microsoft's dnsmos_local.py algorithm, the
                  official ONNX models in dnsmos/ (no clean reference needed: for blind test sets)

Rules the script enforces, because each one has moved published numbers by more than a paper's gain:
  - every file must be 16 kHz and mono: resampling is YOUR decision and must be written down
    (two 16 kHz versions of VCTK-DEMAND made with different resamplers differ by 0.03-0.05 PESQ)
  - enhanced and clean must have the same names; nothing is skipped silently
  - if lengths differ, both are cut to the shorter one, and the number of cut files is reported

Every run appends ONE summary line to <out>/eval_summary.csv (never overwritten) and writes the
per-file scores to <out>/<name>_per_file.csv. With --listen it also saves listening samples.
"""
import argparse
import csv
import datetime
import hashlib
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
HERE = Path(__file__).resolve().parent
VERSION = "1.0+" + hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:7]
INTRUSIVE = ("PESQ", "STOI", "ESTOI", "SI-SDR")
DNSMOS = ("DNSMOS_OVRL", "DNSMOS_SIG", "DNSMOS_BAK", "DNSMOS_P808")
SUMMARY_FIELDS = ["date", "name", "dataset", "n_files", *INTRUSIVE, *DNSMOS, "pesq_failed", "cut_files",
                  "max_len_diff", "enhanced_dir", "clean_dir", "se_eval_version", "note"]


# ------------------------------------------------------------------ intrusive metrics
def si_sdr(ref, est, eps=1e-8):
    """Scale-invariant SDR in dB; both signals are made zero-mean first."""
    ref = ref.astype(np.float64) - ref.mean()
    est = est.astype(np.float64) - est.mean()
    target = np.dot(est, ref) / (np.dot(ref, ref) + eps) * ref
    return float(10 * np.log10((np.sum(target ** 2) + eps) / (np.sum((est - target) ** 2) + eps)))


def intrusive(ref, est):
    from pesq import pesq
    from pystoi import stoi
    try:
        p = float(pesq(SR, ref, est, "wb"))
    except Exception as e:             # PESQ finds no speech at all (e.g. an all-silent output)
        if type(e).__name__ != "NoUtterancesError":
            raise
        p = float("nan")
    return {"PESQ": p, "STOI": float(stoi(ref, est, SR, extended=False)),
            "ESTOI": float(stoi(ref, est, SR, extended=True)), "SI-SDR": si_sdr(ref, est)}


# ------------------------------------------------------------------ DNSMOS (Microsoft DNS Challenge, dnsmos_local.py)
INPUT_LENGTH = 9.01
_dnsmos = {}


def _dnsmos_sessions():
    if not _dnsmos:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads, opts.inter_op_num_threads = 1, 1
        _dnsmos["main"] = ort.InferenceSession(str(HERE / "dnsmos" / "sig_bak_ovr.onnx"), opts, providers=["CPUExecutionProvider"])
        _dnsmos["p808"] = ort.InferenceSession(str(HERE / "dnsmos" / "model_v8.onnx"), opts, providers=["CPUExecutionProvider"])
    return _dnsmos["main"], _dnsmos["p808"]


def _melspec(audio):
    import librosa
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=321, hop_length=160, n_mels=120)
    return ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T


def dnsmos(audio):
    """P.835 OVRL / SIG / BAK (non-personalized polynomial fit) and P.808 MOS, averaged over 9.01 s windows, 1 s hop."""
    main, p808 = _dnsmos_sessions()
    p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
    p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    audio = audio.astype(np.float64)
    len_samples = int(INPUT_LENGTH * SR)
    while len(audio) < len_samples:                       # short clips are repeated until they are long enough
        audio = np.append(audio, audio)
    num_hops = int(np.floor(len(audio) / SR) - INPUT_LENGTH) + 1
    sig, bak, ovr, p8 = [], [], [], []
    for idx in range(num_hops):
        seg = audio[int(idx * SR): int((idx + INPUT_LENGTH) * SR)]
        if len(seg) < len_samples:
            continue
        s_raw, b_raw, o_raw = main.run(None, {"input_1": seg.astype("float32")[np.newaxis, :]})[0][0]
        p8.append(p808.run(None, {"input_1": _melspec(seg[:-160]).astype("float32")[np.newaxis, :, :]})[0][0][0])
        sig.append(p_sig(s_raw)), bak.append(p_bak(b_raw)), ovr.append(p_ovr(o_raw))
    return {"DNSMOS_OVRL": float(np.mean(ovr)), "DNSMOS_SIG": float(np.mean(sig)),
            "DNSMOS_BAK": float(np.mean(bak)), "DNSMOS_P808": float(np.mean(p8))}


# ------------------------------------------------------------------ one file
def read_mono16k(path):
    x, sr = sf.read(path, dtype="float32")
    if sr != SR or x.ndim != 1:
        raise ValueError(f"{path}: {sr} Hz, shape {x.shape} - se_eval only takes 16 kHz mono")
    return x


def score_one(job):
    enh_path, clean_path, want_dnsmos = job
    est = read_mono16k(enh_path)
    row = {"file": Path(enh_path).name}
    if clean_path:
        ref = read_mono16k(clean_path)
        row["len_diff"] = len(est) - len(ref)
        n = min(len(ref), len(est))
        row.update(intrusive(ref[:n], est[:n]))
    if want_dnsmos:
        row.update(dnsmos(est))
    return row


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Shared speech-enhancement evaluation (one protocol for both groups).")
    ap.add_argument("--enhanced", required=True, help="folder of enhanced 16 kHz mono WAV files")
    ap.add_argument("--clean", help="folder of clean references with the same file names (omit for blind test sets)")
    ap.add_argument("--name", required=True, help='what was evaluated, e.g. "GTCRN official ckpt" or "noisy"')
    ap.add_argument("--dataset", required=True, help='which test set, e.g. "vctk-demand-test" (write it the same way every time)')
    ap.add_argument("--dnsmos", action="store_true", help="also compute DNSMOS (needed when there is no clean reference)")
    ap.add_argument("--listen", help="a list of file ids (one per line) to save as listening samples")
    ap.add_argument("--noisy", help="folder of the noisy inputs, for the listening samples")
    ap.add_argument("--out", default="results", help="where eval_summary.csv and the per-file CSV go (default: results)")
    ap.add_argument("--note", default="", help="one line saved with the summary, e.g. which checkpoint")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="parallel processes")
    args = ap.parse_args()

    if not args.clean and not args.dnsmos:
        sys.exit("no clean references and no --dnsmos: there is nothing to compute")
    if args.listen and not Path(args.listen).is_file():      # check before scoring, which can take minutes
        sys.exit(f"--listen {args.listen}: no such file (the lists are in tools/se_eval/listen_lists/)")
    if args.noisy and not Path(args.noisy).is_dir():
        sys.exit(f"--noisy {args.noisy}: no such folder")
    enh_dir = Path(args.enhanced)
    files = sorted(enh_dir.glob("*.wav"))
    if not files:
        sys.exit(f"no .wav files in {enh_dir}")
    jobs = []
    if args.clean:
        clean_dir = Path(args.clean)
        missing = [f.name for f in files if not (clean_dir / f.name).exists()]
        if missing:
            sys.exit(f"{len(missing)} enhanced files have no clean file with the same name, e.g. {missing[:3]}")
        extra = len(list(clean_dir.glob("*.wav"))) - len(files)
        if extra > 0:
            print(f"[WARN] {extra} clean files have no enhanced file - only {len(files)} files are scored")
    to_check = files + ([Path(args.clean) / f.name for f in files] if args.clean else [])
    bad = [f.name for f in to_check if sf.info(f).samplerate != SR or sf.info(f).channels != 1]
    if bad:
        sys.exit(f"{len(bad)} files are not 16 kHz mono, e.g. {bad[:3]} - resample deliberately and write down how")
    jobs = [(str(f), str(Path(args.clean) / f.name) if args.clean else None, args.dnsmos) for f in files]

    print(f"se_eval {VERSION}: {len(files)} files, {args.jobs} processes, "
          f"{'intrusive' if args.clean else ''}{' + DNSMOS' if args.dnsmos else ''}")
    with mp.Pool(args.jobs) as pool:
        rows = []
        for k, row in enumerate(pool.imap(score_one, jobs, chunksize=4), 1):
            rows.append(row)
            if k % 100 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in args.name)
    per_file = out / f"{safe}_per_file.csv"
    keys = ["file"] + [k for k in ("len_diff", *INTRUSIVE, *DNSMOS) if k in rows[0]]
    with open(per_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    summary = {"date": datetime.date.today().isoformat(), "name": args.name, "dataset": args.dataset, "n_files": len(rows),
               "enhanced_dir": str(enh_dir.resolve()), "clean_dir": str(Path(args.clean).resolve()) if args.clean else "",
               "se_eval_version": VERSION, "note": args.note}
    for m in (*INTRUSIVE, *DNSMOS):
        vals = [r[m] for r in rows if m in r]
        summary[m] = f"{np.nanmean(vals):.4f}" if vals else ""
    if args.clean:
        diffs = [r["len_diff"] for r in rows]
        summary["pesq_failed"] = sum(np.isnan(r["PESQ"]) for r in rows)
        summary["cut_files"] = sum(d != 0 for d in diffs)
        summary["max_len_diff"] = max(diffs, key=abs)
    table = out / "eval_summary.csv"
    new = not table.exists() or table.stat().st_size == 0
    with open(table, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if new:
            w.writeheader()
        w.writerow(summary)

    print(f"\n  {args.name}  ({args.dataset}, {len(rows)} files)")
    for m in (*INTRUSIVE, *DNSMOS):
        if summary.get(m):
            print(f"    {m:<12} {float(summary[m]):8.3f}")
    if args.clean:
        print(f"    PESQ failed on {summary['pesq_failed']} files; {summary['cut_files']} files cut to the same length "
              f"(largest difference {summary['max_len_diff']} samples)")
        if summary["cut_files"]:
            print("    [WARN] lengths differ: check that your inference keeps the input length (padding, STFT frames)")
    print(f"  per file: {per_file}\n  summary line appended to: {table}")

    if args.listen:
        ids = [l.split()[0] for l in Path(args.listen).read_text().splitlines() if l.strip() and not l.startswith("#")]
        dest = out / "samples" / safe
        dest.mkdir(parents=True, exist_ok=True)
        by_name = {r["file"]: r for r in rows}
        print(f"\n  listening samples -> {dest}")
        for uid in ids:
            name = f"{uid}.wav"
            if name not in by_name:
                print(f"    [WARN] {name} is not among the enhanced files")
                continue
            sf.write(dest / f"{uid}_enhanced.wav", read_mono16k(enh_dir / name), SR, subtype="PCM_16")
            for kind, folder in (("noisy", args.noisy), ("clean", args.clean)):
                if folder:
                    sf.write(dest / f"{uid}_{kind}.wav", read_mono16k(Path(folder) / name), SR, subtype="PCM_16")
            r = by_name[name]
            print(f"    {uid}  " + "  ".join(f"{m} {r[m]:.2f}" for m in ("PESQ", "STOI", "SI-SDR", "DNSMOS_OVRL") if m in r))


if __name__ == "__main__":
    main()
