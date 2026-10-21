"""
Download VCTK-DEMAND and make the course's 16 kHz version of it - the same files on every computer.

    uv run --project <this folder> <this folder>/prepare_vctk_demand.py --root ~/data/VCTK-DEMAND           test set only (~310 MB)
    uv run --project <this folder> <this folder>/prepare_vctk_demand.py --root ~/data/VCTK-DEMAND --train   + training set (~5 GB)

Source: Valentini-Botinhao (2017), "Noisy speech database for training speech enhancement algorithms and TTS
models", University of Edinburgh DataShare, https://doi.org/10.7488/ds/2117 (48 kHz WAV). Read its licence on
that page before you use the data for anything other than this course.

What it does, in this fixed order:
  1. downloads the zip files into <root>/zips (skips a file that is already there and complete)
     and checks the SHA256 of every zip whose checksum is known
  2. converts every 48 kHz file to 16 kHz with librosa.load(sr=16000, res_type="soxr_hq") and saves it
     as 16-bit PCM. The resampler is fixed on purpose: two 16 kHz versions of this dataset made with
     different resamplers differ by 0.03-0.05 PESQ. With this one, GTCRN's official checkpoint scores
     PESQ 2.867 / STOI 0.940 / SI-SDR 18.80 (paper: 2.87 / 0.940 / 18.83).
  3. with --train: moves the 1,572 utterances listed in splits/vctk_demand_val.txt into val/
  4. checks the number of files and writes <root>/SOURCE.txt with everything above plus a fingerprint
     per folder (a SHA256 over all file names and contents) - two computers with the same fingerprint
     have exactly the same data

Result:
  <root>/test/clean, test/noisy     824 + 824 files   (speakers p232, p257 - never train or pick checkpoints on these)
  <root>/train/clean, train/noisy   10,000 + 10,000   (--train)
  <root>/val/clean, val/noisy       1,572 + 1,572     (--train)
  <root>/logfiles/                  noise type and SNR of every utterance (log_testset.txt, ...)
"""
import argparse
import hashlib
import multiprocessing as mp
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = "https://datashare.ed.ac.uk/bitstreams/{}/download"
ZIPS = {   # name: (bitstream id, SHA256 or None if not recorded yet)
    "clean_testset_wav.zip": ("dec213d3-bf57-4777-9663-c24bdce92d5e", "b60bfc2b9df38466d8eb45057c087c6e5dee196c67a0ed7bca628ae2a8611c4e"),
    "noisy_testset_wav.zip": ("13c1bfbf-14a6-41db-9b41-8f7310f01ad5", "ba4a893f45b627f15209ba30ab584dd495b3680cd6f216dd08d143a257611993"),
    "logfiles.zip": ("11185dc8-9cf1-405b-b858-35bd6a04aedd", "ab37c41f31bc2b6c204cc10c8e5bb8bde800307fdb49f277e366d986fefc9e39"),
    "clean_trainset_28spk_wav.zip": ("245452b6-6235-44b6-a6f9-e7eb19797769", None),
    "noisy_trainset_28spk_wav.zip": ("ecb5a102-bb00-46d3-8af5-40c79823b837", None),
}
SR = 16000


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(name, dest):
    bitstream, expected = ZIPS[name]
    path = dest / name
    if path.exists() and (expected is None or sha256(path) == expected):
        print(f"  {name}: already here")
        return path
    tmp = path.with_suffix(".part")
    t0 = time.time()

    def progress(blocks, block_size, total):
        done = blocks * block_size
        if total > 0 and blocks % 200 == 0:
            print(f"\r  {name}: {100 * done / total:5.1f} % of {total / 1e6:.0f} MB  {done / 1e6 / max(time.time() - t0, 1e-3):5.1f} MB/s",
                  end="", flush=True)
    for attempt in range(4):                 # on a network error wait and try again (2, 4, 8 s)
        try:
            urllib.request.urlretrieve(BASE.format(bitstream), tmp, progress)
            break
        except OSError as e:
            if attempt == 3:
                raise
            print(f"\n  network error ({e}); trying again in {2 ** (attempt + 1)} s", flush=True)
            time.sleep(2 ** (attempt + 1))
    print()
    got = sha256(tmp)
    if expected and got != expected:
        tmp.unlink()
        sys.exit(f"{name}: SHA256 {got} does not match {expected} - the download is broken, run again")
    tmp.rename(path)
    return path


def resample_one(job):
    src, dst = job
    import librosa
    import soundfile as sf
    y, _ = librosa.load(src, sr=SR, res_type="soxr_hq")
    sf.write(dst, y, SR, subtype="PCM_16")


def unzip_and_resample(zip_path, out_dir, tmp_dir, jobs):
    """Extract the 48 kHz WAVs of one zip into tmp_dir, resample them into out_dir, delete the 48 kHz copies."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.endswith(".wav") and "__MACOSX" not in m]
        todo = []
        for m in members:
            name = Path(m).name
            if not (out_dir / name).exists():
                z.extract(m, tmp_dir)
                todo.append((str(tmp_dir / m), str(out_dir / name)))
    if todo:
        with mp.Pool(jobs) as pool:
            for k, _ in enumerate(pool.imap_unordered(resample_one, todo, chunksize=8), 1):
                if k % 1000 == 0 or k == len(todo):
                    print(f"    {out_dir.parent.name}/{out_dir.name}: {k}/{len(todo)}", flush=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return len(members)


def fingerprint(folder):
    """One SHA256 over all (file name, file content) pairs of a folder, in sorted order."""
    h = hashlib.sha256()
    for f in sorted(folder.glob("*.wav")):
        h.update(f.name.encode() + b"\t" + sha256(f).encode() + b"\n")
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="where the data go, e.g. ~/data/VCTK-DEMAND")
    ap.add_argument("--train", action="store_true", help="also the 28-speaker training set (~5 GB download)")
    ap.add_argument("--delete-zips", action="store_true", help="delete the zip files at the end")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    root = Path(args.root).expanduser()
    zips = root / "zips"
    zips.mkdir(parents=True, exist_ok=True)

    names = ["logfiles.zip", "clean_testset_wav.zip", "noisy_testset_wav.zip"]
    if args.train:
        names += ["clean_trainset_28spk_wav.zip", "noisy_trainset_28spk_wav.zip"]
    print("1. download")
    paths = {n: download(n, zips) for n in names}

    print("2. 48 kHz -> 16 kHz (librosa soxr_hq, 16-bit PCM)")
    with zipfile.ZipFile(paths["logfiles.zip"]) as z:
        for m in z.namelist():
            if m.endswith(".txt") and "__MACOSX" not in m:
                (root / "logfiles").mkdir(exist_ok=True)
                (root / "logfiles" / Path(m).name).write_bytes(z.read(m))
    counts = {}
    for kind in ("clean", "noisy"):
        counts[f"test/{kind}"] = unzip_and_resample(paths[f"{kind}_testset_wav.zip"], root / "test" / kind, root / "_48k", args.jobs)
        if args.train:
            unzip_and_resample(paths[f"{kind}_trainset_28spk_wav.zip"], root / "train" / kind, root / "_48k", args.jobs)

    if args.train:
        print("3. validation split (splits/vctk_demand_val.txt)")
        val_ids = [l.strip() for l in (HERE / "splits" / "vctk_demand_val.txt").read_text().splitlines()
                   if l.strip() and not l.startswith("#")]
        for kind in ("clean", "noisy"):
            (root / "val" / kind).mkdir(parents=True, exist_ok=True)
            for uid in val_ids:
                src, dst = root / "train" / kind / f"{uid}.wav", root / "val" / kind / f"{uid}.wav"
                if src.exists():
                    src.rename(dst)

    print("4. check")
    expected = {"test/clean": 824, "test/noisy": 824}
    if args.train:
        expected.update({"train/clean": 10000, "train/noisy": 10000, "val/clean": 1572, "val/noisy": 1572})
    lines, ok = [], True
    for folder, n in expected.items():
        have = len(list((root / folder).glob("*.wav")))
        fp = fingerprint(root / folder)
        ok &= have == n
        lines.append(f"{folder:<12} {have:>6} files (expected {n})  fingerprint {fp[:16]}")
        print(f"  [{'OK' if have == n else 'FAIL'}] {lines[-1]}")
    import librosa
    import soundfile as sf
    import soxr
    source = [f"VCTK-DEMAND, course 16 kHz version - made {time.strftime('%Y-%m-%d %H:%M')} by tools/se_eval/prepare_vctk_demand.py",
              "source: Valentini-Botinhao (2017), University of Edinburgh DataShare, https://doi.org/10.7488/ds/2117",
              "resampling: librosa.load(sr=16000, res_type='soxr_hq'), saved with soundfile as PCM_16",
              f"versions: librosa {librosa.__version__}, soxr {soxr.__version__}, soundfile {sf.__version__}",
              "validation split: " + ("splits/vctk_demand_val.txt (1,572 utterances moved from the training set)"
                                      if args.train else "- (test set only; run again with --train for train/val)"), "",
              "zip files:"] + [f"  {n}  {sha256(p)}" for n, p in paths.items()] + ["", "folders:"] + [f"  {l}" for l in lines]
    (root / "SOURCE.txt").write_text("\n".join(source) + "\n")
    if args.delete_zips:
        shutil.rmtree(zips)
    print(f"\n{'Done' if ok else 'PROBLEM: wrong number of files'} - see {root / 'SOURCE.txt'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
