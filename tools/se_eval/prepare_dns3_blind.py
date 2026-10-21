"""
Download the DNS Challenge 3 (Interspeech 2021) blind test set: 700 noisy recordings, 16 kHz, NO clean references.
If the download stops (network), just run it again: files already there are skipped.

    uv run --project <this folder> <this folder>/prepare_dns3_blind.py --root ~/data/DNS3-blind      (~400 MB)

Score it with DNSMOS only:   se_eval.py --enhanced <your outputs> --dnsmos ...   (no --clean)

Source: microsoft/DNS-Challenge, branch interspeech2021/adddata,
datasets/wideband/blind_testset_wideband/track_1/noisy (Git LFS), licensed CC BY 4.0.
Papers do not always use all of it (some report only the "no reverb" part, some all 700 clips):
check which subset YOUR paper used before you compare with its table, and write it down.
"""
import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

LISTING = ("https://api.github.com/repos/microsoft/DNS-Challenge/contents/datasets/wideband/"
           "blind_testset_wideband/track_1/noisy?ref=interspeech2021/adddata")
MEDIA = ("https://media.githubusercontent.com/media/microsoft/DNS-Challenge/interspeech2021/adddata/datasets/"
         "wideband/blind_testset_wideband/track_1/noisy/{}")


def fetch(url, tries=4):
    """Download url into memory; on a network error wait and try again (2, 4, 8 s)."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "course-se-eval"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except OSError as e:                 # URLError, timeouts and dropped connections are all OSError
            if attempt == tries - 1:
                raise
            print(f"  network error ({e}); trying again in {2 ** (attempt + 1)} s", flush=True)
            time.sleep(2 ** (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="where the data go, e.g. ~/data/DNS3-blind")
    args = ap.parse_args()
    out = Path(args.root).expanduser() / "noisy"
    out.mkdir(parents=True, exist_ok=True)

    names = sorted(e["name"] for e in json.loads(fetch(LISTING)) if e["name"].endswith(".wav"))
    print(f"{len(names)} files listed; downloading the missing ones ...")

    def one(name):
        path = out / name
        if path.exists() and path.stat().st_size > 1000:
            return name
        data = fetch(MEDIA.format(name))
        if not data.startswith(b"RIFF"):
            raise RuntimeError(f"{name}: not a WAV file (got {data[:40]!r})")
        path.write_bytes(data)
        return name

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for k, _ in enumerate(pool.map(one, names), 1):
            if k % 100 == 0 or k == len(names):
                print(f"  {k}/{len(names)}", flush=True)

    import soundfile as sf
    bad = [f.name for f in sorted(out.glob("*.wav")) if (lambda i: i.samplerate != 16000 or i.channels != 1)(sf.info(f))]
    files = sorted(out.glob("*.wav"))
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b"\t" + hashlib.sha256(f.read_bytes()).hexdigest().encode() + b"\n")
    lines = [f"DNS Challenge 3 (Interspeech 2021) blind test set, track 1 - downloaded {time.strftime('%Y-%m-%d %H:%M')}",
             "source: github.com/microsoft/DNS-Challenge, branch interspeech2021/adddata, "
             "datasets/wideband/blind_testset_wideband/track_1/noisy (CC BY 4.0)",
             f"files: {len(files)} (listed {len(names)}), not 16 kHz mono: {len(bad)}",
             f"fingerprint: {h.hexdigest()[:16]}"]
    (out.parent / "SOURCE.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if bad or len(files) != len(names):
        sys.exit("PROBLEM: see above")


if __name__ == "__main__":
    main()
