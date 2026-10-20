"""
Week 1 - Lab 4: Mini-project - a small paired noisy dataset      run:  python lab4_noisy_mix.py

A speech-enhancement dataset is a list of PAIRS:   noisy input  ->  clean target.
How noisy the input is, is set by the signal-to-noise ratio (SNR). Audio engineers
think about it in LEVELS, measured in decibels:

    level_dB(x) = 20 * log10( RMS(x) ),          RMS(x) = sqrt( mean(x^2) )
    SNR_dB      = level_dB(clean) - level_dB(noise)

     20 dB : the noise is far below the speech - slightly noisy
      0 dB : speech and noise at the same level - hard
    -10 dB : the noise is far above the speech - very hard

To hit a target SNR the noise has to sit at  level_dB(clean) - SNR.  So change the noise
level by   gain_dB = level_dB(clean) - SNR - level_dB(noise),
i.e. multiply the noise by   10 ** (gain_dB / 20).

Inputs: out/speech_16k.wav (made by Lab 3) and data/noise.wav (optional - pink noise otherwise).
Your job: fill in the TODOs, then run this file.
"""
import argparse
from math import gcd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from common import DATA, OUT, check, section, summary

SR = 16000
SNRS = [20, 10, 0, -10]
NOISE_FILE = DATA / "noise.wav"
PAIRS = OUT / "pairs"


def pink_noise(length):
    """(Given) Pink noise - equal energy in every octave. Sounds like steady rain or a waterfall."""
    spec = np.fft.rfft(np.random.randn(length))
    f = np.fft.rfftfreq(length)
    spec[0] = 0
    spec[1:] /= np.sqrt(f[1:])
    x = np.fft.irfft(spec, length)
    return (x / x.std()).astype(np.float32)


def load_noise(length, sr=SR):
    """(Given) A random piece of noise, mono float32, exactly `length` samples at `sr`."""
    if not NOISE_FILE.exists():
        return pink_noise(length)
    n, nsr = sf.read(NOISE_FILE, dtype="float32")
    if n.ndim == 2:
        n = n.mean(axis=1)
    g = gcd(nsr, sr)
    n = resample_poly(n, sr // g, nsr // g)
    n = np.tile(n, int(np.ceil(length / len(n))))      # repeat if the recording is too short
    start = np.random.randint(0, len(n) - length + 1)  # random starting point
    return n[start:start + length].astype(np.float32)


# ---------------------------------------------------------------- Exercise 4.1
def level_db(x):
    """Level of x in dB:  20 * log10( sqrt(mean(x^2)) ).

    Add 1e-12 inside the log so that pure silence gives a very low number instead of -inf.
    """
    # TODO
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


def measure_snr(clean, noisy):
    """(Given) Measured SNR in dB. The noise is whatever is left over: noisy - clean."""
    return level_db(clean) - level_db(noisy - clean)


# ---------------------------------------------------------------- Exercise 4.2
def mix_at_snr(clean, noise, snr_db):
    """Change the level of `noise` so that clean + noise has the requested SNR.

    Use level_db() and the gain_dB formula at the top of this file.
    Return (noisy, scaled_noise).
    """
    # TODO
    gain_db = level_db(clean) - snr_db - level_db(noise)
    scaled = noise * 10 ** (gain_db / 20)
    return clean + scaled, scaled


# ---------------------------------------------------------------- Exercise 4.3
def make_pair(clean, noise, snr_db, peak=0.9):
    """Make ONE training pair (noisy, target) at the requested SNR, at a safe volume.

    1. noisy = clean and noise mixed at snr_db
    2. g = peak / max(|noisy|)         -> the loudest sample of noisy becomes exactly `peak`
    3. return (noisy * g, clean * g)   -> the SAME g for both, so the pair still matches
    """
    # TODO
    noisy, _ = mix_at_snr(clean, noise, snr_db)
    g = peak / np.abs(noisy).max()
    return noisy * g, clean * g


# ---------------------------------------------------------------- Exercise 4.4
def make_dataset(clean, snrs, out_dir):
    """Save one pair per SNR, plus a manifest: the table of contents every dataset has.

    For each snr in snrs:
      1. noise = load_noise(len(clean))                (a fresh random piece of noise each time)
      2. noisy, target = make_pair(clean, noise, snr)
      3. save  out_dir / f"noisy_{snr}dB.wav"  and  out_dir / f"clean_{snr}dB.wav"
         (16 kHz, subtype="PCM_16")
      4. rows.append((f"noisy_{snr}dB.wav", f"clean_{snr}dB.wav", snr))
    Then write out_dir / "manifest.csv": the first line is   noisy,clean,snr_db
    followed by one line per row, e.g.   noisy_20dB.wav,clean_20dB.wav,20
    Return rows.
    """
    out_dir.mkdir(exist_ok=True)
    rows = []
    for snr in snrs:
        # TODO: steps 1-4
        noise = load_noise(len(clean))
        noisy, target = make_pair(clean, noise, snr)
        sf.write(out_dir / f"noisy_{snr}dB.wav", noisy, SR, subtype="PCM_16")
        sf.write(out_dir / f"clean_{snr}dB.wav", target, SR, subtype="PCM_16")
        rows.append((f"noisy_{snr}dB.wav", f"clean_{snr}dB.wav", snr))
    # TODO: write manifest.csv
    with open(out_dir / "manifest.csv", "w") as f:
        f.write("noisy,clean,snr_db\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    return rows


def read_manifest(path):
    """(Given) Read manifest.csv back -> list of (noisy_file, clean_file, snr)."""
    lines = path.read_text().strip().splitlines()
    return [(a, b, float(c)) for a, b, c in (line.split(",") for line in lines[1:])]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=str(OUT / "speech_16k.wav"), help="16 kHz mono clean speech (made by Lab 3)")
    args = ap.parse_args()

    np.random.seed(0)
    wav = Path(args.wav)
    if not wav.exists():
        raise SystemExit(f"{wav} not found - run lab3_audio_io.py first (or pass --wav your_16k_file.wav)")
    clean, sr = sf.read(wav, dtype="float32")
    assert sr == SR and clean.ndim == 1, "need a mono 16 kHz file"
    print(f"clean speech: {len(clean) / SR:.2f} s from {wav}")
    print(f"noise: {NOISE_FILE}" if NOISE_FILE.exists() else "data/noise.wav not found -> using pink noise")
    noise = load_noise(len(clean))
    sf.write(OUT / "noise_sample.wav", noise / max(1.0, np.abs(noise).max()), SR)

    section("Exercise 4.1  level in dB")
    sine = np.sin(2 * np.pi * 440 * np.arange(SR) / SR).astype(np.float32)
    check("a full-scale sine sits at -3.01 dB", lambda: abs(level_db(sine) + 3.0103) < 0.01, "RMS of a sine = amplitude / sqrt(2)")
    check("half the amplitude = 6.02 dB lower", lambda: abs(level_db(sine) - level_db(0.5 * sine) - 6.0206) < 0.01, "20 * log10, not 10 * log10")
    check("silence gives a finite number", lambda: np.isfinite(level_db(np.zeros(100))), "add 1e-12 inside the log")

    section("Exercise 4.2  mix at a target SNR")
    try:
        for target in [10, 0, -5]:
            noisy, scaled = mix_at_snr(clean, noise, target)
            check(f"mix at {target:>3} dB measures {target} dB", lambda: abs(measure_snr(clean, noisy) - target) < 0.01,
                  "gain_dB = level(clean) - SNR - level(noise);  factor = 10 ** (gain_dB / 20)")
        noisy5, scaled5 = mix_at_snr(clean, noise, 5)
        check("noisy - clean == scaled noise", lambda: np.allclose(noisy5 - clean, scaled5, atol=1e-5))
    except NotImplementedError:
        check("mix_at_snr() is written", lambda: False, "finish the TODOs in Exercises 4.1 and 4.2")
        summary()
        raise SystemExit(1)

    section("Exercise 4.3  one pair at a safe volume")
    try:
        noisy_p, target_p = make_pair(clean, noise, 5)
    except NotImplementedError:
        check("make_pair() is written", lambda: False, "finish the TODO in Exercise 4.3")
        summary()
        raise SystemExit(1)
    g_expected = 0.9 / np.abs(noisy5).max()
    check("noisy peaks at exactly 0.9", lambda: abs(np.abs(noisy_p).max() - 0.9) < 1e-5)
    check("target = clean x the same gain", lambda: np.allclose(target_p, clean * g_expected, atol=1e-6), "return clean * g, not clean")
    check("the pair still measures 5 dB", lambda: abs(measure_snr(target_p, noisy_p) - 5) < 0.01, "scale noisy and clean by the SAME g")

    section("Exercise 4.4  the dataset")
    try:
        rows = make_dataset(clean, SNRS, PAIRS)
    except NotImplementedError:
        check("make_dataset() is written", lambda: False, "finish the TODOs in Exercise 4.4")
        summary()
        raise SystemExit(1)
    check("8 WAV files saved", lambda: all((PAIRS / f"{kind}_{s}dB.wav").exists() for s in SNRS for kind in ("noisy", "clean")),
          "did you call sf.write twice per SNR?")
    check("manifest.csv: a header and 4 rows", lambda: (PAIRS / "manifest.csv").read_text().strip().splitlines()[0] == "noisy,clean,snr_db"
          and len(read_manifest(PAIRS / "manifest.csv")) == len(SNRS))
    check("every saved pair measures its SNR (within 0.05 dB)", lambda: all(
        abs(measure_snr(sf.read(PAIRS / c, dtype="float32")[0], sf.read(PAIRS / n, dtype="float32")[0]) - s) < 0.05
        for n, c, s in read_manifest(PAIRS / "manifest.csv")), "noisy and clean must be scaled by the same g")
    check("no clipping", lambda: all(np.abs(sf.read(PAIRS / f"noisy_{s}dB.wav")[0]).max() < 1.0 - 1e-4 for s in SNRS))

    fig, axes = plt.subplots(len(SNRS), 1, figsize=(12, 8), sharex=True, sharey=True)
    t = np.arange(len(clean)) / SR
    for ax, snr in zip(axes, SNRS):
        if (PAIRS / f"noisy_{snr}dB.wav").exists() and (PAIRS / f"clean_{snr}dB.wav").exists():
            ax.plot(t, sf.read(PAIRS / f"noisy_{snr}dB.wav")[0], lw=0.5, label="noisy input")
            ax.plot(t, sf.read(PAIRS / f"clean_{snr}dB.wav")[0], lw=0.5, color="k", label="clean target")
        ax.set_ylabel(f"{snr} dB")
        ax.set_ylim(-1, 1)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Each pair: noisy input and its clean target"); fig.tight_layout()
    fig.savefig(OUT / "week1_pairs.png", dpi=120)
    print("\n  saved out/pairs/ (8 WAV files + manifest.csv) and out/week1_pairs.png")
    summary()
