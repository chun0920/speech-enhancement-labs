"""
Week 2 - Lab 3: Measuring quality - SI-SDR, PESQ, STOI        run:  python lab3_metrics.py

"It sounds better" is not a result. Enhancement papers report numbers, almost always these three:

  SI-SDR  scale-invariant signal-to-distortion ratio, in dB, higher = better. Pure arithmetic on the
          samples: how much of the estimate is the reference, and how much is everything else.
  PESQ    perceptual evaluation of speech quality (ITU-T P.862). A model of how listeners would rate
          the QUALITY, from about 1.0 (bad) to 4.64 (identical). Wide-band ("wb") mode for 16 kHz.
  STOI    short-time objective intelligibility, 0 to 1. Predicts how much of the speech a listener
          would UNDERSTAND - intelligibility, not quality.

All three compare an estimate with the clean reference:   metric(reference, estimate).
The order of the arguments matters.

Your job: fill in the TODOs, then run this file.
"""
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
from pesq import pesq
from pystoi import stoi

from common import OUT, SR, check, level_db, load_speech, make_noise, mix_at_snr, section, summary

SNRS = [20, 10, 5, 0, -5, -10]


# ---------------------------------------------------------------- Exercise 3.1
def si_sdr(ref, est, eps=1e-8):
    """Scale-invariant SDR in dB (1-D numpy arrays of equal length).

    1. remove the mean from both:          ref = ref - ref.mean(),  same for est
    2. best scaling of ref to match est:   alpha = dot(est, ref) / dot(ref, ref)
    3. target = alpha * ref                (the part of est that IS the reference)
       residual = est - target             (everything else: noise, distortion)
    4. return 10 * log10( (sum(target**2) + eps) / (sum(residual**2) + eps) )

    Work in float64 (ref.astype(np.float64)) so long signals don't lose precision.
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 3.2
def evaluate(ref, est):
    """Return {"PESQ": ..., "STOI": ..., "SI-SDR": ...} for one estimate.

    PESQ:   pesq(SR, ref, est, "wb")                        (sample rate first!)
    STOI:   stoi(ref, est, SR, extended=False)              (sample rate last!)
    SI-SDR: your si_sdr(ref, est)
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 3.3
def score_noisy(clean, noise, snrs):
    """Score the UNPROCESSED noisy mixture at each SNR. Return a list of dicts, one per SNR:

        {"SNR": snr, "PESQ": ..., "STOI": ..., "SI-SDR": ...}

    For each snr: noisy, _ = mix_at_snr(clean, noise, snr), then evaluate(clean, noisy).
    Hint: {"SNR": snr, **some_dict} merges a dict into a new one.
    """
    rows = []
    for snr in snrs:
        # TODO
        raise NotImplementedError
    return rows


def plain_snr(ref, est):
    """(Given) The plain SNR from Week 1: no rescaling, no mean removal."""
    return level_db(ref) - level_db(est - ref)


def print_table(rows, first="SNR"):
    """(Given) Print a list of dicts as a table."""
    keys = list(rows[0])
    print("  " + "".join(f"{k:>10}" for k in keys))
    for r in rows:
        print("  " + "".join(f"{r[k]:>10}" if isinstance(r[k], str) else f"{r[k]:>10.2f}" for k in keys))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", default="pink", help="white, pink, or the path of a noise WAV")
    args = ap.parse_args()

    clean = load_speech()
    noise = make_noise(len(clean), args.noise)
    noisy0, _ = mix_at_snr(clean, noise, 0)

    section("Exercise 3.1  SI-SDR")
    white10, _ = mix_at_snr(clean, make_noise(len(clean), "white"), 10)
    check("si_sdr(clean, clean) is huge (> 60 dB)", lambda: si_sdr(clean, clean) > 60)
    check("white noise at 10 dB SNR -> SI-SDR close to 10 dB", lambda: abs(si_sdr(clean, white10) - 10) < 0.3,
          "alpha = dot(est, ref) / dot(ref, ref); target = alpha * ref")
    check("turning the volume down does not change it", lambda: abs(si_sdr(clean, 0.5 * noisy0) - si_sdr(clean, noisy0)) < 1e-6,
          "that is what 'scale-invariant' means - did you scale ref by alpha?")
    check("returns a single number in dB", lambda: np.ndim(si_sdr(clean, noisy0)) == 0)

    section("Exercise 3.2  evaluate")
    try:
        perfect = evaluate(clean, clean)
        m0 = evaluate(clean, noisy0)
    except NotImplementedError:
        check("evaluate() is written", lambda: False, "finish the TODOs in Exercises 3.1 and 3.2")
        summary()
        raise SystemExit(1)
    print("  clean vs clean :", {k: round(float(v), 3) for k, v in perfect.items()})
    print("  clean vs 0 dB  :", {k: round(float(v), 3) for k, v in m0.items()})
    check("perfect estimate: PESQ about 4.64, STOI about 1", lambda: perfect["PESQ"] > 4.5 and perfect["STOI"] > 0.99)
    check("PESQ called as pesq(SR, ref, est, 'wb')", lambda: abs(m0["PESQ"] - pesq(SR, clean, noisy0, "wb")) < 1e-9)
    check("STOI called as stoi(ref, est, SR)", lambda: abs(m0["STOI"] - stoi(clean, noisy0, SR, extended=False)) < 1e-9)

    section("Exercise 3.3  noisy speech at six SNRs")
    try:
        rows = score_noisy(clean, noise, SNRS)
    except NotImplementedError:
        check("score_noisy() is written", lambda: False, "finish the TODO in Exercise 3.3")
        summary()
        raise SystemExit(1)
    print_table(rows)
    check("one row per SNR, in order", lambda: [r["SNR"] for r in rows] == SNRS)
    check("PESQ and STOI fall as the SNR falls", lambda: all(a["PESQ"] > b["PESQ"] and a["STOI"] > b["STOI"] for a, b in zip(rows, rows[1:])))
    check("SI-SDR of the noisy input is close to its SNR", lambda: all(abs(r["SI-SDR"] - r["SNR"]) < 1 for r in rows))
    with open(OUT / "lab3_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for ax, key in zip(axes, ("PESQ", "STOI", "SI-SDR")):
        ax.plot([r["SNR"] for r in rows], [r[key] for r in rows], "o-")
        ax.set(title=key, xlabel="input SNR (dB)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Unprocessed noisy speech ({args.noise} noise)"); fig.tight_layout()
    fig.savefig(OUT / "lab3_metrics.png", dpi=110); plt.close(fig)
    print("  saved out/lab3_metrics.csv and out/lab3_metrics.png (question Q6)")

    section("Break it on purpose: the same 5 dB mixture, four ways")
    noisy5, _ = mix_at_snr(clean, noise, 5)
    delayed = np.concatenate([np.zeros(16, np.float32), noisy5[:-16]])  # 16 samples = 1 ms late
    cases = [("as is", clean, noisy5), ("volume x 2", clean, 2 * noisy5),
             ("1 ms late", clean, delayed), ("args swapped", noisy5, clean)]
    table = [{"case": name, "plain SNR": plain_snr(ref, est), **evaluate(ref, est)} for name, ref, est in cases]
    print_table(table)
    print("  Which numbers moved, and which stayed put? (question Q7)")
    summary()
