"""
Week 2 - Lab 4: Mini-project - how good could a mask be?       run:  python lab4_oracle_mask.py

Most enhancement models do one thing: multiply the noisy spectrogram by a MASK - one number
between 0 and 1 per time-frequency bin - and turn the result back into a waveform:

    Y = STFT(noisy) = S + N            (speech + noise, bin by bin)
    enhanced = iSTFT( M * Y )

A real model has to GUESS M from the noisy input alone. This week we cheat: we built the mixture
ourselves, so we know S and N separately and can compute the best mask there is - the "oracle".
Its scores are the ceiling that a mask-predicting network can only approach.

    Ideal binary mask   IBM = 1 where the speech is louder than the noise (|S| > |N|), otherwise 0
    Ideal ratio mask    IRM = sqrt( |S|^2 / (|S|^2 + |N|^2) )        a soft value between 0 and 1

This lab uses stft / istft from Lab 2 and evaluate from Lab 3 - finish those first.
Your job: fill in the TODOs, then run this file.

Experiment for ANSWERS.md afterwards:   python lab4_oracle_mask.py --clean-phase
"""
import argparse
import csv

import matplotlib.pyplot as plt
import torch

from common import OUT, SR, check, load_speech, make_noise, mix_at_snr, save_wav, section, show_tf, summary
from lab2_spectrogram import istft, power_db, stft
from lab3_metrics import evaluate, print_table

SNRS = [10, 5, 0, -5]


# ---------------------------------------------------------------- Exercise 4.1
def ideal_binary_mask(S, N):
    """1.0 where the speech is louder than the noise (|S| > |N|), else 0.0. Float tensor, shape of S."""
    # TODO
    return (S.abs() > N.abs()).float()


# ---------------------------------------------------------------- Exercise 4.2
def ideal_ratio_mask(S, N):
    """sqrt( |S|^2 / (|S|^2 + |N|^2) ), with 1e-12 added to the denominator. Float tensor, shape of S."""
    # TODO
    s2, n2 = S.abs() ** 2, N.abs() ** 2
    return torch.sqrt(s2 / (s2 + n2 + 1e-12))


# ---------------------------------------------------------------- Exercise 4.3
def apply_mask(noisy, mask):
    """Enhance a 1-D noisy waveform tensor: iSTFT( mask * STFT(noisy) ), same length as noisy."""
    # TODO
    return istft(mask * stft(noisy), len(noisy))


def apply_mask_clean_phase(noisy, mask, clean):
    """(Given, for the --clean-phase experiment) Masked magnitude, but the phase of the CLEAN speech."""
    return istft((mask * stft(noisy)).abs() * torch.exp(1j * stft(clean).angle()), len(noisy))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", default="pink", help="white, pink, or the path of a noise WAV")
    ap.add_argument("--clean-phase", action="store_true", help="experiment: borrow the phase of the clean speech")
    args = ap.parse_args()

    clean_np = load_speech()
    noise_np = make_noise(len(clean_np), args.noise)
    clean = torch.from_numpy(clean_np)

    section("Exercises 4.1 and 4.2  the two masks")
    S_toy = torch.tensor([1 + 0j, 0j, 3 + 4j, 0.6j])
    N_toy = torch.tensor([1 + 0j, 2 + 0j, 0j, 0.8 + 0j])
    check("IBM on a toy example = [0, 0, 1, 0]", lambda: ideal_binary_mask(S_toy, N_toy).tolist() == [0, 0, 1, 0],
          "compare magnitudes |S| > |N| (strictly louder), then .float()")
    check("IRM on a toy example = [0.707, 0, 1, 0.6]", lambda: torch.allclose(ideal_ratio_mask(S_toy, N_toy), torch.tensor([0.7071, 0, 1, 0.6]), atol=1e-4),
          "sqrt(|S|^2 / (|S|^2 + |N|^2))")
    noisy0, noise0 = (torch.from_numpy(a) for a in mix_at_snr(clean_np, noise_np, 0))
    try:
        S0, N0 = stft(clean), stft(noise0)
    except NotImplementedError:
        check("stft() from Lab 2 works", lambda: False, "finish Lab 2 first")
        summary()
        raise SystemExit(1)
    check("masks are real, same shape as the spectrogram", lambda: all(
        not m.is_complex() and m.shape == S0.shape for m in (ideal_binary_mask(S0, N0), ideal_ratio_mask(S0, N0))))
    check("IRM stays between 0 and 1", lambda: 0 <= ideal_ratio_mask(S0, N0).min() and ideal_ratio_mask(S0, N0).max() <= 1)

    section("Exercise 4.3  apply a mask")
    try:
        same = apply_mask(noisy0, torch.ones_like(S0.real))
    except NotImplementedError:
        check("apply_mask() is written", lambda: False, "finish the TODO in Exercise 4.3 (and Lab 2)")
        summary()
        raise SystemExit(1)
    check("a mask of ones gives the noisy input back", lambda: len(same) == len(noisy0) and (same - noisy0).abs().max() < 1e-5)
    check("a mask of zeros gives silence", lambda: apply_mask(noisy0, torch.zeros_like(S0.real)).abs().max() < 1e-6)

    section("The oracle scores" + ("  (with the CLEAN phase)" if args.clean_phase else ""))
    enhance = (lambda y, m: apply_mask_clean_phase(y, m, clean)) if args.clean_phase else apply_mask
    rows, by_snr = [], {}
    for snr in SNRS:
        noisy, noise = (torch.from_numpy(a) for a in mix_at_snr(clean_np, noise_np, snr))
        S, N = stft(clean), stft(noise)
        try:
            outs = {"noisy": noisy,
                    "IBM": enhance(noisy, ideal_binary_mask(S, N)),
                    "IRM": enhance(noisy, ideal_ratio_mask(S, N))}
            by_snr[snr] = {name: evaluate(clean_np, x.numpy()) for name, x in outs.items()}
        except NotImplementedError:
            check("everything needed for the scores is written", lambda: False, "finish Exercises 4.1-4.3, and Lab 3 (evaluate)")
            summary()
            raise SystemExit(1)
        rows += [{"SNR": snr, "input": name, **m} for name, m in by_snr[snr].items()]
        if snr == 0:
            tag = "_cleanphase" if args.clean_phase else ""
            for name, x in outs.items():
                save_wav(OUT / "oracle" / f"{name.lower()}_0dB{tag if name != 'noisy' else ''}.wav", x.numpy())
            save_wav(OUT / "oracle" / "clean.wav", clean_np)
            irm0, enhanced0 = ideal_ratio_mask(S, N), outs["IRM"]
    print_table(rows)

    gain = {k: by_snr[0]["IRM"][k] - by_snr[0]["noisy"][k] for k in ("PESQ", "STOI", "SI-SDR")}
    print(f"\n  IRM at 0 dB improves PESQ by {gain['PESQ']:+.2f}, STOI by {gain['STOI']:+.3f}, SI-SDR by {gain['SI-SDR']:+.1f} dB")
    check("IRM beats the noisy input on all three metrics, at every SNR", lambda: all(
        by_snr[s]["IRM"][k] > by_snr[s]["noisy"][k] for s in SNRS for k in ("PESQ", "STOI", "SI-SDR")))
    check("IRM at 0 dB: SI-SDR up by more than 5 dB", lambda: gain["SI-SDR"] > 5)

    csv_name = "week2_oracle_cleanphase.csv" if args.clean_phase else "week2_oracle.csv"
    with open(OUT / csv_name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    top = power_db(stft(noisy0)).max().item()
    show_tf(axes[0, 0], power_db(S0), "clean", vmin=top - 80, vmax=top)
    show_tf(axes[0, 1], power_db(stft(noisy0)), "noisy (0 dB)", vmin=top - 80, vmax=top)
    im = show_tf(axes[1, 0], irm0, "ideal ratio mask (white = keep, black = remove)", vmin=0, vmax=1, cmap="gray")
    fig.colorbar(im, ax=axes[1, 0])
    show_tf(axes[1, 1], power_db(stft(enhanced0)), "noisy x IRM", vmin=top - 80, vmax=top)
    png_name = "week2_oracle_cleanphase.png" if args.clean_phase else "week2_oracle.png"
    fig.tight_layout(); fig.savefig(OUT / png_name, dpi=110); plt.close(fig)
    print(f"\n  saved out/{csv_name}, out/{png_name} and out/oracle/*.wav")
    summary()
