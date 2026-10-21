"""
Week 3 - Lab 4: Mini-project - the exam                          run:  python lab4_evaluate.py

Two trained networks (out/mlp.pt from Lab 2, out/gru.pt from Lab 3) take the same exam:

  - the Week 2 sentence - never used for training
  - pink noise at 10, 5, 0 and -5 dB     the KIND of noise they trained on (stationary, tilted spectrum)
  - babble at 5 and 0 dB                 six people talking at once - a kind of noise they have NEVER heard
  - the oracle IRM from Week 2 as the ceiling (it cheats: it knows the clean speech)

To compare conditions with very different starting points, we ask how much of the way from
"doing nothing" to "the oracle" each network got:

    fraction of ceiling = (model - noisy) / (oracle - noisy)
    0 = no better than the noisy input,  1 = as good as the oracle,  below 0 = made it worse

Your job: fill in the TODOs, then run this file (after Labs 2 and 3 - it loads their trained models).
"""
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (BABBLE, DEVICE, OUT, TEST_SPEECH, check, evaluate, ideal_ratio_mask, istft, level_db, load_audio,
                    make_noise, mix_at_snr, print_table, save_wav, section, show_tf, stft, summary)
from lab2_mlp import FrameMLP, enhance_test
from lab3_gru import MaskGRU

CONDITIONS = [("pink", 10), ("pink", 5), ("pink", 0), ("pink", -5), ("babble", 5), ("babble", 0)]
METRICS = ("PESQ", "STOI", "SI-SDR")


# ---------------------------------------------------------------- Exercise 4.1
def make_condition(kind, snr):
    """Return (clean, noisy, noise) for one exam condition - three 1-D float32 numpy arrays.

      clean = load_audio(TEST_SPEECH)
      noise:  "pink"   -> make_noise(len(clean), tilt=1.0)          (exactly the Week 2 pink noise)
              "babble" -> the first len(clean) samples of load_audio(BABBLE)
      noisy, noise = mix_at_snr(clean, noise, snr)
    """
    # TODO
    raise NotImplementedError


def oracle_enhance(clean, noisy, noise):
    """(Given) The Week 2 ceiling: noisy x ideal ratio mask."""
    S, N = stft(torch.from_numpy(clean)), stft(torch.from_numpy(noise))
    return istft(ideal_ratio_mask(S, N) * stft(torch.from_numpy(noisy)), len(noisy)).numpy()


# ---------------------------------------------------------------- Exercise 4.2
def score_condition(models, kind, snr):
    """Score every system on one condition. Return a list of dicts, one per system:

        {"noise": kind, "SNR": snr, "system": name, "PESQ": ..., "STOI": ..., "SI-SDR": ...}

    Systems, in this order:
      "noisy"   the noisy input itself (no processing)
      each model in `models` (a dict {name: model}):   enhanced = enhance_test(model, noisy)[0]
      "oracle"  oracle_enhance(clean, noisy, noise)
    Score each with evaluate(clean, output).
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 4.3
def fraction_of_ceiling(model_score, noisy_score, oracle_score):
    """(model - noisy) / (oracle - noisy): how much of the possible improvement the model achieved."""
    # TODO
    raise NotImplementedError


def load_models():
    """(Given) The two trained networks from Labs 2 and 3."""
    models = {}
    for name, cls, path in [("MLP", FrameMLP, OUT / "mlp.pt"), ("GRU", MaskGRU, OUT / "gru.pt")]:
        if not path.exists():
            raise SystemExit(f"{path} not found - run lab2_mlp.py and lab3_gru.py first")
        model = cls()
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        models[name] = model.to(DEVICE).eval()
    return models


def find(rows, kind, snr, system):
    """(Given) The row for one condition and system."""
    return next(r for r in rows if r["noise"] == kind and r["SNR"] == snr and r["system"] == system)


if __name__ == "__main__":
    models = load_models()
    print("loaded out/mlp.pt and out/gru.pt | device:", DEVICE)

    section("Exercise 4.1  the exam conditions")
    try:
        clean, noisy, noise = make_condition("pink", 0)
        _, noisy_b, noise_b = make_condition("babble", 5)
    except NotImplementedError:
        check("make_condition() is written", lambda: False, "finish the TODO in Exercise 4.1")
        summary()
        raise SystemExit(1)
    check("three arrays of the same length", lambda: len(clean) == len(noisy) == len(noise) == len(noisy_b) == 64880)
    check("pink at 0 dB measures 0 dB", lambda: abs(level_db(clean) - level_db(noise)) < 0.01)
    check("babble at 5 dB measures 5 dB", lambda: abs(level_db(clean) - level_db(noise_b) - 5) < 0.01)
    week2_noise = make_noise(len(clean), tilt=1.0)
    check("pink is exactly the Week 2 noise", lambda: np.allclose(noise / np.abs(noise).max(), week2_noise / np.abs(week2_noise).max(), atol=1e-6),
          "make_noise(len(clean), tilt=1.0) - no other random generator")

    section("Exercise 4.2  scoring every system on every condition")
    rows = []
    try:
        for kind, snr in CONDITIONS:
            rows += score_condition(models, kind, snr)
            print(f"  {kind:>6} {snr:>3} dB done")
    except NotImplementedError:
        check("score_condition() is written", lambda: False, "finish the TODO in Exercise 4.2")
        summary()
        raise SystemExit(1)
    print_table(rows)
    check("4 systems x 6 conditions = 24 rows", lambda: len(rows) == 24 and {r["system"] for r in rows} == {"noisy", "MLP", "GRU", "oracle"})
    check("matches Week 2: noisy and oracle at pink 0 dB", lambda: abs(find(rows, "pink", 0, "noisy")["PESQ"] - 1.05) < 0.01
          and abs(find(rows, "pink", 0, "oracle")["PESQ"] - 2.90) < 0.01, "same sentence, same pink noise, same STFT as Week 2")
    check("the oracle beats both networks everywhere (PESQ)", lambda: all(
        find(rows, k, s, "oracle")["PESQ"] > max(find(rows, k, s, m)["PESQ"] for m in models) for k, s in CONDITIONS))
    check("the GRU beats doing nothing on STOI for every pink condition", lambda: all(
        find(rows, "pink", s, "GRU")["STOI"] > find(rows, "pink", s, "noisy")["STOI"] for k, s in CONDITIONS if k == "pink"))
    with open(OUT / "week3_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    section("Exercise 4.3  fraction of the ceiling reached")
    try:
        fractions = [{"noise": k, "SNR": s, "system": m,
                      **{x: fraction_of_ceiling(find(rows, k, s, m)[x], find(rows, k, s, "noisy")[x], find(rows, k, s, "oracle")[x]) for x in METRICS}}
                     for k, s in CONDITIONS for m in models]
    except NotImplementedError:
        check("fraction_of_ceiling() is written", lambda: False, "finish the TODO in Exercise 4.3")
        summary()
        raise SystemExit(1)
    print_table(fractions)
    check("0 when the model equals noisy, 1 when it equals the oracle", lambda: fraction_of_ceiling(2.0, 2.0, 3.0) == 0 and fraction_of_ceiling(3.0, 2.0, 3.0) == 1)
    frac = {(r["noise"], r["SNR"], r["system"]): r for r in fractions}
    print(f"\n  STOI fraction of ceiling at 5 dB, GRU:  pink {frac[('pink', 5, 'GRU')]['STOI']:.2f}   "
          f"babble {frac[('babble', 5, 'GRU')]['STOI']:.2f}   (questions Q6, Q7)")
    pink_avg = {m: np.mean([frac[("pink", s, m)]["STOI"] for k, s in CONDITIONS if k == "pink"]) for m in models}
    print(f"  average STOI fraction over the four pink SNRs:  MLP {pink_avg['MLP']:.2f}   GRU {pink_avg['GRU']:.2f}")
    check("on pink, the GRU reaches more of the STOI ceiling than the MLP (average of the 4 SNRs)",
          lambda: pink_avg["GRU"] > pink_avg["MLP"])
    check("babble fools both networks: under 20 % of the STOI ceiling", lambda: all(
        frac[("babble", s, m)]["STOI"] < 0.2 for k, s in CONDITIONS if k == "babble" for m in models))

    section("Plots and sound files")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    snrs = [s for k, s in CONDITIONS if k == "pink"]
    for ax, metric in zip(axes, METRICS):
        for system, style in [("noisy", "k:o"), ("MLP", "-s"), ("GRU", "-^"), ("oracle", "k--o")]:
            ax.plot(snrs, [find(rows, "pink", s, system)[metric] for s in snrs], style, label=system)
        ax.set(title=f"{metric} - pink noise (trained kind)", xlabel="input SNR (dB)")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout(); fig.savefig(OUT / "week3_results.png", dpi=110); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex=True, sharey=True)
    for row, (kind, snr) in enumerate([("pink", 0), ("babble", 0)]):
        c, n, nz = make_condition(kind, snr)
        _, gru_mask = enhance_test(models["GRU"], n)
        Y = stft(torch.from_numpy(n))
        show_tf(axes[row, 0], 10 * torch.log10(Y.abs() ** 2 + 1e-10), f"noisy: {kind} 0 dB")
        show_tf(axes[row, 1], gru_mask, "GRU mask", vmin=0, vmax=1, cmap="gray")
        show_tf(axes[row, 2], ideal_ratio_mask(stft(torch.from_numpy(c)), stft(torch.from_numpy(nz))), "oracle mask", vmin=0, vmax=1, cmap="gray")
        save_wav(OUT / "exam" / f"{kind}_0dB_noisy.wav", n)
        for name, model in models.items():
            save_wav(OUT / "exam" / f"{kind}_0dB_{name}.wav", enhance_test(model, n)[0])
        save_wav(OUT / "exam" / f"{kind}_0dB_oracle.wav", oracle_enhance(c, n, nz))
    fig.tight_layout(); fig.savefig(OUT / "week3_masks.png", dpi=110); plt.close(fig)
    print("  saved out/week3_results.csv, out/week3_results.png, out/week3_masks.png and out/exam/*.wav")
    summary()
