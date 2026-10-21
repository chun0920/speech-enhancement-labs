"""
Week 9 - Lab 1: Mismatched SNR & Noise Stress Testing                run:  python lab1_mismatched.py

In standard training (Weeks 4–5), models are exposed to moderate SNR levels (e.g. 0 to 15 dB).
However, real-world deployment frequently encounters extreme acoustic conditions:
  - Severe acoustic mismatch: -5 dB to 0 dB (cocktail party, construction, airport tarmac)
  - Unseen non-stationary noise types (high-pitch sirens, machine hum, reverberant babble)

In this lab, you:
  1. Implement an exact SNR-controlled speech-noise mixture synthesizer
  2. Perform systematic stress testing across extreme SNR regimes (-5 dB to +15 dB)
  3. Analyze performance degradation curves and identify the model's operational breakdown point
  4. Export results/stress_test_report.md
"""
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

_passed = []


def check(name, fn, hint=""):
    try:
        ok = bool(fn())
    except NotImplementedError:
        ok, hint = False, "not written yet - look for the # TODO"
    except Exception as e:
        ok, hint = False, f"{type(e).__name__}: {e}  {hint}".strip()
    _passed.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok and hint:
        print(f"         hint: {hint}")
    return ok


# ---------------------------------------------------------------- Exercise 1.1
def synthesize_noisy_mixture(clean, noise, snr_db):
    """Mix clean speech and noise waveforms at an exact target SNR (dB).

    Formula:
      P_clean = mean(clean^2)
      P_noise = mean(noise^2)
      target_scale = sqrt(P_clean / (P_noise * 10^(snr_db / 10)))
      noisy = clean + target_scale * noise

    Args:
      clean: 1D numpy array or torch tensor (clean speech)
      noise: 1D numpy array or torch tensor of same length
      snr_db: target Signal-to-Noise Ratio in decibels

    Returns:
      noisy: mixed waveform with the exact target SNR
      actual_snr_db: measured SNR of the mixture in dB
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.2
def evaluate_snr_stress(enhance_fn, clean_signals, noise_signals, snr_levels=(-5, 0, 5, 10, 15), metric_fn=None):
    """Evaluate model performance across a ladder of SNR levels.

    Args:
      enhance_fn: callable taking 1D waveform -> enhanced 1D waveform
      clean_signals: list of clean 1D arrays
      noise_signals: list of noise 1D arrays
      snr_levels: tuple/list of SNR values in dB
      metric_fn: callable(clean, test_wav) -> float score (e.g. PESQ or SI-SDR proxy)

    Returns:
      dict: {snr_db: {"noisy_score": float, "enh_score": float, "delta": float}}
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.3
def format_stress_test_report(stress_results, out_path):
    """Format stress test results into a Markdown report and save to file."""
    # TODO
    raise NotImplementedError


# ================================================================ Test Suite
if __name__ == "__main__":
    print("--- Testing Exercise 1.1: synthesize_noisy_mixture ---")
    np.random.seed(42)
    t = np.linspace(0, 1, 16000, endpoint=False)
    clean_toy = np.sin(2 * np.pi * 440 * t)  # 440 Hz pure tone
    noise_toy = np.random.randn(16000)

    try:
        mix_0, snr_0 = synthesize_noisy_mixture(clean_toy, noise_toy, snr_db=0.0)
    except NotImplementedError:
        mix_0, snr_0 = None, 999.0
    try:
        mix_10, snr_10 = synthesize_noisy_mixture(clean_toy, noise_toy, snr_db=10.0)
    except NotImplementedError:
        mix_10, snr_10 = None, 999.0
    try:
        mix_neg5, snr_neg5 = synthesize_noisy_mixture(clean_toy, noise_toy, snr_db=-5.0)
    except NotImplementedError:
        mix_neg5, snr_neg5 = None, 999.0

    check("synthesizes 0 dB mixture within 0.1 dB",
          lambda: abs(snr_0 - 0.0) < 0.1,
          f"expected SNR ~0.0 dB, got {snr_0:.2f} dB")
    check("synthesizes +10 dB mixture within 0.1 dB",
          lambda: abs(snr_10 - 10.0) < 0.1,
          f"expected SNR ~10.0 dB, got {snr_10:.2f} dB")
    check("synthesizes -5 dB mixture within 0.1 dB",
          lambda: abs(snr_neg5 - (-5.0)) < 0.1,
          f"expected SNR ~-5.0 dB, got {snr_neg5:.2f} dB")

    print("\n--- Testing Exercise 1.2: evaluate_snr_stress ---")
    # Toy enhance function: simple spectral subtraction proxy / Wiener gain filter
    def toy_denoiser(wav):
        return wav * 0.9

    dummy_clean = [clean_toy, np.sin(2 * np.pi * 880 * t)]
    dummy_noise = [noise_toy, np.random.randn(16000)]

    try:
        stress_res = evaluate_snr_stress(toy_denoiser, dummy_clean, dummy_noise, snr_levels=(-5, 0, 10))
    except NotImplementedError:
        stress_res = {}

    check("evaluates all requested SNR levels (-5, 0, 10)",
          lambda: set(stress_res.keys()) == {-5, 0, 10},
          "expected results for keys -5, 0, 10")
    check("computes delta for each SNR level",
          lambda: len(stress_res) > 0 and all("delta" in stress_res[k] for k in stress_res),
          "expected 'delta' field in result dictionary")

    print("\n--- Testing Exercise 1.3: format_stress_test_report ---")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out_report = Path(tmp) / "results" / "stress_test_report.md"
        sample_res = {
            -5: {"noisy_score": 1.25, "enh_score": 1.70, "delta": 0.45},
            0: {"noisy_score": 1.55, "enh_score": 2.20, "delta": 0.65},
            5: {"noisy_score": 1.95, "enh_score": 2.70, "delta": 0.75},
            10: {"noisy_score": 2.35, "enh_score": 2.95, "delta": 0.60},
            15: {"noisy_score": 2.75, "enh_score": 3.10, "delta": 0.35},
        }
        try:
            report_md = format_stress_test_report(sample_res, out_report)
            has_report = out_report.exists() and "-5 dB" in out_report.read_text(encoding="utf-8")
        except NotImplementedError:
            report_md, has_report = "", False

        check("formats markdown report with all SNR rows",
              lambda: has_report and "15 dB" in report_md and "Extreme Breakdown" in report_md,
              "report must contain all SNR rows and regime tags")

    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))
