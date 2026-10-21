"""
Week 5 - Lab 1: Dissecting HybridLoss                              run:  python lab1_loss.py

In Week 4, you saw the three loss components in SEtrain's loss_factory.py. Training GTCRN to
reproduce the paper's 2.87 PESQ requires mastering what this loss optimizes and why.

HybridLoss combines three distinct objectives:
  1. Compressed Complex Spectrogram Loss (Real + Imaginary MSE, weight lamda_ri = 30)
  2. Compressed Magnitude Spectrogram Loss (Magnitude MSE, weight lamda_mag = 70)
  3. Time-Domain Scale-Invariant Signal-to-Noise Ratio (SI-SNR) Loss (weight 1)

In this lab, you implement each mathematical piece from scratch, verify scale invariance,
observe how power-law compression (c = 0.3) preserves high-frequency speech formants,
and verify gradient flow for backpropagation.
"""
import sys
import torch
import torch.nn as nn

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
def power_compress(stft, c=0.3, eps=1e-12):
    """Apply power-law compression to a complex STFT tensor:
         X_c = X / (|X| + eps)**(1 - c)
       This scales the magnitude to |X|**c while preserving the exact phase angle:
         |X_c| = |X|**c,   angle(X_c) = angle(X)

       Args:
         stft: torch.complex64 tensor of shape (..., F, T)
         c: compression exponent (default 0.3)
         eps: small clamp to prevent division by zero

       Returns:
         stft_c: compressed complex tensor
         mag: uncompressed magnitude, clamped by eps
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.2
def spectral_loss(pred_stft_c, true_stft_c, pred_mag, true_mag, c=0.3, lamda_ri=30.0, lamda_mag=70.0):
    """Compute the weighted frequency-domain losses between compressed spectra:
         real_loss = mean((pred_c.real - true_c.real)**2)
         imag_loss = mean((pred_c.imag - true_c.imag)**2)
         ri_loss   = real_loss + imag_loss
         mag_loss  = mean((pred_mag**c - true_mag**c)**2)

       Returns:
         total_spec_loss: lamda_ri * ri_loss + lamda_mag * mag_loss
         ri_loss: unweighted real+imag loss scalar tensor
         mag_loss: unweighted magnitude loss scalar tensor
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.3
def sisnr_loss(pred, true, eps=1e-12):
    """Compute time-domain Scale-Invariant SNR (SI-SNR) loss:
         1. Project pred onto true:
              s_target = (<true, pred> / (||true||**2 + 1e-8)) * true
         2. Noise residual:
              e_noise  = pred - s_target
         3. Scale-invariant loss formula (SEtrain convention):
              sisnr = - 2 * log10( (||s_target|| / (||e_noise|| + eps)) + eps )
            A 10 dB improvement in SI-SNR reduces this loss term by 1.0.

       Args:
         pred, true: waveform tensors of shape (B, T) or (T,)

       Returns:
         sisnr_scalar: mean SI-SNR loss across the batch
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.4
def hybrid_loss(y_pred, y_true, n_fft=512, hop_len=256, win_len=512, window=None,
                c=0.3, lamda_ri=30.0, lamda_mag=70.0, eps=1e-12):
    """Complete forward pass of HybridLoss:
         1. STFT of y_pred and y_true
         2. Power compression with power_compress()
         3. Spectral losses with spectral_loss()
         4. Time-domain SI-SNR loss with sisnr_loss()
         5. Total = total_spec_loss + sisnr_loss

       Returns:
         total_loss, ri_loss, mag_loss, sisnr
    """
    # TODO
    raise NotImplementedError


# ================================================================ Test Suite
if __name__ == "__main__":
    print("--- Testing Exercise 1.1: power_compress ---")
    torch.manual_seed(42)
    dummy_stft = torch.randn(2, 257, 100, dtype=torch.complex64)
    try:
        c_stft, c_mag = power_compress(dummy_stft, c=0.3)
    except NotImplementedError:
        c_stft, c_mag = None, None

    check("preserves complex phase angle",
          lambda: torch.allclose(torch.angle(c_stft), torch.angle(dummy_stft), atol=1e-5),
          "phase angle must remain unchanged after compression")
    check("compresses magnitude to |X|**0.3",
          lambda: torch.allclose(torch.abs(c_stft), torch.abs(dummy_stft).clamp(min=1e-12)**0.3, atol=1e-5),
          "magnitude of compressed tensor should equal |X|**c")

    print("\n--- Testing Exercise 1.2: spectral_loss ---")
    try:
        spec_loss, ri_val, mag_val = spectral_loss(c_stft, c_stft, c_mag, c_mag)
    except NotImplementedError:
        spec_loss, ri_val, mag_val = None, None, None

    check("loss on identical spectra is 0.0",
          lambda: torch.isclose(spec_loss, torch.tensor(0.0), atol=1e-7),
          "identical predictions and targets must yield zero spectral loss")

    dummy_pred_stft = dummy_stft * 1.5
    try:
        pred_c, pred_mag = power_compress(dummy_pred_stft, c=0.3)
        spec_diff, ri_diff, mag_diff = spectral_loss(pred_c, c_stft, pred_mag, c_mag, lamda_ri=30.0, lamda_mag=70.0)
    except NotImplementedError:
        spec_diff, ri_diff, mag_diff = None, None, None

    check("combines weighted RI and magnitude losses correctly",
          lambda: torch.isclose(spec_diff, 30.0 * ri_diff + 70.0 * mag_diff, atol=1e-5),
          "total_spec_loss must equal lamda_ri * ri_loss + lamda_mag * mag_loss")

    print("\n--- Testing Exercise 1.3: sisnr_loss ---")
    clean_wav = torch.randn(2, 16000)
    noisy_wav = clean_wav + 0.5 * torch.randn(2, 16000)
    try:
        sisnr_same = sisnr_loss(clean_wav, clean_wav)
        sisnr_noisy = sisnr_loss(noisy_wav, clean_wav)
        sisnr_noisy_scaled = sisnr_loss(noisy_wav * 3.7, clean_wav)
    except NotImplementedError:
        sisnr_same, sisnr_noisy, sisnr_noisy_scaled = None, None, None

    check("perfect match has low negative SI-SNR loss (< -15.0)",
          lambda: sisnr_same is not None and float(sisnr_same) < -15.0,
          "e_noise is zero for identical signals, so loss is strongly negative")

    check("SI-SNR loss is strictly scale-invariant: loss(3.7*noisy, clean) == loss(noisy, clean)",
          lambda: sisnr_noisy is not None and torch.isclose(sisnr_noisy, sisnr_noisy_scaled, atol=1e-4),
          "scaling the prediction by any constant should not change the SI-SNR loss")

    check("noisy signal has higher (worse) SI-SNR loss than clean signal",
          lambda: sisnr_noisy is not None and float(sisnr_noisy) > float(sisnr_same),
          "adding noise must degrade SI-SNR (increase the loss)")

    print("\n--- Testing Exercise 1.4: hybrid_loss & Backpropagation ---")
    pred_var = noisy_wav.clone().detach().requires_grad_(True)
    try:
        tot_loss, ri_l, mag_l, si_l = hybrid_loss(pred_var, clean_wav)
        tot_loss.backward()
        has_grad = pred_var.grad is not None and not torch.isnan(pred_var.grad).any()
    except NotImplementedError:
        tot_loss, has_grad = None, False

    check("hybrid_loss produces finite scalar loss",
          lambda: tot_loss is not None and torch.isfinite(tot_loss),
          "total loss must be a valid, finite scalar")
    check("gradients propagate back cleanly without NaN",
          lambda: has_grad,
          "check gradient calculation; clamp eps is needed to prevent division by zero")

    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))
