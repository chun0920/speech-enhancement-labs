"""
Week 6 - Lab 2: Causal Architecture & Latency Verification          run:  python lab2_causality.py

Real-time speech communication requires strict causality: the enhanced output at the current
millisecond must never depend on future audio.

GTCRN is designed as a real-time causal network with 0 future frame lookahead.
In this lab, you:
  1. Inspect convolution padding rules to differentiate causal vs non-causal filters
  2. Implement an empirical perturbation test to verify that future changes never leak into the past
  3. Calculate the algorithmic latency of the causal STFT/iSTFT pipeline
  4. Test GTCRN directly on real audio tensors to prove it satisfies strict temporal causality
  5. Test against a deliberately non-causal symmetric layer to prove the test catches violations
"""
import torch
import torch.nn as nn
from pathlib import Path

import my_model

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


# ---------------------------------------------------------------- Exercise 2.1
def is_padding_causal(padding, kernel_size):
    """Check whether a 1D or 2D convolution's temporal padding is causal.

    In PyTorch F.pad or custom Conv2d modules with explicit time padding:
      - Symmetric padding (non-causal): pads equally on left and right:
          pad_left == pad_right == (kernel_size - 1) // 2
      - Causal padding: pads only on the past (left):
          pad_left == (kernel_size - 1) and pad_right == 0

    Args:
      padding: tuple (pad_left, pad_right) for time axis
      kernel_size: int, size of the convolution kernel in time

    Returns:
      bool: True if strictly causal (past-only padding)
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 2.2
def calculate_algorithmic_latency(hop_len=256, sample_rate=16000):
    """Compute the algorithmic latency (in ms and samples) for frame-by-frame streaming.

    In a causal overlap-add STFT framework with no future lookahead, the minimum algorithmic
    buffering delay equals exactly one hop size:
      latency_samples = hop_len
      latency_ms = (hop_len / sample_rate) * 1000.0

    Returns:
      (latency_samples: int, latency_ms: float)
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 2.3
def verify_temporal_causality(model_fn, x, cutoff_step, time_dim=None, tol=1e-5):
    """Verify temporal causality via future-perturbation testing.

    Protocol:
      1. Run model_fn on original input: y1 = model_fn(x)
      2. Construct perturbed input x_pert:
           samples at t >= cutoff_step on time_dim receive arbitrary random perturbation
      3. Run model_fn on perturbed input: y2 = model_fn(x_pert)
      4. Measure maximum absolute difference on past output (t < cutoff_step):
           past_diff = max(|y1[t < cutoff_step] - y2[t < cutoff_step]|)
      5. If past_diff <= tol: model is causal (future change did not alter past output)
         If past_diff > tol: model is non-causal (future leaked into the past)

    Returns:
      (is_causal: bool, past_diff: float)
    """
    # TODO
    raise NotImplementedError


# ================================================================ Test Suite
class NonCausalToyBlock(nn.Module):
    """Toy block with bidirectional context (deliberately non-causal for verification)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)  # symmetric padding

    def forward(self, x):
        return self.conv(x.unsqueeze(1)).squeeze(1)


if __name__ == "__main__":
    print("--- Testing Exercise 2.1: is_padding_causal ---")
    try:
        c1 = is_padding_causal((4, 0), kernel_size=5)
        c2 = is_padding_causal((2, 2), kernel_size=5)
        c3 = is_padding_causal((3, 0), kernel_size=3)
    except NotImplementedError:
        c1, c2, c3 = None, None, None

    check("identifies (4, 0) for kernel 5 as causal",
          lambda: c1 is True,
          "(k-1, 0) is past-only padding")
    check("rejects symmetric padding (2, 2) as non-causal",
          lambda: c2 is False,
          "symmetric padding looks into future samples")
    check("rejects incorrect left padding (3, 0) for kernel 3",
          lambda: c3 is False,
          "expected pad_left == 2 for kernel 3")

    print("\n--- Testing Exercise 2.2: calculate_algorithmic_latency ---")
    try:
        lat_samp, lat_ms = calculate_algorithmic_latency(hop_len=256, sample_rate=16000)
    except NotImplementedError:
        lat_samp, lat_ms = 0, 0.0

    check("latency is 256 samples",
          lambda: lat_samp == 256,
          "one hop is 256 samples")
    check("latency is exactly 16.0 ms at 16 kHz",
          lambda: abs(lat_ms - 16.0) < 1e-5,
          "256 / 16000 * 1000 = 16.0 ms")

    print("\n--- Testing Exercise 2.3: verify_temporal_causality ---")
    torch.manual_seed(42)
    non_causal_toy = NonCausalToyBlock()
    toy_x = torch.randn(1, 16000)
    try:
        non_causal_res, non_causal_diff = verify_temporal_causality(non_causal_toy, toy_x, cutoff_step=8000)
    except NotImplementedError:
        non_causal_res, non_causal_diff = True, 0.0

    check("detects future leakage in non-causal symmetric layer",
          lambda: non_causal_res is False and non_causal_diff > 1e-3,
          "symmetric convolution leaks future perturbation backward into past")

    print(f"\n--- Testing Causality on real {getattr(my_model, 'MODEL_NAME', 'model')} ---")
    try:
        model = my_model.build_model("cpu")
        inputs = my_model.example_input(2.0, "cpu")[0]  # (1, 257, 126, 2)
        cutoff = inputs.shape[2] // 2
        gtcrn_causal_res, gtcrn_diff = verify_temporal_causality(model, inputs, cutoff_step=cutoff)
    except NotImplementedError:
        gtcrn_causal_res, gtcrn_diff = False, 999.0
    except Exception as e:
        gtcrn_causal_res, gtcrn_diff = False, 999.0

    print(f"  {getattr(my_model, 'MODEL_NAME', 'model')} past max diff: {gtcrn_diff}")
    check(f"confirms real {getattr(my_model, 'MODEL_NAME', 'model')} produces 0 difference on past frames",
          lambda: gtcrn_causal_res is True and gtcrn_diff < 1e-6,
          f"{getattr(my_model, 'MODEL_NAME', 'model')} output must not change in past frames")

    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))
