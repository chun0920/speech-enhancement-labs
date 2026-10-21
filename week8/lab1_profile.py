"""
Week 8 - Lab 1: Parameter & MACs Profiling                          run:  python lab1_profile.py

A speech enhancement model cannot be deployed to earbuds, phones, or hearing aids without
meeting strict compute and memory budgets.

GTCRN was specifically designed for ultra-low complexity:
  - Paper reported: 23.7 K trainable parameters, 39.6 MMACs / sec
  - Official repo update: 48.2 K total parameters (including 24.6 K fixed ERB weights),
    and 33.0 MMACs / sec (replacing low-frequency linear-to-ERB matrix multiplication with concatenation)

In this lab, you:
  1. Profile trainable vs non-trainable parameter counts (48,245 total, 23,669 trainable)
  2. Implement a layer-level computational complexity (MACs) profiler
  3. Verify that 1 second of audio costs ~33.0 MMACs
"""
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
def count_parameters(model):
    """Count parameters of a PyTorch module.

    Returns:
      (total_params: int, trainable_params: int, fixed_params: int)
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.2
def profile_macs(model, frames=61, freq_bins=257, device="cpu"):
    """Profile Multiply-Accumulate operations (MACs) for GTCRN on a spectrogram input.

    For 1 second of 16 kHz audio with hop=256, there are approximately 61–63 frames.
    At 61 frames, GTCRN requires ~33.0 MMACs.

    Args:
      model: GTCRN module
      frames: number of STFT time frames (default 61)
      freq_bins: number of frequency bins (default 257)

    Returns:
      (total_macs: int, mmacs: float)
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.3
def format_efficiency_summary(total_p, train_p, fixed_p, mmacs):
    """Format parameter count and MACs into a clean Markdown table string."""
    # TODO
    raise NotImplementedError


# ================================================================ Test Suite
if __name__ == "__main__":
    import tempfile

    print("--- Testing Exercise 1.1: count_parameters ---")
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(10, 20)  # 10*20 + 20 = 220
            self.l2 = nn.Linear(20, 5, bias=False)  # 20*5 = 100
            self.l2.weight.requires_grad = False

    dummy = DummyModel()
    try:
        tot, trn, fix = count_parameters(dummy)
    except NotImplementedError:
        tot, trn, fix = 0, 0, 0

    check("total parameters counted correctly (320)",
          lambda: tot == 320,
          "expected 220 + 100 = 320 total")
    check("trainable parameters counted correctly (220)",
          lambda: trn == 220,
          "expected 220 trainable")
    check("fixed parameters counted correctly (100)",
          lambda: fix == 100,
          "expected 100 fixed")

    print("\n--- Testing Exercise 1.2: profile_macs on Toy Model ---")
    class SimpleConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 4, kernel_size=(3, 3), padding=0, bias=False)

        def forward(self, x):
            # x: (1, 257, 61, 2) -> permute to (1, 2, 61, 257)
            x = x.permute(0, 3, 2, 1)
            return self.conv(x)

    toy = SimpleConvModel()
    try:
        t_macs, t_mmacs = profile_macs(toy, frames=61, freq_bins=257)
    except NotImplementedError:
        t_macs, t_mmacs = 0, 0.0

    # Output shape: H_out = 61 - 2 = 59, W_out = 257 - 2 = 255
    # MACs = 1 * 59 * 255 * 4 * 2 * 3 * 3 = 15045 * 72 = 1,083,240
    check("calculates Conv2d MACs accurately (> 1.0 MMAC)",
          lambda: 1.0 < t_mmacs < 1.2,
          "expected ~1.08 MMACs for toy convolution")

    print("\n--- Testing Exercise 1.3: format_efficiency_summary ---")
    try:
        summary_md = format_efficiency_summary(48245, 23669, 24576, 33.0)
    except NotImplementedError:
        summary_md = ""

    check("formats markdown table with 33.0 MMACs and 23.67 K",
          lambda: "33.0 MMACs" in summary_md and "23.67 K" in summary_md,
          "table must display 33.0 MMACs and 23.67 K parameters")

    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))
