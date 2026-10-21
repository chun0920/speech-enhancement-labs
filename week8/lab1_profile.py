"""
Week 8 - Lab 1: Parameter & MACs Profiling                          run:  python lab1_profile.py

A speech enhancement model cannot be deployed to earbuds, phones, or hearing aids without
meeting strict compute and memory budgets.

GTCRN was specifically designed for ultra-low complexity:
  - Paper reported: 23.7 K trainable parameters, 39.6 MMACs / sec
  - Official repo update: 48.2 K total parameters (including 24.6 K fixed ERB weights),
    and 33.0 MMACs / sec for 61 unpadded frames; 34.08 MMACs for 63 frames (my_model.example_input(1.0))

In this lab, you:
  1. Profile trainable vs non-trainable parameter counts (48,245 total, 23,669 trainable)
  2. Implement computational complexity (MACs) profiling using thop
  3. Verify that 1 second of audio (63 frames) costs ~34.08 MMACs on the real GTCRN model
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
def profile_macs(model, input_tensor=None, device="cpu"):
    """Profile Multiply-Accumulate operations (MACs) for GTCRN on a spectrogram input using thop.

    For 1 second of 16 kHz audio with hop=256, my_model.example_input(1.0) produces
    an STFT tensor of shape (1, 257, 63, 2).
    At 63 frames, GTCRN requires ~34.08 MMACs (measured with thop).
    (Note: 61 frames without STFT center-padding corresponds to 33.0 MMACs).

    Args:
      model: PyTorch module
      input_tensor: optional input tensor; if None, creates (1, 257, 63, 2)
      device: device to run profiling on ("cpu")

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

    check("total parameters counted correctly on toy (320)",
          lambda: tot == 320,
          "expected 220 + 100 = 320 total")
    check("trainable parameters counted correctly on toy (220)",
          lambda: trn == 220,
          "expected 220 trainable")
    check("fixed parameters counted correctly on toy (100)",
          lambda: fix == 100,
          "expected 100 fixed")

    print("\n--- Testing with Real GTCRN Model ---")
    try:
        import my_model
        gtcrn = my_model.build_model("cpu")
        inp_1s = my_model.example_input(1.0, "cpu")[0]
    except Exception as e:
        gtcrn = None
        inp_1s = None
        print(f"  Note: my_model not loaded directly: {e}")

    check("my_model.py available and GTCRN loads",
          lambda: gtcrn is not None,
          "copy my_model.py into this folder (README Step 0)")

    if gtcrn is not None and inp_1s is not None:
        try:
            g_tot, g_trn, g_fix = count_parameters(gtcrn)
        except NotImplementedError:
            g_tot, g_trn, g_fix = 0, 0, 0

        check("real GTCRN total parameters counted correctly (48,245)",
              lambda: g_tot == 48245,
              f"expected 48245 total, got {g_tot}")
        check("real GTCRN trainable parameters counted correctly (23,669)",
              lambda: g_trn == 23669,
              f"expected 23669 trainable, got {g_trn}")
        check("real GTCRN fixed ERB parameters counted correctly (24,576)",
              lambda: g_fix == 24576,
              f"expected 24576 fixed ERB, got {g_fix}")

        print("\n--- Testing Exercise 1.2: profile_macs on Real GTCRN (63 frames) ---")
        try:
            t_macs, t_mmacs = profile_macs(gtcrn, input_tensor=inp_1s)
        except NotImplementedError:
            t_macs, t_mmacs = 0, 0.0
        except Exception as e:
            t_macs, t_mmacs = 0, 0.0
            print(f"  Note: profile_macs raised {type(e).__name__}: {e}")

        check("profiles real GTCRN MACs on example_input(1.0) (~34.08 MMACs via thop)",
              lambda: 33.5 <= t_mmacs <= 34.5,
              f"expected ~34.08 MMACs, got {t_mmacs:.2f} MMACs (ensure thop is installed: pip install thop==0.1.1.post2209072238)")

        print("\n--- Testing Exercise 1.3: format_efficiency_summary ---")
        try:
            summary_md = format_efficiency_summary(g_tot, g_trn, g_fix, t_mmacs)
        except NotImplementedError:
            summary_md = ""
        except Exception as e:
            summary_md = ""
            print(f"  Note: format_efficiency_summary raised {type(e).__name__}: {e}")

        check("formats markdown table with measured GTCRN values",
              lambda: f"{t_mmacs:.2f} MMACs" in summary_md and "23.67 K" in summary_md and "48.25 K" in summary_md,
              "table must display measured MMACs and parameter counts")
    else:
        check("real GTCRN total parameters counted correctly (48,245)", lambda: False, "GTCRN model not loaded")
        check("real GTCRN trainable parameters counted correctly (23,669)", lambda: False, "GTCRN model not loaded")
        check("real GTCRN fixed ERB parameters counted correctly (24,576)", lambda: False, "GTCRN model not loaded")
        check("profiles real GTCRN MACs on example_input(1.0) (~34.08 MMACs via thop)", lambda: False, "GTCRN model not loaded")
        check("formats markdown table with measured GTCRN values", lambda: False, "GTCRN model not loaded")

    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))
