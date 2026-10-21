"""
Week 8 - Lab 2: Real-Time Factor (RTF) & Latency Benchmarking        run:  python lab2_rtf.py

Real-Time Factor (RTF) measures how fast an algorithm processes audio relative to real time:
  RTF = Processing Time (s) / Audio Duration (s)

For interactive communication (Zoom, Teams, Discord):
  - RTF MUST be < 1.0 (smaller is faster, e.g. RTF = 0.05 means 1s of audio is processed in 50 ms)
  - Per-chunk processing time must be well below the hop size (16 ms) to prevent buffer underruns

In this lab, you:
  1. Implement a precise benchmarking harness with CUDA synchronization and warmup iterations
  2. Measure RTF on CPU on the real GTCRN model
  3. Measure chunk-by-chunk streaming latency (16 ms hop size) on the real GTCRN model
  4. Generate results/efficiency_table.md for your group's final report
"""
import copy
import platform
import time
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


# ---------------------------------------------------------------- Exercise 2.1
def benchmark_execution_time(fn, warmup=3, trials=10, is_cuda=False):
    """Accurately measure execution time of a callable fn.

    Protocol:
      1. Run fn() `warmup` times to prime instruction caches and GPU kernels
      2. If is_cuda: torch.cuda.synchronize() before and after timing
      3. Run fn() `trials` times, measuring total wall-clock time
      4. Return mean elapsed time per call in seconds

    Returns:
      mean_seconds: float
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 2.2
def measure_rtf(model_fn, audio_duration_sec, is_cuda=False, warmup=2, trials=5):
    """Compute Real-Time Factor (RTF):
         RTF = mean_processing_time_sec / audio_duration_sec
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 2.3
def measure_chunk_latency_ms(chunk_fn, is_cuda=False, warmup=5, trials=20):
    """Measure the execution latency of processing a single 16 ms chunk in milliseconds."""
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 2.4
def generate_efficiency_table(results_dict, out_path):
    """Write a Markdown efficiency table to out_path and return its text.

    results_dict: {"hardware": str, "cpu_rtf": float, "gpu_rtf": float or None, "chunk_latency_ms": float,
                   "params_k": float, "mmacs": float}
    One row per number, with the real-time threshold next to it where there is one. Write the CPU RTF with
    4 decimals (e.g. 0.0250) and the time per frame with 2 decimals and " ms" (e.g. "7.70 ms").
    """
    # TODO
    raise NotImplementedError


# ================================================================ Test Suite
def cpu_name():
    """(Given) The CPU model name, for the table (Linux)."""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "CPU"


if __name__ == "__main__":
    import tempfile

    print("--- Testing Exercise 2.1: benchmark_execution_time on dummy task ---")
    def sleep_work():
        time.sleep(0.005)

    try:
        t_sec = benchmark_execution_time(sleep_work, warmup=2, trials=3, is_cuda=False)
    except NotImplementedError:
        t_sec = 0.0

    check("measures ~5 ms elapsed time accurately",
          lambda: 0.004 < t_sec < 0.010,
          "expected elapsed time close to 5 ms")

    print("\n--- Testing on Real GTCRN Model ---")
    try:
        import my_model
        gtcrn = my_model.build_model("cpu")
        inp_1s = my_model.example_input(1.0, "cpu")[0]
        inp_chunk = my_model.example_input(0.016, "cpu")[0]
    except Exception as e:
        gtcrn = None
        inp_1s = None
        inp_chunk = None
        print(f"  Note: my_model not loaded directly: {e}")

    check("my_model.py available and GTCRN loads",
          lambda: gtcrn is not None,
          "copy my_model.py into this folder (README Step 0)")

    if gtcrn is not None:
        def run_1s():
            with torch.no_grad():                    # inference only: no autograd bookkeeping in the timing
                gtcrn(inp_1s)

        print("\n--- Testing Exercise 2.2: measure_rtf on Real GTCRN ---")
        try:
            cpu_rtf = measure_rtf(run_1s, audio_duration_sec=1.0, is_cuda=False, warmup=2, trials=5)
        except NotImplementedError:
            cpu_rtf = 999.0

        check("real GTCRN CPU RTF < 1.0 (faster than real-time)",
              lambda: 0.001 < cpu_rtf < 0.5,
              f"expected RTF < 1.0, got {cpu_rtf:.4f}")

        def run_chunk():
            with torch.no_grad():
                gtcrn(inp_chunk)

        print("\n--- Testing Exercise 2.3: measure_chunk_latency_ms on Real GTCRN ---")
        try:
            chunk_lat = measure_chunk_latency_ms(run_chunk, is_cuda=False, warmup=5, trials=20)
        except NotImplementedError:
            chunk_lat = 999.0

        check("real GTCRN chunk latency < 16.0 ms (streaming budget)",
              lambda: 0.01 < chunk_lat < 16.0,
              f"expected chunk latency < 16.0 ms, got {chunk_lat:.2f} ms")

        print("\n--- Testing Exercise 2.4: generate_efficiency_table ---")
        try:
            import thop
            macs_1s, _ = thop.profile(copy.deepcopy(gtcrn), inputs=(inp_1s,), verbose=False)
        except ImportError:
            macs_1s = float("nan")
        out_f = Path("results") / "efficiency_table.md"
        measured = {
            "hardware": f"{cpu_name()}, {torch.get_num_threads()} threads",
            "cpu_rtf": cpu_rtf,
            "gpu_rtf": None,
            "chunk_latency_ms": chunk_lat,
            "params_k": sum(p.numel() for p in gtcrn.parameters() if p.requires_grad) / 1000,
            "mmacs": macs_1s / 1e6,
        }
        try:
            tbl = generate_efficiency_table(measured, out_f)
            has_tbl = out_f.exists() and f"{cpu_rtf:.4f}" in out_f.read_text(encoding="utf-8")
            print(f"  saved {out_f}")
        except NotImplementedError:
            tbl, has_tbl = "", False

        check("writes results/efficiency_table.md with the measured RTF and latency",
              lambda: has_tbl and f"{chunk_lat:.2f} ms" in tbl,
              "table must contain actual measured values")
    else:
        check("real GTCRN CPU RTF < 1.0 (faster than real-time)", lambda: False, "GTCRN model not loaded")
        check("real GTCRN chunk latency < 16.0 ms (streaming budget)", lambda: False, "GTCRN model not loaded")
        check("writes efficiency table with actual measured RTF and latency", lambda: False, "GTCRN model not loaded")

    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))
