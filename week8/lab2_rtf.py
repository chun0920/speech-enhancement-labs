"""
Week 8 - Lab 2: Real-Time Factor (RTF) & Latency Benchmarking        run:  python lab2_rtf.py

Real-Time Factor (RTF) measures how fast an algorithm processes audio relative to real time:
  RTF = Processing Time (s) / Audio Duration (s)

For interactive communication (Zoom, Teams, Discord):
  - RTF MUST be < 1.0 (smaller is faster, e.g. RTF = 0.05 means 1s of audio is processed in 50 ms)
  - Per-chunk processing time must be well below the hop size (16 ms) to prevent buffer underruns

In this lab, you:
  1. Implement a precise benchmarking harness with CUDA synchronization and warmup iterations
  2. Measure RTF on CPU (single-thread and multi-thread) and GPU
  3. Measure chunk-by-chunk streaming latency (16 ms hop size)
  4. Generate results/efficiency_table.md for your group's final report
"""
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
    """Generate a clean Markdown efficiency benchmark table.

    results_dict format:
      {
        "hardware": str,
        "cpu_rtf": float,
        "gpu_rtf": float (or None),
        "chunk_latency_ms": float,
        "params_k": float,
        "mmacs": float
      }
    """
    # TODO
    raise NotImplementedError


# ================================================================ Test Suite
if __name__ == "__main__":
    import tempfile

    print("--- Testing Exercise 2.1: benchmark_execution_time ---")
    def sleep_work():
        time.sleep(0.005)

    try:
        t_sec = benchmark_execution_time(sleep_work, warmup=2, trials=3, is_cuda=False)
    except NotImplementedError:
        t_sec = 0.0

    check("measures ~5 ms elapsed time accurately",
          lambda: 0.004 < t_sec < 0.010,
          "expected elapsed time close to 5 ms")

    print("\n--- Testing Exercise 2.2: measure_rtf ---")
    try:
        rtf_dummy = measure_rtf(sleep_work, audio_duration_sec=1.0, is_cuda=False, warmup=1, trials=2)
    except NotImplementedError:
        rtf_dummy = 999.0

    check("computes RTF < 1.0 for 5 ms task on 1s audio",
          lambda: 0.003 < rtf_dummy < 0.015,
          "expected RTF ~0.005")

    print("\n--- Testing Exercise 2.3: measure_chunk_latency_ms ---")
    try:
        lat_ms = measure_chunk_latency_ms(sleep_work, is_cuda=False, warmup=1, trials=2)
    except NotImplementedError:
        lat_ms = 0.0

    check("returns latency in milliseconds (~5.0 ms)",
          lambda: 4.0 < lat_ms < 10.0,
          "expected latency in ms")

    print("\n--- Testing Exercise 2.4: generate_efficiency_table ---")
    with tempfile.TemporaryDirectory() as tmp:
        out_f = Path(tmp) / "results" / "efficiency_table.md"
        sample_dict = {
            "hardware": "RTX 3060 / Intel i7",
            "cpu_rtf": 0.035,
            "gpu_rtf": 0.004,
            "chunk_latency_ms": 1.25,
            "params_k": 23.67,
            "mmacs": 33.0,
        }
        try:
            tbl = generate_efficiency_table(sample_dict, out_f)
            has_tbl = out_f.exists() and "0.035" in out_f.read_text(encoding="utf-8")
        except NotImplementedError:
            tbl, has_tbl = "", False

        check("writes efficiency table with RTF and MMACs",
              lambda: has_tbl and "33.0 MMACs" in tbl,
              "table must contain formatted RTF and MMACs")

    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))
