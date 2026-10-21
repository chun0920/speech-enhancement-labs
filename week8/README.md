# Week 8 — Computational Efficiency: Parameters, MACs & Real-Time Factor (RTF)

In Week 7, you successfully reproduced GTCRN's speech enhancement quality (PESQ 2.87).
However, for real-time applications such as hearing aids, edge IoT, and video conferencing,
audio quality is only half the requirement: **the algorithm must be computationally lightweight
and run faster than real time.**

In this week, you profile the model's computational footprint on identical hardware:
1. **Parameter count**: Trainable weights (23.7 K) vs unlearnable ERB weights (24.6 K), total 48.2 K
2. **Computational complexity**: Multiply-Accumulate operations per second (34.08 MMACs / sec for 1.0s audio / 63 frames via thop; 33.0 MMACs / sec for 61 unpadded frames)
3. **Real-Time Factor (RTF)** on CPU and GPU ($\text{RTF} = \text{Processing Time} / \text{Audio Duration}$)
4. **Per-chunk latency**: Execution time for a 16 ms streaming hop ($256 / 16000\text{ s}$)

---

## What's in the folder

| File | Purpose |
|---|---|
| `README.md` | this guide |
| `check_env.py` | checks this week's setup |
| `lab1_profile.py` | **your work**: parameter breakdown and MACs profiler |
| `lab2_rtf.py` | **your work**: execution timing harness, RTF measurement, chunk latency |
| `ANSWERS.md` | efficiency questions |

Prerequisites: `my_model.py` and `infer_dir.py` from earlier weeks.

---

## Step 0 — Setup

```bash
cp -r ~/speech-enhancement-labs/week8 ~/week8_<yourname> && cd ~/week8_<yourname>
cp ~/week7_<yourname>/my_model.py ~/week7_<yourname>/infer_dir.py .
conda activate se
pip install thop==0.1.1.post2209072238
python check_env.py
```

---

## Step 1 — Profile Parameters and MACs (Lab 1, ≈ 30 min)

```bash
python lab1_profile.py
```

Complete the exercises:
1. `count_parameters()`: Separates trainable from fixed weights.
2. `profile_macs()`: Calculates model MACs on `example_input(1.0)` (63 frames, ~34.08 MMACs via thop).
3. `format_efficiency_summary()`: Generates the efficiency table.

All checks must pass with **All done!**.

---

## Step 2 — Benchmark Real-Time Factor (RTF) and Chunk Latency (Lab 2, ≈ 45 min)

```bash
python lab2_rtf.py
```

Complete the exercises:
1. `benchmark_execution_time()`: Harness with warmup iterations and CUDA synchronization.
2. `measure_rtf()`: Computes RTF for full utterance inference on CPU (must be < 1.0).
3. `measure_chunk_latency_ms()`: Measures 16 ms chunk streaming latency (must be < 16.0 ms).
4. `generate_efficiency_table()`: Exports `results/efficiency_table.md`.

All checks must pass with **All done!**.

---

## Step 3 — Answer the Conceptual Questions

Answer questions Q1–Q7 in `ANSWERS.md` and review with your mentor.
 Master's students: incorporate `results/efficiency_table.md` into your final report.
