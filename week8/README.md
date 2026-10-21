# Week 8 — Computational Efficiency: Parameters, MACs & Real-Time Factor (RTF)

In Week 7, you successfully reproduced GTCRN's speech enhancement quality (PESQ 2.87).
However, for real-time applications such as hearing aids, edge IoT, and video conferencing,
audio quality is only half the requirement: **the algorithm must be computationally lightweight
and run faster than real time.**

In this week, you profile the model's computational footprint on identical hardware:
1. **Parameter count**: Trainable weights (23.7 K) vs unlearnable ERB weights (24.6 K), total 48.2 K
2. **Computational complexity**: Multiply-Accumulate operations per second (33.0 MMACs / sec)
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
python check_env.py
```

---

## Step 1 — Profile Parameters and MACs (Lab 1, ≈ 30 min)

```bash
python lab1_profile.py
```

Complete the exercises:
1. `count_parameters()`: Separates trainable from fixed weights.
2. `profile_macs()`: Calculates layer-by-layer MACs for 1 second of audio (61 frames).
3. `format_efficiency_summary()`: Generates the efficiency table.

All 5 checks must pass with **All done!**.

---

## Step 2 — Benchmark Real-Time Factor (RTF) and Chunk Latency (Lab 2, ≈ 45 min)

```bash
python lab2_rtf.py
```

Complete the exercises:
1. `benchmark_execution_time()`: Precise timing harness with warmup and CUDA synchronization.
2. `measure_rtf()`: Real-Time Factor calculation for full utterance inference.
3. `measure_chunk_latency_ms()`: Millisecond latency per 16 ms hop.
4. `generate_efficiency_table()`: Exports `results/efficiency_table.md`.

All 4 checks must pass with **All done!**.

---

## Step 3 — Questions (`ANSWERS.md`)

Answer questions Q1–Q7 in `ANSWERS.md`.

---

## Hand-in

Your folder `week8_<yourname>/` containing:
- Both lab scripts completed (`lab1_profile.py` 5/5, `lab2_rtf.py` 4/4)
- `results/efficiency_table.md` containing your machine's benchmark
- `ANSWERS.md` filled in
