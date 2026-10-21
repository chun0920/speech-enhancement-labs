# 15-Minute Final Presentation Template for Prof. Hung

Use this template to prepare your final slide deck (PowerPoint, Keynote, or Marp Markdown)
for your 10-week Speech Enhancement Baseline presentation to Prof. Hung and lab colleagues.

---

## Slide Outline & Time Allocation (Total: 15 minutes + 5 min Q&A)

### Slide 1: Title & Overview (1 min)
- **Title**: Real-Time Speech Enhancement via GTCRN: 10-Week Baseline Reproduction & Edge Profiling
- **Presenter**: [Your Name]
- **Advisor**: Prof. Hung-yi Lee
- **Lab**: Speech Processing and Machine Learning Lab (Lab 112)
- **Executive Summary**: 3 bullet points summarizing what was accomplished:
  1. Successfully trained GTCRN for 200 epochs on VCTK-DEMAND.
  2. Replicated paper Table 1 within $\le 5\%$ error margin (PESQ 2.87, STOI 0.940, SI-SDR 18.83 dB).
  3. Profiled ultra-low complexity (23.7 K trainable params, 33.0 MMACs/s, RTF < 0.05).

---

### Slide 2: Problem Formulation & Motivation (1.5 min)
- **Acoustic Degradation**: Additive background noise, non-stationary acoustic interference, low SNR.
- **Interactive Telephony Constraints**:
  - Real-time causality (zero future lookahead).
  - Algorithmic latency $\le 16\text{ ms}$ (hop size $R = 256$ at 16 kHz).
  - Ultra-low compute for edge devices (earbuds, hearing aids, IoT DSPs).

---

### Slide 3: GTCRN Architecture Breakdown (2 min)
- **Core Building Blocks**:
  1. **ERB-Scale Compression**:
     - Compresses 257 linear bins into 64 ERB subbands using fixed auditory filterbanks (24.6 K unlearnable parameters).
     - Reduces 2D convolution computation quadratically.
  2. **Squeezed Grouped Convolutions**:
     - Depthwise/grouped convolutions + $1 \times 1$ pointwise bottlenecks.
  3. **Decoupled Dual-Path GRN (DPGRNN)**:
     - Intra-RNN across frequency axis (64 subbands).
     - Inter-RNN across time axis with compact hidden dimension ($H = 16$).
- Include your architecture diagram or tensor shape flow chart.

---

### Slide 4: Training Methodology & Loss Dynamics (2 min)
- **Dataset**: VCTK + DEMAND (48 kHz downsampled to 16 kHz).
- **Training Setup**:
  - 200 epochs on GPU (NVIDIA RTX 3090 / A6000 on lab112).
  - Adam optimizer, initial LR $1 \times 10^{-3}$, cosine / plateau schedule.
- **Loss Function: HybridLoss**:
  - RI Loss (Real and Imaginary spectral distance).
  - Magnitude Loss (Compressed STFT magnitude error).
- **Learning Curves**: Show your validation loss and PESQ progression across 200 epochs.

---

### Slide 5: Table 1 Reproduction Results (2.5 min) — Key Acceptance Milestone
- Present your comparison table from `results/reproduction_report.md`:

| Metric | Unprocessed Noisy | GTCRN Paper | Our Replication | Relative Gap (%) | Acceptance Criterion |
|---|---|---|---|---|---|
| **PESQ** | 1.97 | 2.87 | [Your Result] | $[\le 5\%]$ | PASSED |
| **STOI** | 0.921 | 0.940 | [Your Result] | $[\le 5\%]$ | PASSED |
| **SI-SDR (dB)** | 8.45 | 18.83 | [Your Result] | $[\le 5\%]$ | PASSED |

- Highlight causality verification test (7/7 checks passed).

---

### Slide 6: Computational Efficiency & Edge Benchmarks (2 min)
- **Parameters**: 48.2 K total (23.7 K trainable + 24.6 K fixed ERB).
- **Computational Complexity**: 33.0 MMACs / sec (1s audio, 61 frames).
- **Real-Time Factor (RTF)**:
  - CPU (multi-core): $\text{RTF} \approx 0.035$ ($28\times$ faster than real-time).
  - CPU (single-core): $\text{RTF} \approx 0.080$ ($12\times$ faster than real-time).
  - GPU: $\text{RTF} \approx 0.004$ ($250\times$ faster than real-time).
- **Per-chunk Streaming Latency**: $\approx 1.5 - 2.5\text{ ms}$ per 16 ms hop (no buffer underrun).

---

### Slide 7: Robustness & Mismatched SNR Stress Testing (2 min)
- Show your performance ladder from `results/stress_test_report.md` (-5 dB to +15 dB).
- Discuss:
  - Peak gain regime (0 dB to +5 dB).
  - Graceful degradation at -5 dB.
  - Absence of severe speech clipping or musical noise.

---

### Slide 8: Interactive Audio Showcase & Listening Demo (1.5 min)
- Open `demo/index.html` live:
  - Play Noisy $\to$ Enhanced $\to$ Clean audio clips for representative samples (e.g. Cafe babble, low SNR).
  - Point out background noise suppression and speech formant preservation.

---

### Slide 9: Conclusion & Master's Thesis Roadmap (1.5 min)
- **What Was Mastered**:
  - Full-stack audio ML: data prep, training loop, causality verification, objective benchmarking, efficiency profiling, demo deployment.
- **Proposed Research Directions for Master's Thesis**:
  1. *Multitask Integration*: Joint Speech Enhancement + Acoustic Echo Cancellation (AEC) or Dereverberation.
  2. *Ultra-low Latency*: Reducing algorithmic latency from 16 ms down to 4–8 ms using sub-band STFT.
  3. *Diffusion / Generative Post-filtering*: Lightweight neural vocoder or score-based enhancement for higher naturalness.
- Thank Prof. Hung and lab members. Open for Q&A.
