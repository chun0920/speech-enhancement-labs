# Report 2 — GTCRN: The Role of Each Loss Component in HybridLoss (master's students)

Presented to Prof. Hung. About 10 minutes. Build it from your implementation in `lab1_loss.py`, the training
curves in `results/training_curves.csv`, and the GTCRN / SEtrain loss design.

---

## 1. Executive Summary

In one sentence, explain what problem HybridLoss solves that single-domain losses (such as pure MSE or pure time-domain loss) cannot.

> 

---

## 2. Multi-Objective Framework Overview

Write out the complete loss formulation and explain why speech enhancement benefits from combining time-domain and frequency-domain supervision:

$$\mathcal{L}_{total} = \lambda_{RI} \mathcal{L}_{RI} + \lambda_{Mag} \mathcal{L}_{Mag} + \mathcal{L}_{SISNR}$$

| Component | Domain | Mathematical Expression | Weight ($\lambda$) | Primary Acoustic Role |
|---|---|---|---|---|
| Compressed Complex Loss ($\mathcal{L}_{RI}$) | Frequency (Complex) | | 30 | |
| Compressed Magnitude Loss ($\mathcal{L}_{Mag}$) | Frequency (Magnitude) | | 70 | |
| SI-SNR Loss ($\mathcal{L}_{SISNR}$) | Time (Waveform) | | 1 | |

---

## 3. Power-Law Spectral Compression ($c = 0.3$)

Explain the psychoacoustic rationale for compressing spectrogram magnitudes ($X_c = X / |X|^{1-c}$ with $c = 0.3$):
1. How does the dynamic range of speech spectrograms behave across frequency (vowels vs consonants, low harmonics vs high-frequency fricatives)?
2. What would happen if $c = 1.0$ (linear magnitude) was used instead?
3. Why not use log magnitude ($\log |X|$)? (Hint: numerical stability near zero energy and perceptual compression curve shape).

> 

---

## 4. Complex Spectrogram Loss ($\mathcal{L}_{RI}$) vs Magnitude Loss ($\mathcal{L}_{Mag}$)

1. Why is Magnitude loss weighted higher ($\lambda_{Mag} = 70$) than Complex loss ($\lambda_{RI} = 30$)?
2. What role does phase information play in speech intelligibility vs perceptual quality (PESQ)?
3. Why can a model not achieve high PESQ by predicting magnitude alone?

> 

---

## 5. Time-Domain Scale-Invariant SNR ($\mathcal{L}_{SISNR}$)

1. What does "Scale-Invariant" mean mathematically? Write the projection formula of predicted signal $\hat{s}$ onto ground truth $s$.
2. Why is scale invariance crucial when training with deep neural networks? (Hint: prevents trivial arbitrary gain adjustments from dominating gradient updates).
3. How does this term supervise global waveform alignment across all frequency bands simultaneously?

> 

---

## 6. Experimental Observations from Training Curves

From your training run on VCTK-DEMAND (`results/training_curves.csv`):
1. How did training loss and validation loss evolve over the first 20 epochs (warmup phase) compared to subsequent epochs?
2. At what epoch did validation loss reach its minimum? Did validation PESQ peak at the exact same epoch?
3. Did you observe any signs of overfitting (generalization gap widening)?

> 

---

## 7. Discussion Questions for Prof. Hung

List 2–3 technical questions or discussion points regarding loss design in modern speech enhancement (e.g. perceptual metric losses like differentiable PESQ, metricGAN, or multi-resolution STFT loss).

> 
