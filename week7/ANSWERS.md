# Week 7 ◆ — Questions

Answer these questions as you complete Lab 1, Lab 2, and review your reproduction results with Prof. Hung.

---

### Q1. Reproduction Fidelity
Record your final reproduced numbers vs the GTCRN paper (Table 1):

| Metric | Paper (Table 1) | Your Reproduction | Relative Gap (%) | Status (<= 5%) |
|---|---|---|---|---|
| PESQ (wb) | 2.87 | | | |
| STOI | 0.940 | | | |
| SI-SDR (dB) | 18.83 | | | |

Did your model pass the $\le 5\%$ scientific reproduction criterion?

> 

---

### Q2. Dataset Fingerprint Integrity
Why was it critical to verify the SHA256 fingerprints of `test/clean` (`f161abab16d8391e`) and `test/noisy` (`205263899f4f980b`) before running final evaluation? What happened when torchaudio's default resampler was used instead of `soxr_hq` in Week 3?

> 

---

### Q3. STFT Window Impact on Reproduction
In Week 4 and Week 5, you investigated the STFT window.
If you had trained with SEtrain's default plain Hann window instead of the square-root Hann window, what gap in PESQ would have been introduced according to the official checkpoint calibration?

> 

---

### Q4. Checkpoint Selection Discipline
Which checkpoint did you evaluate on the test set (`best_model_*.tar` or `model_200.tar`)?
Did you evaluate multiple checkpoints on the test set, or did you evaluate only the one chosen on the validation set? Why does scientific rigor forbid "cherry-picking" checkpoints on the test set?

> 

---

### Q5. Comparing Reproduced vs Official Checkpoint
How does your reproduced checkpoint compare to the authors' official pre-trained checkpoint evaluated in Week 3 (PESQ 2.867, STOI 0.940, SI-SDR 18.80)? Are the two models functionally equivalent?

> 

---

### Q6. Sources of Experimental Variance
Even when using the exact same random seed (43) and dataset, slight differences can exist between GPU architectures (e.g. RTX 3060 vs RTX 5060 Ti). Mention two non-deterministic operations in deep learning libraries that cause small numerical variations.

> 

---

### Q7. Preparing for Next Week (Computational Efficiency)
Now that audio quality reproduction is verified, what other dimension must be evaluated before a speech enhancement model can be deployed to edge devices (e.g. smartphones or earbuds)?

> 
