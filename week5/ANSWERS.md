# Week 5 — Questions

Work through these questions as you complete Lab 1, Lab 2, and monitor your 200-epoch training run.

---

### Q1. Time vs Frequency Domain Loss
Why does SEtrain combine both frequency-domain loss (compressed STFT MSE) and time-domain loss (SI-SNR) instead of using only one domain? What blind spots does each domain have when used in isolation?

> 

---

### Q2. Power-Law Compression ($c = 0.3$)
In `lab1_loss.py`, you implemented $X_c = X / |X|^{1-c}$ with $c = 0.3$.
1. Why is linear magnitude ($c = 1.0$) inadequate for speech spectrograms?
2. How does $c = 0.3$ alter the relative gradient contribution between loud low-frequency vowel formants (e.g. 200–800 Hz) and quiet high-frequency consonants (e.g. 4–8 kHz fricatives like /s/, /f/)?

> 

---

### Q3. Spectral Loss Weights ($\lambda_{RI} = 30, \lambda_{Mag} = 70$)
Why is the magnitude loss weighted more than double the complex (real + imaginary) loss? What happens if you train with $\lambda_{Mag} = 0$ (complex loss only)?

> 

---

### Q4. SI-SNR Loss Formulation
In `sisnr_loss`, the loss term is:
$$\mathcal{L}_{SISNR} = -2 \log_{10}\left( \frac{\|s_{target}\|}{\|e_{noise}\| + \epsilon} + \epsilon \right)$$
Standard SI-SNR in decibels is:
$$\text{SI-SNR (dB)} = 10 \log_{10}\left( \frac{\|s_{target}\|^2}{\|e_{noise}\|^2} \right) = 20 \log_{10}\left( \frac{\|s_{target}\|}{\|e_{noise}\|} \right)$$
How does the loss value numerically relate to the standard SI-SNR in dB? If a model improves SI-SNR by +10 dB, by how much does $\mathcal{L}_{SISNR}$ change?

> 

---

### Q5. Learning Rate Warmup
SEtrain uses a 25,000-step linear warmup (the first 20 epochs on VCTK-DEMAND with batch size 8).
1. Why is warmup necessary when training deep recurrent/convolutional networks from scratch with Adam?
2. What risk occurs if you initialize the optimizer immediately at `lr = 1e-3` on step 1?

> 

---

### Q6. Cosine Annealing Schedule
From step 25,000 to step 250,000 (epochs 20–200), the learning rate decays via cosine annealing from $10^{-3}$ down to $10^{-6}$.
Why is a continuous decay superior to keeping the learning rate constant at $10^{-3}$ throughout the entire 200 epochs?

> 

---

### Q7. Training Dynamics and Generalization Gap
Looking at your `results/training_curves.csv`:
1. In the first 10 epochs, did validation loss track training loss closely?
2. What is the definition of the "generalization gap"? At what point in training would you start to suspect overfitting?

> 

---

### Q8. Lab Server Execution & Engineering Record
1. Did you run the training inside `tmux` or with `nohup`? How did you verify the job was running without blocking your terminal?
2. Did you modify the STFT window in `models/gtcrn_end2end.py` to square-root Hann (`.pow(0.5)`)? Record your choice and reasoning.

> 
