# Week 9 — Questions

Answer these questions as you complete Lab 1 and Lab 2.

---

### Q1. Distribution Mismatch & Negative SNR Regimes
In `lab1_mismatched.py`, you evaluated the model across SNRs from -5 dB to +15 dB.
1. Why does a model trained strictly on moderate SNRs (0 to 15 dB) suffer steep degradation when tested at negative SNRs (-5 dB)?
2. In machine learning terms, what is this failure mode called, and what data augmentation strategies during training can prevent it?

> 

---

### Q2. Peak Enhancement Gain vs High-SNR Regimes
From your stress test report (`results/stress_test_report.md`):
1. In which SNR regime (negative, moderate, or high) is the improvement gain ($\Delta = \text{Score}_{enh} - \text{Score}_{noisy}$) largest?
2. Why is the numerical gain relatively small at high SNRs (+15 dB), and what risk exists when enhancing already-clean speech?

> 

---

### Q3. Speech Distortion vs Musical Noise Tradeoff
In severe noise conditions (< 0 dB), speech enhancement algorithms encounter a fundamental tradeoff:
1. What is "musical noise" (musical tone artifacts), and what spectral behavior causes human listeners to perceive it?
2. What happens to low-energy speech phonemes (e.g. unvoiced fricatives /s/, /f/, /th/) when the model suppresses noise aggressively?

> 

---

### Q4. Offline Portability for Academic Showcases
In `lab2_make_demo.py`, `demo/index.html` renders SVG waveform previews directly and uses standard HTML5 `<audio>` tags with zero external JavaScript or online CDN dependencies.
Why is complete offline portability essential when delivering thesis deliverables, conference supplementary materials, or lab demonstrations?

> 

---

### Q5. Objective Metric Discrepancies vs Human Perception
Objective metrics like PESQ, STOI, and SI-SDR are proxies for speech quality, but they do not always correlate perfectly with human listening preferences.
Give two concrete examples of artifacts where a model might achieve a high PESQ score yet sound unnatural or annoying to human ears.

> 

---

### Q6. Cross-Language & Tonal Generalization
GTCRN was trained on the VCTK dataset (English speakers with British/American/Scottish accents).
If this model is deployed to Mandarin Chinese or Taiwanese Hokkien without retraining:
1. What acoustic challenges arise specifically from tonal languages (F0 pitch contours and lexical tones)?
2. How would you design a minimal fine-tuning curriculum to adapt the model to Taiwanese Mandarin?

> 
