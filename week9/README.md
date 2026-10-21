# Week 9 — Robustness Stress Testing & Interactive Audio Demo Showcase

In previous weeks, you trained GTCRN, reproduced benchmark metrics (PESQ 2.87), and measured
computational efficiency (33.0 MMACs, RTF < 0.05).
However, real-world deployment presents challenges that cannot be captured by standard test sets:
1. **Mismatched SNR stress testing**: How gracefully does the model degrade when SNR drops to severe levels (0 dB, -5 dB)?
2. **Subjective listening & showcase**: How do you convince human listeners, thesis committee members, and clients that your model works effectively?

In this week, you:
1. Build an SNR mixture synthesizer and stress test GTCRN across a wide range of SNRs (-5 dB to +15 dB).
2. Generate `results/stress_test_report.md` analyzing the model's operational breakdown point.
3. Build a standalone, modern HTML5 audio demonstration page (`demo/index.html`) featuring side-by-side audio players and embedded SVG waveform visualizations.

---

## What's in the folder

| File | Purpose |
|---|---|
| `README.md` | this guide |
| `check_env.py` | checks this week's setup |
| `lab1_mismatched.py` | **your work**: SNR mixture synthesizer and stress test evaluator |
| `lab2_make_demo.py` | **your work**: waveform SVG extractor and HTML demo builder |
| `ANSWERS.md` | robustness & subjective evaluation questions |

Prerequisites: `my_model.py` and `infer_dir.py` from earlier weeks.

---

## Step 0 — Setup

```bash
cp -r ~/speech-enhancement-labs/week9 ~/week9_<yourname> && cd ~/week9_<yourname>
cp ~/week8_<yourname>/my_model.py ~/week8_<yourname>/infer_dir.py .
conda activate se
python check_env.py
```

---

## Step 1 — Mismatched SNR Stress Testing (Lab 1, ≈ 30 min)

```bash
python lab1_mismatched.py
```

Complete the exercises:
1. `synthesize_noisy_mixture()`: Controls speech/noise mixture scale for exact target SNR (dB).
2. `evaluate_snr_stress()`: Evaluates enhancement across ladder of SNR levels (-5 dB to 15 dB).
3. `format_stress_test_report()`: Exports `results/stress_test_report.md`.

All 6 checks must pass with **All done!**.

---

## Step 2 — Build Interactive Audio Showcase (Lab 2, ≈ 45 min)

```bash
python lab2_make_demo.py
```

Complete the exercises:
1. `downsample_waveform_envelope()`: Downsamples waveform into peak envelope points.
2. `generate_svg_waveform()`: Renders zero-dependency SVG waveform bars.
3. `build_html_demo()`: Generates `demo/index.html` with side-by-side HTML5 audio controls.

All 4 checks must pass with **All done!**.
You can open `demo/index.html` in your web browser to test interactive playback!

---

## Step 3 — Questions (`ANSWERS.md`)

Answer questions Q1–Q6 in `ANSWERS.md`.

---

## Hand-in

Your folder `week9_<yourname>/` containing:
- Both completed lab scripts (`lab1_mismatched.py` 6/6, `lab2_make_demo.py` 4/4)
- `results/stress_test_report.md`
- `demo/index.html`
- `ANSWERS.md` filled in
