# Week 3 — Answers

Name:

Answer in your own words; two or three sentences each. Quote the numbers your tools printed where a
question asks about them.

**Q1 (the dataset).** Which test set did your group choose, and why? What can its metrics tell you that the
other test set's cannot — and the other way round?

>

**Q2 (one 16 kHz version).** The course does not let anyone resample the data by hand: one script makes the
16 kHz files and prints fingerprints. Two 16 kHz versions made with different resamplers differ by 0.03–0.05
PESQ. Why does that matter for comparing two groups, and for comparing with a paper?

>

**Q3 (calibration).** Copy your calibration table (`results/calibration.md`). Why do we score the noisy
files — which involve no model at all — before scoring GTCRN? What would you conclude if the noisy row did
not match the paper?

>

**Q4 (reading the official code).** In `infer.py`, which STFT settings and which window does GTCRN use?
What would happen to the scores if your `enhance()` used a different window or hop than the one the
checkpoint was trained with? (You may try it - write down what you tried.)

>

**Q5 (your numbers vs the paper's).** Copy the table from `results/week3_table.md`. How large is the gap
for each metric? Name two things that could make a correctly run official checkpoint score differently
from the paper.

>

**Q6 (listening).** Listen to all 10 samples. What is left of the noise at 2.5 dB and at 12.5 dB? Is the
speech changed? Do the per-file numbers that se_eval printed agree with what you hear — find one case where
they do not.

>

**Q7 (speed).** `infer_dir.py` printed a real-time factor. What does RTF 0.01 mean? Why is "whole file
at once, on a GPU" not the RTF that matters for a hearing aid or a phone call? (Week 8 measures it properly.)

>
