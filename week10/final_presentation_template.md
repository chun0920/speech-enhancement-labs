# Final Presentation — Outline (15 min + 5 min questions)

An outline for the Week 10 acceptance talk to Prof. Hung. Every number on your slides must come from
a file you produced (the file is named next to each slide). Write "not measured" rather than a guess.

---

### Slide 1 — Title (0.5 min)
- Title, your name, your group, date
- Model and dataset you used (e.g. GTCRN on VCTK-DEMAND)

### Slide 2 — The task (1.5 min)
- What speech enhancement does; what your dataset contains (noise types, SNRs, number of test pairs)
- The constraints your model was designed for (causal? streaming? small?) — in your own words

### Slide 3 — The model (2 min)
- Your block diagram from Report 1 / Week 4 (`week4_<yourname>/`), with tensor shapes
- Which parts are causal and why (Week 6)

### Slide 4 — Training (2 min)
- Configuration you actually used: epochs, batch size, learning-rate schedule, STFT window
- Curves: train loss, val loss, val PESQ (`week5_<yourname>/results/training_curves.csv`)
- Wall-clock time and the GPU you trained on

### Slide 5 — Reproduction ◆ (3 min)
- Your table from `week7_<yourname>/results/reproduction_report.md`:

| Row | PESQ | STOI | SI-SDR (dB) | Source |
|---|---|---|---|---|
| Noisy (paper) | 1.97 | 0.921 | 8.45 | GTCRN paper, Table 1 |
| GTCRN (paper) | 2.87 | 0.940 | 18.83 | GTCRN paper, Table 1 |
| Official checkpoint (se_eval) | | | | your Week 3 table |
| Your training | | | | your Week 7 table |
| Gap to paper (%) | | | | ≤ 5 % success, 5–10 % explain, > 10 % investigate |

- Which checkpoint you report and how you chose it (validation set only, never the test set)
- If the gap is above 5 %: what you checked and what you think the cause is

### Slide 6 — Efficiency (2 min)
- Your table from `week8_<yourname>/results/efficiency_table.md`: parameters, MACs per second, RTF,
  per-frame latency — all measured on one named machine (CPU model, GPU model, thread count)
- Compare with the paper's 23.7 K parameters and 39.6 MMACs/s, and explain any difference

### Slide 7 — Mismatched conditions and comparison (2 min)
- Your Week 9 results; the group comparison table

### Slide 8 — Listening (1.5 min)
- Play two or three clips from your demo page (`week9_<yourname>/demo/index.html`):
  noisy → enhanced → clean. Say what you hear, including what still sounds wrong.

### Slide 9 — What you learned, what is next (1 min)
- One thing that went wrong and how you found it
- One concrete question you would like to work on next
