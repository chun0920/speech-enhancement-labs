# Week 6 — Mid-Training Checkpoints & Causal Architecture Verification

In Week 5, you launched the full 200-epoch training schedule on lab112 and analyzed loss dynamics.
As training progresses past milestones (e.g., epochs 50, 100, 150, 200), checkpoints are saved
to disk.

Before Week 7's final evaluation on the test set, you must inspect and score intermediate checkpoints
on the **validation set (`val/`)**. You will track metric improvement across training, implement
automated checkpoint scoring, and choose the optimal model state.
Furthermore, you will experimentally prove that GTCRN respects strict temporal causality (0 future
frame lookahead).

When you finish you can:

1. Systematically evaluate milestone checkpoints on the validation set using `se_eval`
2. Formulate a scientific checkpoint selection strategy (peak validation PESQ vs lowest loss vs last epoch)
3. Explain how causal convolution padding ($k_t - 1, 0$) and unidirectional RNNs guarantee real-time causality
4. Implement a future-perturbation test to mathematically verify that zero information leaks backward from the future
5. Calculate the algorithmic buffering latency of an overlap-add streaming pipeline
6. Master's students: Report 3 — Causal Architecture Design

---

## What's in the folder

| File | Purpose |
|---|---|
| `README.md` | this guide |
| `check_env.py` | checks this week's setup |
| `lab1_checkpoint_eval.py` | **your work**: milestone checkpoint detector, evaluator, and selection tool |
| `lab2_causality.py` | **your work**: causal convolution padding rules and temporal perturbation test |
| `report3_template.md` | master's students: Report 3 (Causal Architecture Design) |
| `ANSWERS.md` | questions on checkpoint selection and streaming causality |

Prerequisites: `my_model.py`, `infer_dir.py`, `dataloader_vctk.py`, and `setrain_model.py` from previous weeks.

---

## Step 0 — Setup

```bash
cp -r ~/speech-enhancement-labs/week6 ~/week6_<yourname> && cd ~/week6_<yourname>
cp ~/week5_<yourname>/my_model.py ~/week5_<yourname>/infer_dir.py ~/week5_<yourname>/dataloader_vctk.py ~/week4_<yourname>/setrain_model.py .
conda activate se
python check_env.py
```

---

## Step 1 — Milestone Checkpoint Evaluation (Lab 1, ≈ 45 min)

```bash
python lab1_checkpoint_eval.py
```

Finish the four exercises:
1. `find_available_checkpoints()`: Indexes all regular and `best_model` checkpoints in chronological order.
2. `format_eval_command()`: Formats execution strings for `infer_dir.py --adapter setrain_model` and `se_eval`.
3. `parse_eval_summary()`: Reads `results/eval_summary.csv` and parses metric columns.
4. `select_best_checkpoint()`: Selects the top checkpoint by peak validation PESQ.

Run until all 9 checks show `[PASS]` and you see **All done!**.

### Running validation evaluation on your server checkpoints:
Once your training on lab112 has produced checkpoints (e.g. `model_050.tar`, `model_100.tar`; never `best_model_*.tar`, see Week 5):
```bash
RUN=$(ls -d ~/exp/gtcrn_vctk_full_* | tail -1)
# Example: Evaluate milestone epoch 100 on val/
python infer_dir.py --adapter setrain_model --checkpoint $RUN/checkpoints/model_100.tar \
                    --noisy ~/data/VCTK-DEMAND/val/noisy --out out/val_ep100
se_eval --enhanced out/val_ep100 --clean ~/data/VCTK-DEMAND/val/clean \
        --name gtcrn-val-ep100 --dataset vctk-demand-val
```

Choose the checkpoint with the best validation scores from your own se_eval rows; Week 7 tests its
`model_<epoch>.tar`, and only that one. Once your validation rows are in `results/eval_summary.csv`,
`python lab1_checkpoint_eval.py` also prints which one Exercise 1.4 picks.

---

## Step 2 — Temporal Causality Verification (Lab 2, ≈ 45 min)

```bash
python lab2_causality.py
```

Finish the three exercises:
1. `is_padding_causal()`: Verifies whether convolution temporal padding is causal ($k_t - 1, 0$) vs symmetric.
2. `calculate_algorithmic_latency()`: Computes the algorithmic latency of causal STFT processing: one window, not one hop.
3. `verify_temporal_causality()`: Injects random perturbations into future audio samples ($t \ge T_0$) and proves that past enhanced audio ($t < T_0$) is unaffected.

Run until all 8 checks show `[PASS]` and you see **All done!**.

---

## Step 3 — Master's Students: Report 3

Master's students: fill in `report3_template.md` (approx. 10-minute presentation to Prof. Hung).
Topic: **Causal 架構設計 (Causal Architecture Design)**:
- Latency breakdown: Algorithmic latency vs processing latency vs transmission buffer
- Asymmetric causal Conv2d padding vs bidirectional lookahead
- Unidirectional DPGRNN temporal recurrence
- Comparison between causal and non-causal speech enhancement models

---

## Step 4 — Questions (`ANSWERS.md`)

Answer questions Q1–Q8 in `ANSWERS.md`.

---

## Hand-in

Your folder `week6_<yourname>/` containing:
- Both lab scripts finished (`lab1_checkpoint_eval.py` 9/9, `lab2_causality.py` 8/8, both showing **All done!**)
- `results/eval_summary.csv` with evaluated validation milestones
- `ANSWERS.md` filled in
- Master's students: `report3_template.md` filled in
