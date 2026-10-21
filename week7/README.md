# Week 7 ◆ — Paper Reproduction & Scientific Gap Analysis

This is Acceptance Week (◆). Prof. Hung checks whether your GTCRN baseline reproduction succeeds.

You take your trained model (the checkpoint chosen on the validation set in Week 6), run inference
on the untouched VCTK-DEMAND test set (824 pairs), score it with `se_eval`, and compare your
numbers against Table 1 of the GTCRN paper.

### Target Numbers (GTCRN Paper, Table 1):
| System | PESQ (wb) | STOI | SI-SDR (dB) |
|---|---|---|---|
| Unprocessed Noisy | 1.97 | 0.921 | 8.45 |
| **GTCRN (Paper)** | **2.87** | **0.940** | **18.83** |

### Acceptance Criteria (from Elective 2):
- **Relative gap $\le 5\%$**: Reproduction Success!
- **Relative gap $5\% - 10\%$**: Acceptable with written explanation of discrepancy.
- **Relative gap $> 10\%$**: Systematic issue; debug with mentor before acceptance.

When you finish you can:
1. Verify dataset integrity using cryptographic SHA256 fingerprints
2. Execute publication-grade scientific reproduction protocols
3. Quantify reproduction gaps on PESQ, STOI, and SI-SDR
4. Generate `results/reproduction_report.md` for Prof. Hung's acceptance review

---

## What's in the folder

| File | Purpose |
|---|---|
| `README.md` | this guide |
| `check_env.py` | checks this week's setup |
| `paper_numbers.csv` | official paper metrics from Table 1 |
| `lab1_final_eval.py` | **your work**: test set fingerprint verification and inference pipeline |
| `lab2_reproduce_check.py` | **your work**: relative gap computation, criteria classifier, and report generator |
| `ANSWERS.md` | reproduction analysis questions |

Prerequisites: `my_model.py`, `infer_dir.py`, `setrain_model.py` from earlier weeks.

---

## Step 0 — Setup

```bash
cp -r ~/speech-enhancement-labs/week7 ~/week7_<yourname> && cd ~/week7_<yourname>
cp ~/week6_<yourname>/my_model.py ~/week6_<yourname>/infer_dir.py ~/week6_<yourname>/setrain_model.py .
conda activate se
python check_env.py
```

---

## Step 1 — Final Test Set Evaluation (Lab 1, ≈ 30 min)

```bash
python lab1_final_eval.py
```

Complete the exercises:
1. `compute_folder_fingerprint()`: Computes the 16-hex SHA256 dataset fingerprint.
2. `verify_test_fingerprints()`: Confirms `test/clean` (`f161abab16d8391e`) and `test/noisy` (`205263899f4f980b`).
3. `format_test_eval_run()`: Formats commands for test set inference and `se_eval`.
4. `load_reproduced_numbers()`: Extracts reproduced test metrics from `results/eval_summary.csv`.

Once all 7 checks pass:
Run inference and evaluation on the real test set with your chosen checkpoint:
```bash
RUN=$(ls -d ~/exp/gtcrn_vctk_full_* | tail -1)
EP=200        # the epoch you chose in Week 6 from your se_eval scores on val/ (or 200, the last one)
CKPT=$RUN/checkpoints/model_$(printf %03d $EP).tar   # never best_model_*.tar: it holds the last epoch's weights (Week 5)

python infer_dir.py --adapter setrain_model --checkpoint $CKPT \
                    --noisy ~/data/VCTK-DEMAND/test/noisy --out out/test_enhanced
se_eval --enhanced out/test_enhanced --clean ~/data/VCTK-DEMAND/test/clean \
        --name gtcrn-reproduced --dataset vctk-demand-test
```

---

## Step 2 — Reproduction Check & Gap Analysis (Lab 2, ≈ 30 min)

```bash
python lab2_reproduce_check.py
```

Complete the exercises:
1. `load_paper_numbers()`: Reads `paper_numbers.csv`.
2. `compute_relative_gap()`: Computes percentage gap against Table 1.
3. `classify_reproduction()`: Evaluates the $\le 5\%$ threshold.
4. `generate_reproduction_report()`: Generates `results/reproduction_report.md`.

Once all 8 checks pass, run it against your reproduced numbers to generate your acceptance report!

---

## Step 3 — Acceptance Review Preparation (`ANSWERS.md`)

Answer questions Q1–Q8 in `ANSWERS.md` and prepare to present `results/reproduction_report.md` to Prof. Hung.

---

## Hand-in

Your folder `week7_<yourname>/` containing:
- Both lab scripts completed (Lab 1 7/7, Lab 2 8/8)
- `results/eval_summary.csv` containing the `gtcrn-reproduced` row on `vctk-demand-test`
- `results/reproduction_report.md` showing **SUCCESS**
- `ANSWERS.md` filled in
