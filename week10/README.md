# Week 10 ◆ — Final Acceptance, Repository Audit & Master's Presentation

Congratulations on reaching the final milestone of the 10-week Speech Enhancement Baseline curriculum!
Over the past 9 weeks, you have transitioned from a beginner exploring audio features to a rigorous
researcher capable of training, verifying, profiling, and presenting a state-of-the-art neural speech enhancement system.

In this capstone week, you:
1. **Audit repository hygiene and completeness**: Run an automated linter (`lab1_audit_repo.py`) to verify that no oversized files (> 10 MB) or untracked checkpoints pollute your repository, and that all 10-week reports and deliverables are complete.
2. **Prepare your 15-minute presentation**: Use `final_presentation_template.md` to build your slide deck for your final evaluation meeting with Prof. Hung.
3. **Reflect on the 10-week journey**: Answer the comprehensive synthesis questions in `ANSWERS.md`.

---

## What's in the folder

| File | Purpose |
|---|---|
| `README.md` | this guide |
| `check_env.py` | checks this week's setup |
| `lab1_audit_repo.py` | **your work**: automated repository linter & deliverables audit |
| `final_presentation_template.md` | slide outline and speaking notes for Prof. Hung |
| `ANSWERS.md` | final 10-week synthesis and reflection questions |

Prerequisites: All completed code, models, reports, and demos from Weeks 1–9.

---

## Step 0 — Setup

```bash
cp -r ~/speech-enhancement-labs/week10 ~/week10_<yourname> && cd ~/week10_<yourname>
conda activate se
python check_env.py
```

---

## Step 1 — Your group's repository and its audit (Lab 1, ≈ 60 min)

**1a.** Finish the four TODOs of `lab1_audit_repo.py` and run `python lab1_audit_repo.py` until you see
**All done!** (5 checks).

**1b.** Put your group's work into one repository of its own — never into `speech-enhancement-labs`, which you
must not push to. Your mentor tells you where it goes (e.g. a private GitHub repository of your group). It needs
at least these files, copied from your week folders:

| File in your repository | From |
|---|---|
| `README.md` | write it: what is in the repository and how to run the evaluation |
| `.gitignore` | must contain `*.tar`, `__pycache__` and `.venv`: checkpoints and environments stay out of git |
| `configs/cfg_train_vctk_full.yaml` | Week 5 |
| `results/week3_table.md` | Week 3 |
| `results/training_curves.csv` | Week 5 (`python lab2_monitor.py <your log>`) |
| `results/reproduction_report.md` | Week 7 |
| `results/efficiency_table.md` | Week 8 |
| `results/stress_test_report.md`, `results/comparison_table.md` | Week 9 |
| `demo/` (`index.html` and `audio/`) | Week 9 |

Then audit it from this folder:

```bash
python lab1_audit_repo.py ~/<your repository>
```

It writes `results/repo_audit_report.md` here: files over 10 MB, missing `.gitignore` patterns, missing files.
Fix them and run it again until the report ends with **PASS (Ready for Acceptance)**.

---

## Step 2 — Prepare Presentation Slides (≈ 60 min)

Open `final_presentation_template.md` and build your 15-minute slide deck. It lists nine slides
(task, model, training, reproduction, efficiency, mismatched conditions, listening, lessons) and, for each
one, the file from an earlier week that the numbers must come from.

---

## Step 3 — Final Reflection (`ANSWERS.md`)

Answer questions Q1–Q6 in `ANSWERS.md`.

---

## Hand-in

- Your group's repository (Step 1b)
- `results/repo_audit_report.md` from this folder, ending with **PASS (Ready for Acceptance)**
- `lab1_audit_repo.py` finished (5/5)
- Your slide deck (PDF or Markdown)
- `ANSWERS.md` filled in
