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
cp ~/week9_<yourname>/my_model.py ~/week9_<yourname>/infer_dir.py .
conda activate se
python check_env.py
```

---

## Step 1 — Automated Repository Hygiene Audit (Lab 1, ≈ 30 min)

```bash
python lab1_audit_repo.py
```

Complete the exercises:
1. `audit_file_sizes()`: Flags any files exceeding the 10 MB GitHub threshold.
2. `audit_gitignore_rules()`: Verifies essential ignore patterns for checkpoints and environments.
3. `audit_course_deliverables()`: Checks presence and completeness of all required milestones.
4. `generate_audit_report()`: Exports `results/repo_audit_report.md`.

All 5 checks must pass with **All done!**.

---

## Step 2 — Prepare Presentation Slides (≈ 60 min)

Open `final_presentation_template.md` and build your 15-minute slide deck:
- Slide 1: Title & Executive Summary
- Slide 2: Problem Formulation & Low-Latency Telephony Constraints
- Slide 3: GTCRN Architecture (ERB, Squeezed Conv, DPGRNN)
- Slide 4: Training Methodology & Loss Curves
- Slide 5: Table 1 Reproduction Results (PESQ 2.87, STOI 0.940, SI-SDR 18.83 dB)
- Slide 6: Computational Efficiency (23.7 K trainable params, 33.0 MMACs, RTF < 0.05)
- Slide 7: Mismatched SNR Stress Testing (-5 dB to +15 dB)
- Slide 8: Interactive Audio Showcase Demo
- Slide 9: Conclusion & Master's Thesis Research Proposal

---

## Step 3 — Final Reflection (`ANSWERS.md`)

Answer questions Q1–Q6 in `ANSWERS.md`.

---

## Hand-in

Your completed `speech-enhancement-labs` student repository containing:
- Clean git history with zero oversized files
- `results/repo_audit_report.md` showing **PASS (Ready for Acceptance)**
- Your completed slide deck (PDF or Markdown)
- `ANSWERS.md` filled in
