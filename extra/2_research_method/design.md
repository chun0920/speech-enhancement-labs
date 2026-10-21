# Week 4 design — [fill in: your name], [fill in: date]

Fill this in **before** you train anything new, and commit it. `lab4_one_change.py` reads part 5 and
refuses to run if it is empty or not committed — the plan has to exist before the result does.
Replace every placeholder in square brackets. Two or three sentences per box are enough; "unknown"
is an honest answer, an empty box is not.

## 1. Idea card

**Source**: error analysis of the baseline (Lab 2).

**Pain point + evidence**: [fill in: who has which problem, with YOUR numbers from Lab 2 — the largest category, its share, how many of the failures a human could still follow]

**Why the current method cannot solve it**: [fill in: what the Week 3 recipe trains on, and what the network therefore learned "noise" to be]

**Hypothesis**: If [fill in: the one change], then [fill in: which metric on which test condition moves], because [fill in: the physical reason].

**Minimum test**: train the Week 3 GRU with the new training noise — same network, data, loss, steps — with the
three seeds of Lab 1, and compare with Lab 1's three baseline runs on the 20-sentence test set.

**Closest prior work** (open each one yourself and read at least the abstract; write one line each — what it did, and how this week's experiment differs):

- Y. Xu, J. Du, L.-R. Dai, C.-H. Lee, "A regression approach to speech enhancement based on deep neural networks," *IEEE/ACM Trans. Audio, Speech, and Language Processing*, 2015. https://doi.org/10.1109/TASLP.2014.2364452 — [fill in]
- J. Chen, Y. Wang, S. E. Yoho, D. L. Wang, E. W. Healy, "Large-scale training to increase speech intelligibility for hearing-impaired listeners in novel noises," *J. Acoust. Soc. Am.*, 2016. Free full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC5392064/ — [fill in]

**What it needs**: [fill in: data, GPU time — what do you already have?]

**What I am unsure about**: [fill in: one or two sentences]

## 2. One-sentence claim

On [fill in: task], adding [fill in: the one change] to [fill in: the baseline] improves [fill in: metric] on [fill in: test condition] by [fill in: how much you expect], at the cost of [fill in: what may get worse].

## 3. Type of contribution

[fill in: one of: new component / new architecture / new way of training / new loss] — the one cell that changes: [fill in]

## 4. Table 1, before any numbers

In each cell write where its number will come from — not the number:
**R** = we re-run it ourselves, **O** = our own new run, **C** = copied from a paper (only allowed if the test is *identical*),
**—** = not run this week (give the reason below the table).

Table 1: Speech enhancement on 20 LibriSpeech test-clean sentences (0 and 5 dB, averaged).

| Method | Params | pink STOI | pink PESQ | pink SI-SDR | babble STOI | babble PESQ | babble SI-SDR |
|---|---|---|---|---|---|---|---|
| Noisy input (no processing) | 0 | [fill in] | [fill in] | [fill in] | [fill in] | [fill in] | [fill in] |
| MLP, Week 3 (no memory) | 859,137 | [fill in] | [fill in] | [fill in] | [fill in] | [fill in] | [fill in] |
| GRU, Week 3 recipe (baseline) | 856,321 | [fill in] | [fill in] | [fill in] | [fill in] | [fill in] | [fill in] |
| GRU + speech-like noise in training (ours) | 856,321 | [fill in] | [fill in] | [fill in] | [fill in] | [fill in] | [fill in] |
| GTCRN (Rong et al., ICASSP 2024), a published lightweight model | [fill in] | [fill in] | [fill in] | [fill in] | [fill in] | [fill in] | [fill in] |
| Oracle IRM (upper bound — it knows the clean speech) | — | [fill in] | [fill in] | [fill in] | [fill in] | [fill in] | [fill in] |

Why each **—** is empty: [fill in]

## 5. Decided before running

`lab4_one_change.py` reads these lines. Write plain numbers after the `=` (STOI units, e.g. `0.03`).
"Gain" = mean STOI of ours over the 3 seeds − mean STOI of the baseline over its 3 seeds, on babble
(0 and 5 dB averaged). "Pink drop" = baseline − ours, on pink.

```
success_babble_stoi_gain   = [fill in]   # the idea holds if the gain is at least this ...
max_pink_stoi_drop         = [fill in]   # ... and pink drops by no more than this
abandon_babble_stoi_gain   = [fill in]   # below this, drop the idea
predicted_babble_stoi_gain = [fill in]   # your honest guess
predicted_pink_stoi_change = [fill in]   # your honest guess (negative = worse)
```

## 6. Review

Show parts 1–5 to your mentor or a classmate before Lab 4. They write one line here (or you write
what they said). Earlier lines are never changed — a new version gets a new line.

| Date | Version | Reviewer | Verdict (go / change first) |
|---|---|---|---|
| [fill in] | v1 | [fill in] | [fill in] |
