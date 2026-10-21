# Optional 2 — Research Like a Lab: Reproduce, Listen, Predict, Change One Thing

> **Optional.** Until 2026-10-21 this was Week 4. It builds on Optional 1 (its GRU and its data), so
> do that one first. Inside this folder, "Week 3" means Optional 1. Week 7 of the main path
> (reproducing your baseline's paper numbers) uses the same reproduction rule and `results.csv` habits.

In Week 3 the GRU did well on the noise it trained on and nothing at all on babble, and in Q8 you
proposed one change to fix that. This week you test that change the way a research lab tests an
idea, so that the answer can be trusted: reproduce the baseline first, find out by listening what
it really gets wrong, write down the plan and your prediction *before* any result exists, change
one thing only, run three seeds, and let the numbers you committed to decide.

The workflow follows the research-method guides of Prof. Hung-Shin Lee's lab at National Taiwan
Normal University ([SigSpeechAI-NTNU Lab-Hub](https://github.com/SigSpeechAI-NTNU/Lab-Hub),
CC BY 4.0), cut down to one week. A real project takes months for the same steps.

When you finish you can:

1. reproduce a published result and decide with a fixed rule whether it reproduced
2. keep an append-only `results.csv` in which every number points to the commit that produced it
3. do an error analysis by ear, and tell a problem of the method from a problem of the data
4. write a design before running: a one-sentence claim with its cost, Table 1 with the source of
   every cell, and the numbers that will decide success or failure
5. run a one-change comparison over three seeds and judge it against the spread between seeds

## The path this week

```
Lab 1  reproduce the Week 3 GRU ──► results.csv (3 seeds)
Lab 2  listen to 20 random outputs ──► error_analysis.csv: what is wrong, and can a human still follow?
Lab 3  design.md (card, claim, Table 1, rule, prediction) ──► commit ──► the new training data
Lab 4  train ours (3 seeds) ──► results.csv ──► Table 1 built from the CSV ──► verdict by YOUR rule
```

## Seven rules of the week

These are the habits the week is about. Each lab enforces some of them.

| Rule | Why | Where |
|---|---|---|
| Reproduce before you change anything | a gain over a broken baseline means nothing | Lab 1 |
| Every number points to a commit | three months later you can still find the code that made it | Labs 1, 4 |
| `results.csv` only grows — never delete or edit a row | failed and surprising runs are information too | Labs 1, 4 |
| Listen, don't just read numbers | numbers say how much is wrong, not what | Lab 2 |
| Write the plan, the rule and the prediction before the result | otherwise you will explain any result after the fact | Labs 3, 4 |
| Change one thing; prove it (p_babble = 0 must give Week 3) | otherwise you do not know what caused the change | Lab 3 |
| Three seeds, mean ± std; a gain must beat the seed spread | one run can be lucky | Labs 1, 4 |

## What's in the folder

| File | Purpose | Time |
|---|---|---|
| `README.md` | this guide | |
| `pyproject.toml`, `uv.lock`, `.python-version` | the exact environment, for `uv` | 10 min |
| `check_env.py` | environment check | |
| `lab1_reproduce.py` | Lab 1 — reproduce the Week 3 result; three seeds; `results.csv` | 35 min |
| `lab2_listen.py` | Lab 2 — error analysis: listen to 20 random outputs and label them | 45 min |
| `design.md` | your design: idea card, claim, Table 1, the deciding numbers, review | 30 min |
| `lab3_new_data.py` | Lab 3 — babble from other training speakers; the one-change proof; the 10-example test | 25 min |
| `lab4_one_change.py` | Lab 4 — train ours with three seeds, Table 1 from `results.csv`, the verdict | 30 min |
| `log.md` | your lab notebook: one short entry per day you work | |
| `common.py` | helpers: everything from Weeks 1–3, finished, plus the test set and the records | |
| `ANSWERS.md` | your answers to the questions | |
| `data/` | training clips, 20 test sentences, the babble, the Week 3 sentence (`data/README.txt`) | |
| `results/` | created by the labs: `results.csv`, `error_analysis.csv`, `table1.md` — committed | |
| `out/` | created by the labs: models, sound files, figures — not committed | |

The labs work like the earlier weeks: replace each `raise NotImplementedError` under a `# TODO`, run
the file, and repeat until the last line says **All done!** Do them in order: Lab 2 listens to Lab 1's
model, Lab 4 uses Lab 1's code, Lab 3's dataset and Lab 1's numbers.

## Setup (≈ 15 min)

**1. Your own copy, as its own git repository.** Copy the week out of the clone, then make the copy a
repository. It stays on the server — you never push it anywhere.

```bash
cp -r ~/speech-enhancement-labs/extra/2_research_method ~/extra2_<yourname>
cd ~/extra2_<yourname>
git init -b main
git config user.name "Your Name"
git config user.email "you@example.com"
git add -A
git commit -m "Week 4 starting files"
```

**2. The environment, with uv.** Research labs pin their environments so that a number can be
re-made on another computer months later. This week we stop using the conda environment and use
[uv](https://docs.astral.sh/uv/): `pyproject.toml` says what we need, `uv.lock` pins the exact version
of every package, `.python-version` pins Python. Check whether uv is already installed:

```bash
uv --version
```

If that says `command not found`, install it once (it goes into `~/.local/bin`), then open a new terminal:

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

(The lab server has `wget` but no `curl`. On a computer with `curl`: `curl -LsSf https://astral.sh/uv/install.sh | sh`.)

Then build the environment and check it:

```bash
uv sync
uv run check_env.py
```

`uv sync` creates `.venv/` in this folder with exactly the versions in `uv.lock`. The first time it
downloads about 3 GB (PyTorch for CUDA 13.0) and takes a few minutes; after that it reuses its cache.
Every line of `check_env.py` should start with `[ OK ]`.

From now on, run every lab with `uv run`, for example `uv run lab1_reproduce.py` — no `conda activate`,
no `pip install`. If you need a package, it goes into `pyproject.toml` with `uv add <package>` (not this week).

## Committing before each run

`lab1_reproduce.py` and `lab4_one_change.py` write the commit of your code into every row of
`results.csv`, and refuse to start a full run if a `.py` file or `design.md` has changes that are not
committed. So before a full run:

```bash
git add -A
git commit -m "Lab 1 TODOs done"          # say what changed
```

While you are still writing the TODOs, `uv run lab1_reproduce.py --steps 0` (and the same for Lab 4)
runs only the checks: no training, no commit needed. Commit your `results/` folder too after each lab —
`results.csv` and your labels are part of your work.

## Order and time

```bash
uv run lab1_reproduce.py                  # 3 trainings, about 3 min on the GPU; saves out/gru_s0.pt ... s2.pt
uv run lab2_listen.py                     # saves 20 pairs in out/listen/ and an empty label sheet
                                          # ... listen, fill in results/error_analysis.csv ...
uv run lab2_listen.py --tally             # reveals which was which and counts your labels
                                          # ... fill in design.md parts 1-4 ...
uv run lab3_new_data.py
                                          # ... fill in design.md part 5, get it reviewed, commit ...
uv run lab4_one_change.py                 # 3 trainings, about 3 min; writes results/table1.md
```

Listen to the files in `out/` with VS Code Remote-SSH or copy them to your laptop with `scp -r` as in Week 1.

## Lab 2: the categories

For each of the 20 outputs, write the **biggest** problem you hear — one word per line:

| Category | What you hear in the output |
|---|---|
| `ok` | clean enough — nothing you would complain about |
| `noise-left` | a steady hiss or rumble is still clearly there |
| `voices-left` | other people's voices are still clearly there |
| `muffled` | the target voice sounds dull, far away or underwater |
| `choppy` | bits of words are missing, or the sound pumps on and off |
| `warble` | chirpy, watery, "musical" artifacts |
| `other` | anything else — say what in the note |

And `human_can_tell`: listen to the **noisy** file. `Y` if you can follow the main talker's sentence,
`N` if you cannot. Label what you hear, not what you expect; you are not told which noise each pair has.

## design.md, part by part

| Part | What it is | What a reviewer looks for |
|---|---|---|
| 1. Idea card | the problem with evidence, why the method cannot solve it, your hypothesis (*if … then … because …*), the closest prior work | is the evidence yours (Lab 2), or a guess? |
| 2. One-sentence claim | what improves, on what, by how much — **and at what cost** | a claim with no cost has not been thought through |
| 3. Type of contribution | new component / new architecture / new way of training / new loss — and the one cell that changes | if you cannot name one cell, it is a direction, not a method |
| 4. Table 1, empty | every cell gets its source: R (re-run), O (own run), C (copied — only for an identical test), — (not run, with a reason) | can every number be traced? Is any comparison unfair? |
| 5. Decided before running | the gain that counts as success, the cost you accept, the line below which you give up, and your prediction | are the lines set before the result, and are they reasonable? |
| 6. Review | one line by your mentor or a classmate | |

The prior-work box lists two real papers. Open them and read at least the abstract before you write
the line about each — never cite a paper you have not opened.

## Training time

Each lab that trains runs 3 × 2000 steps: about 3 minutes on the lab server's GPU (a little longer
when others use it), 45–60 minutes on a computer without a GPU. Scoring the 80 test mixtures adds
about 15 seconds per model.

## Your numbers vs. your classmates'

The noisy and oracle rows involve no training, so they must match a classmate's exactly. Trained
models differ slightly from GPU to GPU even with the same seed — and Lab 1 shows you that they also
differ from seed to seed, by more than you might think. That is exactly why Lab 4 compares three
seeds against three seeds.

## Hand-in (before next week)

Your folder `extra2_<yourname>/`, which is a git repository, with:

1. the four finished `lab*.py` files, committed, each ending with **All done!**
2. `results/`: `results.csv`, `error_analysis.csv` (your labels) and `table1.md`, committed
3. `design.md`, filled in and committed **before** the commit that has Lab 4's results — `git log` shows it
4. `log.md` with an entry for each day you worked
5. `ANSWERS.md`, filled in
6. `out/lab4_table1.png` and `out/compare/` (not in git — leave them in the folder)

## Troubleshooting

| Problem | Fix |
|---|---|
| `uv: command not found` | install it (Setup step 2), then open a new terminal so `~/.local/bin` is on your PATH |
| `uv sync` stops while building `pesq` | it needs a C compiler (`gcc`); the lab server has one — on another computer, tell your mentor |
| `check_env.py`: "Python ... from this folder's .venv" fails | you started it with `python check_env.py`; use `uv run check_env.py` |
| `check_env.py`: `CUDA error: no kernel image is available` | your `uv.lock` was changed: copy `pyproject.toml` and `uv.lock` again from the clone, then `uv sync` |
| `check_env.py`: "this folder is its own git repository" fails | Setup step 1: `git init -b main` inside your copy |
| `This folder is inside another git repository` | you are working inside the clone — copy the folder out (Setup step 1) |
| `Your code has changes that are not committed` | `git add -A && git commit -m "..."`, then run again |
| `[WARN] git is not installed` | the labs still run; tell your mentor so git can be installed |
| `out/gru_s0.pt not found` | run `lab1_reproduce.py` (the full run, without `--steps`) first |
| Lab 4: "all five numbers are written in design.md" fails | part 5 needs plain numbers after each `=`, e.g. `0.02` |
| `CUDA out of memory` | someone else is using the GPU heavily — wait a few minutes and run again |
| anything else | screenshot the whole error and send it to your mentor |
