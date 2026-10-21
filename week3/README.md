# Week 3 — Your Baseline: GTCRN's Official Checkpoint on the Test Set  ◆

From this week on you work with a real, published speech-enhancement model, following the lab's
ten-week plan. This week you run the **authors' own checkpoint** on the standard test set, score it
with the evaluation script both groups share, and put your numbers next to the paper's.

◆ **Acceptance week** (checked by Prof. Hung): the evaluation table and the listening samples.

When you finish you can:

1. set up a published model from its official repository and run its checkpoint on a whole test set
2. prepare a standard dataset exactly the way everyone else in the lab does, and prove it (fingerprints)
3. calibrate the scoring before scoring a model: the "Noisy" row must match the paper
4. compare your numbers with the paper's and explain any gap

## The ten weeks

| Week | Everyone | Hand-in |
|---|---|---|
| 1–2 | Python, PyTorch, audio, STFT, PESQ / STOI / SI-SDR | labs (done) |
| **3 ◆** | **the official checkpoint on the test set, scored** | **evaluation table + listening samples** |
| 4 | read the baseline's code; run its training for a few epochs | layer table, a short training run, a time estimate |
| 5–6 | full training (on lab112); loss and metric curves; score checkpoints on the validation set | curves, validation scores |
| 7 ◆ | training done: your numbers against the paper's, and why they differ | reproduction table |
| 8 | parameters, MACs, real-time factor — on one machine | efficiency table |
| 9 | mismatched test conditions; the comparison table; a demo page | table, demo page |
| 10 ◆ | report; clean repository | repository, table, demo page, report |

Master's students also present the model's design (Weeks 4–6) and run extra baselines (Weeks 6–10).

## The baseline: GTCRN

**GTCRN** — X. Rong, T. Sun, X. Zhang, Y. Hu, C. Zhu, J. Lu, "GTCRN: A speech enhancement model requiring
ultralow computational resources," ICASSP 2024. Code: https://github.com/Xiaobin-Rong/gtcrn (MIT licence).

- about 48 K parameters and 33 MMACs per second (the repository's numbers; the paper reports 23.7 K and 39.6 M
  for an earlier variant — the README explains why), causal, 16 kHz
- official checkpoints for **two datasets**, so your group chooses one of them (Step 1)

## What's in the folder

| File | Purpose |
|---|---|
| `README.md` | this guide |
| `check_env.py` | checks everything this week needs: run it after each setup step |
| `my_model.py` | **your work**: GTCRN wrapped so the course tools can use it (also used in Weeks 4 and 8) |
| `infer_dir.py` | runs `my_model.py` on every file of a folder; checks the outputs; measures the real-time factor |
| `lab1_calibrate.py` | **your work**: check the scoring before scoring the model |
| `lab2_table.py` | **your work**: builds the evaluation table you hand in |
| `paper_numbers.csv` | **your work**: the paper's numbers, with table and page |
| `paper_notes.md` | master's students: notes on the GTCRN paper |
| `ANSWERS.md` | your answers to the questions |
| `results/` | made by the tools: `eval_summary.csv`, per-file scores, `calibration.md`, `week3_table.md`, `samples/` |
| `out/` | made by the tools: the enhanced files |

The shared evaluation tool is not in this folder: it is `tools/se_eval` in the course repository, one copy for
both groups (see its `README.md`).

## Step 0 — A new computer (≈ 30 min, once)

The lab computers start empty. You need four things: the Week 1 conda environment, git, uv, and a working GPU.

```bash
nvidia-smi                    # the GPU and, top right, the highest "CUDA Version" the driver supports
git --version                 # if missing, ask your mentor (installing it needs sudo)
```

**The Python environment.** Install Miniconda and create the `se` environment exactly as in Week 1's README,
with one difference: pick the PyTorch build that matches the driver. `CUDA Version: 13.0` or higher → `cu130`
(the lab server lab112 needs this one); `12.8` → `cu128`; `12.6` → `cu126`.

```bash
conda activate se
pip install torch --index-url https://download.pytorch.org/whl/cu130    # or cu128 / cu126, see above
pip install numpy scipy matplotlib soundfile einops
```

**uv** runs the shared evaluation tool with exactly the package versions written in its `uv.lock`:

```bash
uv --version || curl -LsSf https://astral.sh/uv/install.sh | sh          # then open a new terminal
```

(On a computer without `curl`: `wget -qO- https://astral.sh/uv/install.sh | sh`.)

**The course repository and the evaluation tool:**

```bash
git clone https://github.com/chun0920/speech-enhancement-labs.git ~/speech-enhancement-labs   # or git pull if you have it
cp -r ~/speech-enhancement-labs/week3 ~/week3_<yourname> && cd ~/week3_<yourname>
SE=~/speech-enhancement-labs/tools/se_eval
(cd $SE && uv sync)
alias se_eval="uv run --project $SE $SE/se_eval.py"         # put these two lines in ~/.bashrc to keep them
```

## Step 1 — Choose your dataset (as a group)

GTCRN has an official checkpoint for each of these. The two groups may choose differently.

| | VCTK-DEMAND | DNS3 blind test set |
|---|---|---|
| test data | 824 noisy utterances **with clean references**, 2 speakers, 5 noises, 4 SNRs | 700 noisy recordings (650 real, 50 synthetic singing), **no clean references** |
| metrics | PESQ, STOI, ESTOI, SI-SDR (and DNSMOS) | DNSMOS only (a network that predicts listener ratings) |
| download | test 310 MB; training set (Week 4) 5 GB | test 400 MB; training data must first be synthesized from hundreds of GB of raw speech and noise |
| training (Weeks 5–7) | hours per run | days per run |
| GTCRN paper | Table 1: PESQ 2.87, STOI 0.940, SI-SDR 18.83 | Table 2: DNSMOS P.808 3.44, OVRL 2.70 |

VCTK-DEMAND is the one most papers report, and it can be trained within Weeks 5–7. Choose DNS3 only if your
group has a reason; write the reason in `ANSWERS.md` (Q1).

## Step 2 — The official code (≈ 5 min)

```bash
git clone https://github.com/Xiaobin-Rong/gtcrn.git ~/baselines/gtcrn
cd ~/baselines/gtcrn && git checkout 502ebfa && cd -          # the version this course was tested with
ls ~/baselines/gtcrn/checkpoints                                # model_trained_on_vctk.tar, model_trained_on_dns3.tar
```

Read `~/baselines/gtcrn/README.md` and `infer.py` (20 lines) before Step 5.

## Step 3 — The data (≈ 10–20 min, mostly download)

Everyone must score on **identical** files: the same test set converted to 16 kHz the same way (two 16 kHz
versions of VCTK-DEMAND made with different resamplers differ by 0.03–0.05 PESQ — as much as many papers'
improvements). So nobody resamples by hand; the shared tool does it and prints a fingerprint.

```bash
uv run --project $SE $SE/prepare_vctk_demand.py --root ~/data/VCTK-DEMAND       # VCTK-DEMAND test set
uv run --project $SE $SE/prepare_dns3_blind.py --root ~/data/DNS3-blind         # or: DNS3 blind test set
python check_env.py
```

Compare the fingerprints in `~/data/VCTK-DEMAND/SOURCE.txt` with these; they must be identical:

| folder | files | fingerprint |
|---|---|---|
| `test/clean` | 824 | `f161abab16d8391e` |
| `test/noisy` | 824 | `205263899f4f980b` |

DNS3: `~/data/DNS3-blind/SOURCE.txt` must say 700 files, fingerprint `fb00aef45b7d84b0`. Its download comes from
GitHub and can take 45 minutes; the script tries again by itself after a network error.

## Step 4 — Calibrate the scoring (Lab 1, ≈ 20 min)

```bash
se_eval --enhanced ~/data/VCTK-DEMAND/test/noisy --clean ~/data/VCTK-DEMAND/test/clean \
        --name noisy --dataset vctk-demand-test --note "no processing"
```

(DNS3: `se_eval --enhanced ~/data/DNS3-blind/noisy --dnsmos --name noisy --dataset dns3-blind-track1`.)
Add `--dnsmos` to also get DNSMOS; it adds about 5 minutes. Then open the GTCRN paper, copy its "Noisy" row into
`paper_numbers.csv` (with table and page in `source`), finish the TODOs in `lab1_calibrate.py` and run it.

DNS3: expect the noisy row **not** to match, even with the right data — DNSMOS SIG and BAK miss the paper's noisy
row by 3–5 %. DNSMOS is a neural network with several released versions, and papers seldom say which one they used.
Do not stop there: finish Steps 5–6, compare your GTCRN row with the paper's, find out what you can (Q3), and then
run `python lab1_calibrate.py --dataset dns3-blind-track1 --explained "<what you checked and what you found>"`.

## Step 5 — Wrap the model (≈ 40 min)

Fill in the TODOs of `my_model.py` by reading `~/baselines/gtcrn/infer.py`. Test on 5 files, then run all:

```bash
python infer_dir.py --noisy ~/data/VCTK-DEMAND/test/noisy --out out/enhanced --limit 5
python infer_dir.py --noisy ~/data/VCTK-DEMAND/test/noisy --out out/enhanced
```

(DNS3: `--noisy ~/data/DNS3-blind/noisy`, and set `CHECKPOINT` to `model_trained_on_dns3.tar`.)
On a GPU the 824 files take well under a minute.

## Step 6 — Score, listen, build the table (≈ 40 min)

```bash
se_eval --enhanced out/enhanced --clean ~/data/VCTK-DEMAND/test/clean --name gtcrn-official \
        --dataset vctk-demand-test --note "official checkpoint" \
        --listen $SE/listen_lists/vctk_demand_test.txt --noisy ~/data/VCTK-DEMAND/test/noisy
python lab2_table.py --model gtcrn-official
```

DNS3 instead: no `--clean`, add `--dnsmos`, `--dataset dns3-blind-track1`, the list
`$SE/listen_lists/dns3_blind_track1.txt` with `--noisy ~/data/DNS3-blind/noisy`; then
`python lab2_table.py --model gtcrn-official --dataset dns3-blind-track1 --data ~/data/DNS3-blind`.

The listening list is the same for both groups (10 utterances, chosen by a rule, written at the top of the
file), so the Week 9 demo page can put everyone's outputs side by side. Listen to all 10: noisy, enhanced, clean.

## Hand-in (◆)

Your folder `week3_<yourname>/` with:

1. `my_model.py`, `lab1_calibrate.py`, `lab2_table.py` finished (both labs end with **All done!**)
2. `paper_numbers.csv` filled in, with sources
3. `results/`: `eval_summary.csv`, `calibration.md`, **`week3_table.md`** and **`samples/`** — the acceptance items
4. `out/enhanced/inference_log.json`
5. `ANSWERS.md`; master's students also `paper_notes.md`

## Troubleshooting

| Problem | Fix |
|---|---|
| `CondaToSNonInteractiveError: Terms of Service have not been accepted` | a new Miniconda asks you to accept Anaconda's terms for its default channels. Either read and accept them yourself (the message shows the command), or use the free conda-forge channel instead: `conda create -n se -c conda-forge --override-channels python=3.11` |
| `CUDA error: no kernel image is available` | the PyTorch build is too old for this GPU: reinstall with the index from Step 0 |
| `ModuleNotFoundError: einops` | `pip install einops` in the `se` environment (GTCRN needs it) |
| `se_eval: command not found` | the `alias` line of Step 0 is missing in this terminal |
| `uv sync` fails building `pesq` | it compiles C code: `gcc` must be installed (ask your mentor) |
| se_eval: `files are not 16 kHz mono` | use the files made by `prepare_vctk_demand.py`, never resample by hand |
| se_eval: `files cut to the same length` | your `enhance()` returns a different length: use `torch.istft(..., length=len(noisy))` |
| `Missing key(s) in state_dict` | wrong model class or wrong checkpoint key — print `ckpt.keys()` |
| your noisy row does not match the paper | check the fingerprints (Step 3) and that you copied the right row of the right table |
| anything else | screenshot the whole error and send it to your mentor |
