"""
Week 4 - Lab 1: Reproduce first                                  run:  uv run lab1_reproduce.py

This week you work the way a research lab works. Every project starts the same way: before you
change anything, re-run the method you want to improve - exactly as it was published - and check
that you get the published numbers. Our "paper" is Week 3. Its recipe (the GRU, the data, 2000 steps,
seed 0) and its published result on the Week 2 sentence with pink noise at 0 dB:

    PESQ 1.28    STOI 0.717    SI-SDR 8.76 dB

The lab rule: until the baseline reproduces, you do not change the method. If your numbers are
different, either your code differs from the paper's or the paper is not reproducible - and you
must know which before you build on it.

Then you train the same recipe with two more seeds and score all three runs on this week's bigger
test set (20 sentences from 20 speakers, pink noise and babble, 0 and 5 dB). Every number goes into
results/results.csv: one row per number, and no row is ever deleted.

Your job: fill in the TODOs, commit (README > Setup), then run this file. It trains three networks:
about 3 minutes on the lab server's GPU.
Quick test while you are still writing the TODOs (trains nothing, writes nothing):
    uv run lab1_reproduce.py --steps 0
"""
import argparse
import csv
import datetime
from pathlib import Path

import numpy as np
import torch

from common import (DEVICE, FIELDS, OUT, RESULTS, RESULTS_CSV, SPLITS, TEST_SET_NAME, TEST_SPEECH, MaskGRU,
                    NoisyMixDataset, all_passed, check, code_version, enhance, evaluate, load_audio, make_noise,
                    mix_at_snr, oracle, read_results, require_committed, rows_for, score_split_means, section,
                    summary, test_set, train, train_files)

PUBLISHED = {"PESQ": 1.28, "STOI": 0.717, "SI-SDR": 8.76}   # Week 3 answer key, GRU, pink 0 dB, seed 0
SEEDS = (0, 1, 2)
BASELINE = "gru-week3"


# ---------------------------------------------------------------- Exercise 1.1
def make_run_id(name, seed, day):
    """A run's name: 'YYYYMMDD_name_sSEED', e.g. make_run_id("gru-week3", 1, date(2026, 10, 20)) -> '20261020_gru-week3_s1'.

    day is a datetime.date; day.strftime("%Y%m%d") gives '20261020'.
    The same name is used in results.csv and for the model file, so a number can be traced to its model.
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.2
def append_rows(rows, path=RESULTS_CSV):
    """Add rows (a list of dicts with exactly the keys in FIELDS) to the END of the CSV file at `path`.

    - write the header line only if the file does not exist yet (or is empty)
    - never rewrite or delete what is already in the file: open it with mode "a" (append)
    - csv.DictWriter(f, fieldnames=FIELDS) does the formatting; .writeheader() and .writerows(rows)
    - open the file with newline="" (the csv module wants that)
    A row with a key that is not in FIELDS must raise an error - DictWriter does that for you.
    """
    path = Path(path)
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.3
def relative_gap(mine, published):
    """|mine - published| / |published|, e.g. relative_gap(1.20, 1.28) = 0.0625 (6.25 %)."""
    # TODO
    raise NotImplementedError


def verdict(gap):
    """The lab's rule for a reproduction:
        gap <= 0.05          -> "reproduced"
        0.05 < gap <= 0.10   -> "close: write down possible reasons"
        gap > 0.10           -> "NOT reproduced: find the cause before changing anything"
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.4
def mean_std(values):
    """(mean, standard deviation) of a list of numbers. Use the SAMPLE standard deviation, which divides by
    n - 1: np.std(values, ddof=1). With 3 seeds that is the honest estimate of how much a run varies."""
    # TODO
    raise NotImplementedError


def week3_exam(model):
    """(Given) The Week 3 condition: the Week 2 sentence + the Week 2 pink noise at 0 dB."""
    clean = load_audio(TEST_SPEECH)
    noisy, _ = mix_at_snr(clean, make_noise(len(clean), tilt=1.0), 0)
    return evaluate(clean, enhance(model, noisy))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000, help="training steps per seed; 0 = only run the checks")
    args = ap.parse_args()
    today = datetime.date.today()

    section("Exercise 1.1  run ids")
    check("make_run_id('gru-week3', 1, 2026-10-20) == '20261020_gru-week3_s1'",
          lambda: make_run_id("gru-week3", 1, datetime.date(2026, 10, 20)) == "20261020_gru-week3_s1",
          'f"{day.strftime(\'%Y%m%d\')}_{name}_s{seed}"')

    section("Exercise 1.2  append-only results file")
    tmp = RESULTS / "_check.csv"
    tmp.unlink(missing_ok=True)
    row = {k: "x" for k in FIELDS}
    try:
        append_rows([row, row], tmp)
        first = tmp.read_text()
        append_rows([{**row, "value": "0.5"}], tmp)
        second = tmp.read_text()
        with open(tmp, newline="") as f:
            back = list(csv.DictReader(f))
    except NotImplementedError:
        check("append_rows() is written", lambda: False, "finish the TODO in Exercise 1.2")
    else:
        check("header written once, then 3 rows", lambda: len(back) == 3 and second.count("run_id") == 1,
              "write the header only when the file is new")
        check("the old rows are left exactly as they were", lambda: second.startswith(first) and back[2]["value"] == "0.5",
              'open(path, "a", newline="") - "a" adds at the end, "w" would erase the file')

        def bad_row():
            try:
                append_rows([{**row, "my_column": 1}], tmp)
            except ValueError:
                return True
            return False
        check("a row with an unknown column is refused", bad_row, "use csv.DictWriter(f, fieldnames=FIELDS)")
    tmp.unlink(missing_ok=True)

    section("Exercise 1.3  the reproduction rule")
    check("relative_gap(1.20, 1.28) = 0.0625", lambda: abs(relative_gap(1.20, 1.28) - 0.0625) < 1e-9)
    check("relative_gap works for negative numbers too", lambda: abs(relative_gap(-4.5, -5.0) - 0.1) < 1e-9, "abs() of both")
    check("4 % -> reproduced, 7 % -> close, 12 % -> NOT reproduced",
          lambda: verdict(0.04) == "reproduced" and verdict(0.07).startswith("close") and verdict(0.12).startswith("NOT"))
    check("exactly 5 % still counts as reproduced", lambda: verdict(0.05) == "reproduced", "gap <= 0.05")

    section("Exercise 1.4  mean and spread of the seeds")
    check("mean_std([1, 2, 3]) == (2.0, 1.0)", lambda: np.allclose(mean_std([1, 2, 3]), (2.0, 1.0)), "np.std(values, ddof=1)")

    if not all_passed():
        summary()
        raise SystemExit(1)
    if args.steps == 0:
        print("\n  --steps 0: checks only. Commit, then run without --steps for the real runs.")
        summary()
        raise SystemExit(0)

    version = code_version()
    print(f"\n  code version: {version}   device: {DEVICE}")
    if args.steps == 2000:
        require_committed(version)
    else:
        print(f"  --steps {args.steps} is a test run: nothing is written to results.csv")

    section(f"Training the Week 3 recipe, seeds {SEEDS} ({args.steps} steps each)")
    models = {}
    for seed in SEEDS:
        print(f"  seed {seed}:")
        model = MaskGRU()
        train(model, NoisyMixDataset(train_files(), seed=seed), args.steps, seed=seed)
        models[seed] = model
        if args.steps == 2000:
            torch.save(model.state_dict(), OUT / f"gru_s{seed}.pt")

    section("Reproduction: the Week 3 condition (Week 2 sentence, pink noise, 0 dB)")
    w3 = {seed: week3_exam(m) for seed, m in models.items()}
    print(f"  {'':<12}" + "".join(f"{m:>10}" for m in PUBLISHED))
    print(f"  {'published':<12}" + "".join(f"{v:>10.3f}" for v in PUBLISHED.values()))
    for seed in SEEDS:
        print(f"  {'seed ' + str(seed):<12}" + "".join(f"{w3[seed][m]:>10.3f}" for m in PUBLISHED))
    print("\n  seed 0 (the seed of the paper) against the published numbers:")
    for m, pub in PUBLISHED.items():
        gap = relative_gap(w3[0][m], pub)
        print(f"    {m:<7} {w3[0][m]:7.3f} vs {pub:6.3f}   gap {100 * gap:4.1f} %   -> {verdict(gap)}")
    print("\n  the three seeds, mean +- std (questions Q1, Q2):")
    for m in PUBLISHED:
        mu, sd = mean_std([w3[s][m] for s in SEEDS])
        print(f"    {m:<7} {mu:7.3f} +- {sd:.3f}   (largest - smallest: {max(w3[s][m] for s in SEEDS) - min(w3[s][m] for s in SEEDS):.3f})")

    section("This week's exam: 20 sentences x {pink, babble} x {0, 5 dB}")
    items = test_set()
    print(f"  {len(items)} mixtures; scoring 'noisy', 'oracle' and the three GRUs ...")
    scores = {"noisy": score_split_means(lambda it: it["noisy"], items), "oracle-irm": score_split_means(oracle, items)}
    for seed in SEEDS:
        scores[f"s{seed}"] = score_split_means(lambda it, m=models[seed]: enhance(m, it["noisy"]), items)
    print(f"  {'STOI':<12}" + "".join(f"{s:>12}" for s in SPLITS))
    for name, sc in scores.items():
        print(f"  {name:<12}" + "".join(f"{sc[s]['STOI']:>12.3f}" for s in SPLITS))

    check("the baseline helps on pink: STOI above 'noisy' at 0 dB for every seed",
          lambda: all(scores[f"s{s}"]["pink-0dB"]["STOI"] > scores["noisy"]["pink-0dB"]["STOI"] + 0.05 for s in SEEDS),
          "is this a full 2000-step run?")
    check("babble still fools it: STOI gain under 0.01 at both babble SNRs (the Week 3 finding)",
          lambda: all(abs(scores[f"s{s}"][sp]["STOI"] - scores["noisy"][sp]["STOI"]) < 0.01
                      for s in SEEDS for sp in SPLITS if sp.startswith("babble")))

    if args.steps == 2000:
        rows = []
        for seed in SEEDS:
            rid = make_run_id(BASELINE, seed, today)
            rows += rows_for(rid, "lab1_reproduce.py", "week3-sentence", {"pink-0dB": w3[seed]}, seed, version,
                             "Week 3 recipe unchanged; reproduction" if seed == 0 else "Week 3 recipe unchanged")
            rows += rows_for(rid, "lab1_reproduce.py", TEST_SET_NAME, scores[f"s{seed}"], seed, version, "Week 3 recipe unchanged")
        rows += rows_for(make_run_id("noisy", 0, today), "lab1_reproduce.py", TEST_SET_NAME, scores["noisy"], "-", version,
                         "no processing")
        rows += rows_for(make_run_id("oracle-irm", 0, today), "lab1_reproduce.py", TEST_SET_NAME, scores["oracle-irm"], "-",
                         version, "upper bound: knows the clean speech")
        before = len(read_results())
        append_rows(rows)
        print(f"\n  added {len(rows)} rows to {RESULTS_CSV} (it now has {before + len(rows)}); models saved as out/gru_s*.pt")
        check("results.csv grew by exactly the new rows", lambda: len(read_results()) == before + len(rows))
    summary()
