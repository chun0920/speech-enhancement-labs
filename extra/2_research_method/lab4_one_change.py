"""
Week 4 - Lab 4: Change one thing, three seeds, Table 1 from the CSV     run:  uv run lab4_one_change.py

design.md is filled in and committed: the claim, Table 1 with its source codes, and the numbers that
decide success or failure - all written before any result exists. Now:

  1. train "ours" - the Week 3 GRU, trained with Lab 3's BabbleMixDataset - with the same three seeds
     as the baseline in Lab 1
  2. score it on the same 80 test mixtures and append every number to results/results.csv
  3. build Table 1 FROM results.csv - never by copying numbers by hand
  4. judge the result by the rule you wrote in design.md, not by how you feel about it

A gain is only real if it is larger than the difference between two seeds of the same recipe.
A gain that is suspiciously large is first treated as a bug (is a test voice hiding in the training
data? is the test set the same?), and only then as a discovery.

Your job: fill in the TODOs, commit, then run this file (about 3-4 minutes on the lab server's GPU).
Quick test while you are still writing the TODOs (trains nothing, writes nothing):
    uv run lab4_one_change.py --steps 0
"""
import argparse
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (DEVICE, OUT, RESULTS, SPLITS, TEST_SET_NAME, MaskGRU, all_passed, check, code_version, enhance,
                    load_gru, read_results, require_committed, rows_for, save_wav, score_split_means, section, summary,
                    system_of, test_set, train, train_files)
from lab1_reproduce import SEEDS, append_rows, make_run_id, mean_std
from lab3_new_data import BabbleMixDataset

OURS, BASELINE = "gru-speechnoise", "gru-week3"
NAMES = ("success_babble_stoi_gain", "max_pink_stoi_drop", "abandon_babble_stoi_gain",
         "predicted_babble_stoi_gain", "predicted_pink_stoi_change")
PINK, BAB = [s for s in SPLITS if s.startswith("pink")], [s for s in SPLITS if s.startswith("babble")]


# ---------------------------------------------------------------- Exercise 4.1
def read_numbers(text):
    """The 'name = number' lines of design.md part 5 -> {name: float}.

    For every line of text (text.splitlines()):
      - cut off a comment: line.split("#")[0]
      - skip it if there is no "="; otherwise split it ONCE at "=" and .strip() both sides
      - keep it only if the name is in NAMES and float(value) works (a ValueError means: not a number yet)
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 4.2
def newest_per_seed(rows):
    """rows: results.csv rows of ONE system, split and metric, in file order. Returns {seed: value as float}.

    Rows are never deleted, so running a lab again adds new rows for the same seed: keep the NEWEST
    one (the last in the file) for each seed. Skip rows whose value is "NA" (a failed run).
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 4.3
def judge(gain, pink_drop, spread, baseline_value, rule):
    """The verdict on the hypothesis. Check in THIS order and return the first that applies:

      1. gain > 0.30 * baseline_value     -> "suspicious: a gain this large is a bug until proven otherwise"
      2. gain <= 2 * spread               -> "inconclusive: the gain is within the seed-to-seed spread"
      3. gain < rule["abandon_babble_stoi_gain"]                          -> "abandon"
      4. gain >= rule["success_babble_stoi_gain"]
         and pink_drop <= rule["max_pink_stoi_drop"]                      -> "holds"
      5. anything else                    -> "inconclusive: between your two lines, or too costly on pink"
    """
    # TODO
    raise NotImplementedError


def seed_values(rows, system, splits, metric):
    """(Given) {seed: value averaged over `splits`} for one system on this week's test set, newest run per seed."""
    per_split = [newest_per_seed([r for r in rows if system_of(r["run_id"]) == system and r["dataset"] == TEST_SET_NAME
                                  and r["split"] == sp and r["metric"] == metric]) for sp in splits]
    seeds = set.intersection(*(set(d) for d in per_split)) if per_split else set()
    return {s: float(np.mean([d[s] for d in per_split])) for s in sorted(seeds)}


def cell(rows, system, splits, metric):
    """(Given) One Table 1 cell: 'mean +- std' over seeds, or the single value for noisy / oracle."""
    v = list(seed_values(rows, system, splits, metric).values())
    if not v:
        return "?"
    fmt = ".2f" if metric == "SI-SDR" else ".3f"
    return f"{v[0]:{fmt}}" if len(v) == 1 else f"{np.mean(v):{fmt}} ± {np.std(v, ddof=1):{fmt}}"


def write_table1(rows, path):
    """(Given) Table 1 and the per-SNR analysis table, built only from results.csv."""
    metrics = ("STOI", "PESQ", "SI-SDR")
    lines = [f"Table 1: 20 LibriSpeech test-clean sentences, 0 and 5 dB averaged; mean ± std over seeds. "
             f"Built from results/results.csv on {datetime.date.today()}.", "",
             "| Method | Params | " + " | ".join(f"pink {m}" for m in metrics) + " | " + " | ".join(f"babble {m}" for m in metrics) + " |",
             "|---|---|" + "---|" * 6]
    table = [("Noisy input (no processing)", "0", "noisy"), ("MLP, Week 3 (no memory)", "859,137", None),
             ("GRU, Week 3 recipe (baseline)", "856,321", BASELINE), ("GRU + speech-like noise in training (ours)", "856,321", OURS),
             ("GTCRN (Rong et al., ICASSP 2024)", "", None), ("Oracle IRM (upper bound)", "—", "oracle-irm")]
    for name, params, system in table:
        cells = [cell(rows, system, PINK, m) for m in metrics] + [cell(rows, system, BAB, m) for m in metrics] if system else ["—"] * 6
        lines.append(f"| {name} | {params} | " + " | ".join(cells) + " |")
    lines += ["", "Analysis: STOI by SNR (where does the change help?)", "",
              "| Method | " + " | ".join(SPLITS) + " |", "|---|" + "---|" * len(SPLITS)]
    for name, system in (("Noisy", "noisy"), ("Baseline", BASELINE), ("Ours", OURS), ("Oracle IRM", "oracle-irm")):
        lines.append(f"| {name} | " + " | ".join(cell(rows, system, [sp], "STOI") for sp in SPLITS) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def plot(rows, path):
    """(Given) STOI per test condition: baseline vs ours (bars = mean, lines = each seed), noisy and oracle as marks."""
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(SPLITS))
    for k, (system, label) in enumerate(((BASELINE, "baseline (Week 3 recipe)"), (OURS, "ours (+ speech-like noise)"))):
        per = [seed_values(rows, system, [sp], "STOI") for sp in SPLITS]
        ax.bar(x + (k - 0.5) * 0.36, [np.mean(list(p.values())) for p in per], 0.34, label=label, alpha=0.8)
        for j, p in enumerate(per):
            ax.plot([x[j] + (k - 0.5) * 0.36] * len(p), list(p.values()), "k.", ms=4)
    for system, style, label in (("noisy", "kx", "noisy input"), ("oracle-irm", "k_", "oracle IRM")):
        ax.plot(x, [list(seed_values(rows, system, [sp], "STOI").values())[0] for sp in SPLITS], style, ms=12, mew=2, label=label)
    ax.set(xticks=x, xticklabels=SPLITS, ylabel="STOI", title="Week 4: one change, three seeds", ylim=(0.55, 1.0))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000, help="training steps per seed; 0 = only run the checks")
    args = ap.parse_args()

    section("Exercise 4.1  reading design.md")
    example = "# notes\nsuccess_babble_stoi_gain = 0.03   # my line\nmax_pink_stoi_drop=0.01\nabandon_babble_stoi_gain = [fill in]\n**R** = re-run\n"
    check("reads 'name = number', ignores comments, placeholders and other lines",
          lambda: read_numbers(example) == {"success_babble_stoi_gain": 0.03, "max_pink_stoi_drop": 0.01},
          'line.split("#")[0], then line.split("=", 1), then float(value) inside try / except ValueError')

    section("Exercise 4.2  the newest row of each seed")
    rows_ex = [{"seed": "0", "value": "0.70"}, {"seed": "1", "value": "0.71"}, {"seed": "0", "value": "0.72"}, {"seed": "2", "value": "NA"}]
    check("a re-run replaces the older value; NA rows are skipped",
          lambda: newest_per_seed(rows_ex) == {"0": 0.72, "1": 0.71}, "go through the rows in order; a later row overwrites")

    section("Exercise 4.3  the verdict")
    rule = {"success_babble_stoi_gain": 0.02, "max_pink_stoi_drop": 0.01, "abandon_babble_stoi_gain": 0.005}
    check("0.03 gain, 0.005 pink drop, spread 0.002 -> holds", lambda: judge(0.03, 0.005, 0.002, 0.7, rule) == "holds")
    check("0.03 gain but pink drops 0.02 -> inconclusive", lambda: judge(0.03, 0.02, 0.002, 0.7, rule).startswith("inconclusive"))
    check("0.003 gain, spread 0.002 -> inconclusive (not more than 2 x the spread)",
          lambda: judge(0.003, 0.0, 0.002, 0.7, rule).startswith("inconclusive: the gain is within"))
    check("0.004 gain with spread 0.001 -> abandon", lambda: judge(0.004, 0.0, 0.001, 0.7, rule) == "abandon")
    check("0.25 gain on a 0.7 baseline -> suspicious", lambda: judge(0.25, 0.0, 0.002, 0.7, rule).startswith("suspicious"),
          "check the rules in the order given")

    section("design.md part 5")
    try:
        rule = read_numbers(Path("design.md").read_text(encoding="utf-8"))
    except NotImplementedError:
        rule = {}
    missing = [n for n in NAMES if n not in rule]
    check("all five numbers are written in design.md", lambda: not missing, f"still missing: {', '.join(missing)}")
    if not all_passed():
        summary()
        raise SystemExit(1)
    if args.steps == 0:
        print("\n  --steps 0: checks only. Commit design.md and your code, then run without --steps.")
        summary()
        raise SystemExit(0)

    version = code_version()
    print(f"\n  code version: {version}   device: {DEVICE}")
    if args.steps != 2000:
        raise SystemExit("  the comparison is only fair with the same 2000 steps as the baseline - run without --steps")
    require_committed(version)
    rows = read_results()
    have = seed_values(rows, BASELINE, BAB, "STOI")
    check("Lab 1's three baseline runs are in results.csv", lambda: len(have) == len(SEEDS), "run lab1_reproduce.py first")
    if not all_passed():
        summary()
        raise SystemExit(1)
    print("  decided before running:  " + "   ".join(f"{n} = {rule[n]:g}" for n in NAMES))

    section(f"Training ours: the Week 3 GRU + babble in training, seeds {SEEDS}")
    today, items, new_rows = datetime.date.today(), test_set(), []
    for seed in SEEDS:
        print(f"  seed {seed}:")
        model = MaskGRU()
        train(model, BabbleMixDataset(train_files(), seed=seed), args.steps, seed=seed)
        torch.save(model.state_dict(), OUT / f"ours_s{seed}.pt")
        means = score_split_means(lambda it, m=model: enhance(m, it["noisy"]), items)
        new_rows += rows_for(make_run_id(OURS, seed, today), "lab4_one_change.py", TEST_SET_NAME, means, seed, version,
                             "one change: 50 % of examples get babble of 3-8 other training speakers")
        print("    STOI " + "   ".join(f"{sp} {means[sp]['STOI']:.3f}" for sp in SPLITS))
    before = len(rows)
    append_rows(new_rows)
    rows = read_results()
    check("results.csv grew by exactly the new rows", lambda: len(rows) == before + len(new_rows))

    section("Table 1, built from results.csv")
    for line in write_table1(rows, RESULTS / "table1.md"):
        print("  " + line)
    plot(rows, OUT / "lab4_table1.png")
    print(f"\n  saved {RESULTS / 'table1.md'} and {OUT / 'lab4_table1.png'}")

    section("The verdict, by the rule in design.md")
    ours_b, base_b = seed_values(rows, OURS, BAB, "STOI"), seed_values(rows, BASELINE, BAB, "STOI")
    ours_p, base_p = seed_values(rows, OURS, PINK, "STOI"), seed_values(rows, BASELINE, PINK, "STOI")
    gain = mean_std(list(ours_b.values()))[0] - mean_std(list(base_b.values()))[0]
    pink_drop = mean_std(list(base_p.values()))[0] - mean_std(list(ours_p.values()))[0]
    spread = max(mean_std(list(ours_b.values()))[1], mean_std(list(base_b.values()))[1])
    print(f"  babble STOI gain   {gain:+.4f}   (you predicted {rule['predicted_babble_stoi_gain']:+.4f}; "
          f"success at >= {rule['success_babble_stoi_gain']:g}, abandon below {rule['abandon_babble_stoi_gain']:g})")
    print(f"  pink STOI change   {-pink_drop:+.4f}   (you predicted {rule['predicted_pink_stoi_change']:+.4f}; "
          f"allowed drop {rule['max_pink_stoi_drop']:g})")
    print(f"  seed-to-seed spread on babble (largest std): {spread:.4f}")
    result = judge(gain, pink_drop, spread, mean_std(list(base_b.values()))[0], rule)
    print(f"\n  VERDICT: {result}   (questions Q8, Q9)")
    check("the new training data changed the babble scores at all", lambda: abs(gain) > 0.005,
          "if ours equals the baseline, BabbleMixDataset never gives babble - check p_babble and Exercise 3.3")

    section("Listen: the same sentence, baseline vs ours")
    base = load_gru(OUT / "gru_s0.pt")
    ours = load_gru(OUT / "ours_s0.pt")
    first = items[0]["sentence"]
    for it in (i for i in items if i["sentence"] == first):
        save_wav(OUT / "compare" / f"{it['split']}_noisy.wav", it["noisy"])
        save_wav(OUT / "compare" / f"{it['split']}_baseline.wav", enhance(base, it["noisy"]))
        save_wav(OUT / "compare" / f"{it['split']}_ours.wav", enhance(ours, it["noisy"]))
    print(f"  saved out/compare/ (sentence {first}) - question Q10")
    summary()
