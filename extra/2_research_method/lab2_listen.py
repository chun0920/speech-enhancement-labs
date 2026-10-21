"""
Week 4 - Lab 2: Listen before you change anything                run:  uv run lab2_listen.py
                                                                  then: uv run lab2_listen.py --tally

Lab 1 gave you numbers. Numbers tell you HOW MUCH is wrong, not WHAT is wrong. Before deciding what to
change, find out what the baseline actually gets wrong - with your ears, not with a plot.

  1. This file picks 20 of the 80 test mixtures AT RANDOM - each sentence once, 5 for each condition,
     not the ones that look interesting - and saves, for each one, the noisy input and the baseline's
     output (seed 0 from Lab 1) under an anonymous number: out/listen/07_noisy.wav, 07_output.wav, ...
     You are not told which noise or SNR each one has, so you judge what you hear, not what you expect.
  2. Listen to every pair and fill in results/error_analysis.csv, one line per number:
       category        the BIGGEST problem you hear in the output - one of CATEGORIES below
       human_can_tell  Y if YOU can follow the target sentence in the NOISY file, N if you cannot
       note            a few words; required for "other"
  3. Run with --tally. It reveals which mixture was which and counts your labels.

Why human_can_tell? If you can hear the sentence through the noise but the network cannot clean it up,
the problem is in the METHOD - that is where a research idea comes from. If even you cannot follow
the sentence, a better network will not help much: the information is not there.

Your job: fill in the TODOs, run, listen, label, run --tally.
"""
import argparse
import csv
from collections import Counter

import numpy as np

from common import (OUT, RESULTS, SPLITS, all_passed, check, enhance, evaluate, load_gru, save_wav, section, summary,
                    test_set)

N_LISTEN = 20
CATEGORIES = {
    "ok":          "clean enough - nothing you would complain about",
    "noise-left":  "a steady hiss or rumble is still clearly there",
    "voices-left": "other people's voices are still clearly there",
    "muffled":     "the target voice sounds dull, far away or underwater",
    "choppy":      "bits of words are missing, or the sound pumps on and off",
    "warble":      "chirpy, watery, 'musical' artifacts",
    "other":       "anything else - say what in the note",
}
LISTEN = OUT / "listen"
KEY = LISTEN / "key.csv"                      # which number is which mixture - only --tally reads it
LABELS = RESULTS / "error_analysis.csv"       # your labels - committed with your results


# ---------------------------------------------------------------- Exercise 2.1
def pick_random(n_items, n, seed=0):
    """n DIFFERENT indices out of range(n_items), chosen at random - a list of ints.

    rng = np.random.default_rng(seed);  rng.choice(n_items, size=n, replace=False)
    replace=False means no index can be picked twice. Return them as a list of Python ints.
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 2.2
def tally(labels):
    """labels: a list of dicts, each with a "category". Returns [(category, count, share), ...], most common first.

    counts = Counter(l["category"] for l in labels)     counts.most_common() lists (category, count), largest first
    share = count / len(labels)
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 2.3
def method_problem_share(labels):
    """Among the outputs that are NOT "ok": the share where a human could follow the noisy sentence (human_can_tell == "Y").

    Those are the failures of the method. Return 0.0 if every output is "ok".
    """
    # TODO
    raise NotImplementedError


def read_csv(path):
    with open(path, newline="") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def prepare():
    """(Given) Save the 20 anonymous pairs and an empty label sheet.

    Every test sentence once: the 20 sentences in a random order, 5 for each of the 4 conditions (so
    no condition is left out by chance, and you never hear a sentence twice), then all 20 in a random
    order (so the number tells you nothing about the condition).
    """
    model = load_gru(OUT / "gru_s0.pt")
    items = test_set()
    sentences = sorted({it["sentence"] for it in items})
    index = {(it["sentence"], it["split"]): i for i, it in enumerate(items)}
    order, per = pick_random(len(sentences), len(sentences), seed=0), N_LISTEN // len(SPLITS)
    chosen = [index[sentences[j], sp] for k, sp in enumerate(SPLITS) for j in order[k * per:(k + 1) * per]]
    picks = [chosen[j] for j in pick_random(len(chosen), len(chosen), seed=99)]
    key = []
    for k, idx in enumerate(picks, start=1):
        it, sid = items[idx], f"{k:02d}"
        out = enhance(model, it["noisy"])
        save_wav(LISTEN / f"{sid}_noisy.wav", it["noisy"])
        save_wav(LISTEN / f"{sid}_output.wav", out)
        gain = evaluate(it["clean"], out)["STOI"] - evaluate(it["clean"], it["noisy"])["STOI"]
        key.append({"sample_id": sid, "sentence": it["sentence"], "split": it["split"], "stoi_gain": f"{gain:.3f}"})
    with open(KEY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    print(f"  saved {N_LISTEN} pairs in {LISTEN}/  (NN_noisy.wav = input, NN_output.wav = baseline output)")
    if LABELS.exists():
        print(f"  {LABELS} already exists - your labels are kept, nothing was overwritten")
    else:
        with open(LABELS, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "category", "human_can_tell", "note"])
            w.writerows([[f"{k:02d}", "", "", ""] for k in range(1, N_LISTEN + 1)])
        print(f"  wrote an empty label sheet: {LABELS}")
    print("\n  Categories (write the word on the left):")
    for c, meaning in CATEGORIES.items():
        print(f"    {c:<12} {meaning}")
    print("\n  Listen to every pair, fill in the sheet, then:  uv run lab2_listen.py --tally")


def report(labels):
    """(Given) Join your labels with the key and print what you found."""
    key = {r["sample_id"]: r for r in read_csv(KEY)}
    for lab in labels:
        lab.update(noise=key[lab["sample_id"]]["split"].split("-")[0], split=key[lab["sample_id"]]["split"],
                   stoi_gain=float(key[lab["sample_id"]]["stoi_gain"]))
    section("Your labels, now with the answer key")
    print(f"  {'id':<4}{'mixture':<13}{'STOI gain':>10}   {'category':<13}{'can tell':<10}note")
    for lab in sorted(labels, key=lambda l: (l["split"], l["sample_id"])):
        print(f"  {lab['sample_id']:<4}{lab['split']:<13}{lab['stoi_gain']:>+10.3f}   {lab['category']:<13}{lab['human_can_tell']:<10}{lab['note']}")

    section("Tally (copy this table into design.md and ANSWERS.md, question Q4)")
    noises = sorted({l["noise"] for l in labels})
    print(f"  {'category':<13}{'count':>6}{'share':>8}" + "".join(f"{n:>9}" for n in noises) + "   mean STOI gain")
    for cat, count, share in tally(labels):
        in_cat = [l for l in labels if l["category"] == cat]
        per_noise = "".join(f"{sum(l['noise'] == n for l in in_cat):>9}" for n in noises)
        print(f"  {cat:<13}{count:>6}{100 * share:>7.0f}%{per_noise}   {np.mean([l['stoi_gain'] for l in in_cat]):+.3f}")
    top, top_count, top_share = tally(labels)[0]
    print(f"\n  largest category: {top} ({top_count} of {len(labels)}, {100 * top_share:.0f} %)")
    print(f"  not 'ok' but a human could follow the noisy sentence: {100 * method_problem_share(labels):.0f} % of the failures")
    for n in noises:
        sub = [l for l in labels if l["noise"] == n]
        print(f"    {n:<7} {100 * method_problem_share(sub):3.0f} %  ({sum(l['category'] != 'ok' for l in sub)} failures of {len(sub)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tally", action="store_true", help="read your labels and count them")
    args = ap.parse_args()

    section("Exercise 2.1  a random sample")
    try:
        a, b, c = pick_random(80, 20, 0), pick_random(80, 20, 0), pick_random(80, 20, 1)
    except NotImplementedError:
        check("pick_random() is written", lambda: False, "finish the TODO in Exercise 2.1")
    else:
        check("20 different indices between 0 and 79", lambda: len(set(a)) == 20 and all(0 <= i < 80 for i in a),
              "rng.choice(n_items, size=n, replace=False)")
        check("plain Python ints", lambda: all(type(i) is int for i in a), "[int(i) for i in ...]")
        check("the same seed gives the same sample, another seed another one", lambda: a == b and a != c)
        check("random, not simply the first 20", lambda: sorted(a) != list(range(20)))

    section("Exercise 2.2 and 2.3  counting labels")
    fake = [{"category": c, "human_can_tell": h} for c, h in
            [("voices-left", "Y"), ("ok", "Y"), ("muffled", "N"), ("voices-left", "Y"), ("voices-left", "N"), ("muffled", "Y")]]
    check("tally: most common first, with shares",
          lambda: tally(fake) == [("voices-left", 3, 0.5), ("muffled", 2, 2 / 6), ("ok", 1, 1 / 6)],
          "Counter(...).most_common(), share = count / len(labels)")
    check("method_problem_share: 3 of the 5 failures could be followed -> 0.6",
          lambda: abs(method_problem_share(fake) - 0.6) < 1e-9, 'leave out the "ok" ones first')
    check("method_problem_share is 0.0 when everything is ok",
          lambda: method_problem_share([{"category": "ok", "human_can_tell": "Y"}]) == 0.0)
    if not all_passed():
        summary()
        raise SystemExit(1)

    if not args.tally:
        section("Preparing the listening test")
        prepare()
        summary()
        raise SystemExit(0)

    section("Reading your labels")
    if not LABELS.exists() or not KEY.exists():
        raise SystemExit("run  uv run lab2_listen.py  first (without --tally)")
    labels = read_csv(LABELS)
    for lab in labels:                                   # accept "Muffled" and "y" too
        lab["category"], lab["human_can_tell"] = lab["category"].lower(), lab["human_can_tell"].upper()
    missing = [l["sample_id"] for l in labels if not l["category"] or not l["human_can_tell"]]
    unknown = [f"{l['sample_id']}: {l['category']}" for l in labels if l["category"] and l["category"] not in CATEGORIES]
    bad_yn = [l["sample_id"] for l in labels if l["human_can_tell"] and l["human_can_tell"] not in ("Y", "N")]
    no_note = [l["sample_id"] for l in labels if l["category"] == "other" and not l["note"]]
    check(f"all {N_LISTEN} samples are labelled", lambda: len(labels) == N_LISTEN and not missing,
          f"still empty: {', '.join(missing)}")
    check("every category is one of the list", lambda: not unknown, f"unknown: {'; '.join(unknown)}  (spelling as in CATEGORIES)")
    check("human_can_tell is Y or N", lambda: not bad_yn, f"check sample(s) {', '.join(bad_yn)}")
    check('every "other" has a note', lambda: not no_note, f"add a note to {', '.join(no_note)}")
    if all_passed():
        report(labels)
    summary()
