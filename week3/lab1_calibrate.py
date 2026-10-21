"""
Week 3 - Lab 1: Calibrate the judge before you judge                   run:  python lab1_calibrate.py

Before scoring GTCRN, check the scoring itself. The noisy test files need no model, and every paper
reports their scores (the "Noisy" row). If se_eval and your data are right, your noisy row must match
the paper's almost exactly - there is no training and no randomness in it. If it does not match, every
number you produce this week is off by an unknown amount, and you would blame the model for a problem
in the data or in the metric.

The second check: recompute SI-SDR for a few files with YOUR OWN code (the Week 2 formula) and compare
with se_eval's per-file scores. A script that both groups trust should have been checked by someone.

Before running this file (README, Steps 3-4):
  1. fill in the "noisy" row of paper_numbers.csv: the numbers, and in `source` the table and page
  2. score the noisy test files with se_eval, with --name noisy
Then:  python lab1_calibrate.py                    (VCTK-DEMAND)
       python lab1_calibrate.py --dataset dns3-blind-track1   (if your group uses the DNS3 blind test set)
If the noisy row still does not match after you checked the data, the sample rate and the numbers you copied,
find the cause (README, Step 4) and run again with --explained "what you checked and what you found".
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf

RESULTS = Path("results")
METRICS = ("PESQ", "STOI", "ESTOI", "SI-SDR", "DNSMOS_OVRL", "DNSMOS_SIG", "DNSMOS_BAK", "DNSMOS_P808")
_passed = []


def check(name, fn, hint=""):
    """Run fn() and print PASS if it returns True, otherwise FAIL plus a hint."""
    try:
        ok = bool(fn())
    except NotImplementedError:
        ok, hint = False, "not written yet - look for the # TODO"
    except Exception as e:
        ok, hint = False, f"{type(e).__name__}: {e}  {hint}".strip()
    _passed.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok and hint:
        print(f"         hint: {hint}")
    return ok


def summary():
    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- Exercise 1.1
def read_latest(rows, name, dataset):
    """From the rows of eval_summary.csv (or paper_numbers.csv), the LAST row whose "name" and "dataset"
    match. Return {metric: float} for every metric in METRICS that has a number in that row (skip empty
    cells). Return None if no row matches.

    eval_summary.csv only grows: if you scored "noisy" twice, the newest row is the last one.
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.2
def relative_gap(mine, paper):
    """|mine - paper| / |paper|, e.g. relative_gap(1.95, 1.97) = 0.0102 (1.02 %)."""
    # TODO
    raise NotImplementedError


def calibration_verdict(gap):
    """The noisy row involves no model, so the tolerance is tight:
        gap <= 0.01   -> "matches"
        gap <= 0.03   -> "close: find out why"     (e.g. a different 16 kHz version of the data)
        otherwise     -> "does NOT match: stop and find the cause"
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.3
def my_si_sdr(ref, est):
    """SI-SDR in dB, your Week 2 formula, with both signals made zero-mean first:
        ref, est = ref - mean(ref), est - mean(est)      (use float64)
        target   = (est . ref) / (ref . ref) * ref
        SI-SDR   = 10 * log10( |target|^2 / |est - target|^2 )
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="vctk-demand-test", help="the --dataset you gave se_eval")
    ap.add_argument("--data", default=str(Path.home() / "data" / "VCTK-DEMAND" / "test"),
                    help="folder with clean/ and noisy/ (for the SI-SDR cross-check)")
    ap.add_argument("--explained", default="",
                    help="only if the noisy row still does not match after you checked everything: what you checked and found")
    args = ap.parse_args()

    print("--- Exercise 1.1 and 1.2 ---")
    rows_ex = [{"name": "noisy", "dataset": "d", "PESQ": "1.9", "STOI": ""},
               {"name": "noisy", "dataset": "d", "PESQ": "1.97", "STOI": "0.921"}]
    check("read_latest takes the LAST matching row and skips empty cells",
          lambda: read_latest(rows_ex, "noisy", "d") == {"PESQ": 1.97, "STOI": 0.921}, "keep overwriting `found` in the loop")
    check("read_latest returns None when nothing matches", lambda: read_latest(rows_ex, "gtcrn", "d") is None)
    check("relative_gap(1.95, 1.97) = 0.0102", lambda: abs(relative_gap(1.95, 1.97) - 0.02 / 1.97) < 1e-9)
    check("0.5 % matches, 2 % is close, 5 % does not match",
          lambda: calibration_verdict(0.005) == "matches" and calibration_verdict(0.02).startswith("close")
          and calibration_verdict(0.05).startswith("does NOT"))
    rng = np.random.default_rng(0)
    s, n = rng.standard_normal(16000), rng.standard_normal(16000)
    check("my_si_sdr: 0.5 x the reference (plus a trace of noise) is a near-perfect estimate (> 100 dB)",
          lambda: my_si_sdr(s, 0.5 * s + 1e-6 * n) > 100,
          "scale invariant: the target is rescaled to the estimate")
    check("my_si_sdr: s + noise of equal power is about 0 dB", lambda: abs(my_si_sdr(s, s + n)) < 0.2)
    if not all(_passed):                     # the next part uses your functions: finish them first
        summary()
        raise SystemExit(1)

    print("\n--- Calibration: your noisy row against the paper's ---")
    summary_csv, paper_csv = RESULTS / "eval_summary.csv", Path("paper_numbers.csv")
    if not summary_csv.exists():
        raise SystemExit("results/eval_summary.csv not found - score the noisy files with se_eval first (README, Step 4)")
    mine = read_latest(read_csv(summary_csv), "noisy", args.dataset)
    paper = read_latest(read_csv(paper_csv), "noisy", args.dataset)
    check(f'se_eval has a "noisy" row for {args.dataset}', lambda: mine is not None, "se_eval ... --name noisy --dataset <the same name>")
    check('paper_numbers.csv has the paper\'s "noisy" numbers', lambda: bool(paper), "fill in the noisy row of paper_numbers.csv")
    lines = []
    if mine and paper:
        print(f"  {'metric':<12}{'yours':>9}{'paper':>9}{'gap':>8}   verdict")
        for m in METRICS:
            if m in mine and m in paper:
                gap = relative_gap(mine[m], paper[m])
                lines.append(f"| {m} | {mine[m]:.3f} | {paper[m]:.3f} | {100 * gap:.2f} % | {calibration_verdict(gap)} |")
                print(f"  {m:<12}{mine[m]:>9.3f}{paper[m]:>9.3f}{100 * gap:>7.2f}%   {calibration_verdict(gap)}")
        compared = [m for m in METRICS if m in mine and m in paper]
        check("at least two metrics compared", lambda: len(compared) >= 2, "the paper's noisy row must have numbers for the metrics se_eval computed")
        all_match = all(calibration_verdict(relative_gap(mine[m], paper[m])) == "matches" for m in compared)
        if all_match or not args.explained:
            check("every compared metric matches the paper (within 1 %)", lambda: all_match,
                  "check the data (SOURCE.txt), the sample rate, and that you copied the paper's numbers right; "
                  "if all of that is right, find the cause and run again with --explained (README, Step 4)")
        else:
            check("not every metric matches, and you wrote down what you checked and found (--explained)",
                  lambda: len(args.explained.split()) >= 8, "one or two full sentences: what you checked, what you found")

    print("\n--- Cross-check: your own SI-SDR against se_eval's, 3 files ---")
    per_file = RESULTS / "noisy_per_file.csv"
    data = Path(args.data).expanduser()
    if per_file.exists() and "SI-SDR" in read_csv(per_file)[0] and (data / "clean").exists():
        diffs = []
        for r in read_csv(per_file)[:3]:
            ref, _ = sf.read(data / "clean" / r["file"], dtype="float32")
            est, _ = sf.read(data / "noisy" / r["file"], dtype="float32")
            ours = my_si_sdr(ref, est)
            diffs.append(abs(ours - float(r["SI-SDR"])))
            print(f"  {r['file']}: yours {ours:.4f} dB, se_eval {float(r['SI-SDR']):.4f} dB")
        check("your SI-SDR equals se_eval's (within 0.001 dB)", lambda: max(diffs) < 1e-3,
              "both must remove the mean first and use the clean file as the reference")
    else:
        print("  skipped: no clean references (blind test set) or no per-file scores yet")

    if lines:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "calibration.md").write_text(
            f"Calibration on {args.dataset}: the noisy (unprocessed) row, ours vs. the paper\n\n"
            "| metric | ours | paper | gap | verdict |\n|---|---|---|---|---|\n" + "\n".join(lines) + "\n"
            + (f"\nNot every metric matches. What was checked and found: {args.explained}\n" if args.explained else ""))
        print("\n  saved results/calibration.md")
    summary()
