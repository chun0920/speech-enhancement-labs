"""
Week 7 - Lab 2: Scientific Reproduction Check & Gap Analysis       run:  python lab2_reproduce_check.py

This is the Acceptance Week (◆) verification script.
Prof. Hung checks whether your GTCRN training reproduced the published results.

Reproduction Criteria (from Elective 2 / Lab-Hub):
  - Relative gap <= 5.0%:  SUCCESS (reproduction verified)
  - Relative gap 5% - 10%: ACCEPTABLE WITH EXPLANATION (explain minor discrepancy)
  - Relative gap > 10.0%:  NEEDS INVESTIGATION (systematic error, e.g. STFT window mismatch)

In this lab, you:
  1. Load official paper numbers from paper_numbers.csv (GTCRN Table 1: PESQ 2.87, STOI 0.940, SI-SDR 18.83)
  2. Compute relative percentage gaps on PESQ, STOI, and SI-SDR
  3. Classify the reproduction result against the scientific criteria
  4. Generate results/reproduction_report.md for acceptance review
"""
import csv
from pathlib import Path

_passed = []


def check(name, fn, hint=""):
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


# ---------------------------------------------------------------- Exercise 2.1
def load_paper_numbers(paper_csv_path):
    """Load baseline reference numbers from paper_numbers.csv.

    Returns:
      dict keyed by system name:
        {
          'noisy': {'pesq': float, 'stoi': float, 'sisdr': float, 'source': str},
          'gtcrn-paper': {'pesq': float, 'stoi': float, 'sisdr': float, 'source': str}
        }
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 2.2
def compute_relative_gap(reproduced_val, paper_val):
    """Compute the relative percentage gap:
         gap_pct = abs(reproduced_val - paper_val) / abs(paper_val) * 100.0
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 2.3
def classify_reproduction(pesq_gap, stoi_gap, sisdr_gap):
    """Classify reproduction fidelity based on the maximum metric gap:
         - max_gap <= 5.0:  'SUCCESS'
         - max_gap <= 10.0: 'ACCEPTABLE_WITH_EXPLANATION'
         - max_gap > 10.0:  'NEEDS_INVESTIGATION'

    Returns:
      (status: str, max_gap: float)
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 2.4
def generate_reproduction_report(reproduced, paper, out_path):
    """Generate a markdown report comparing reproduced metrics with paper Table 1."""
    # TODO
    raise NotImplementedError


# ================================================================ Test Suite
if __name__ == "__main__":
    import tempfile

    print("--- Testing Exercise 2.1: load_paper_numbers ---")
    with tempfile.TemporaryDirectory() as tmp:
        csv_file = Path(tmp) / "paper_numbers.csv"
        csv_file.write_text(
            "system,dataset,pesq,stoi,sisdr,source\n"
            "noisy,vctk-demand-test,1.97,0.921,8.45,GTCRN Table 1\n"
            "gtcrn-paper,vctk-demand-test,2.87,0.940,18.83,GTCRN Table 1\n",
            encoding="utf-8"
        )
        try:
            papers = load_paper_numbers(csv_file)
        except NotImplementedError:
            papers = {}

        check("loads both noisy and gtcrn-paper entries",
              lambda: "noisy" in papers and "gtcrn-paper" in papers,
              "expected noisy and gtcrn-paper loaded")
        check("parses numerical metrics correctly",
              lambda: papers["gtcrn-paper"]["pesq"] == 2.87 and papers["gtcrn-paper"]["sisdr"] == 18.83,
              "expected PESQ 2.87 and SI-SDR 18.83")

    print("\n--- Testing Exercise 2.2: compute_relative_gap ---")
    try:
        g1 = compute_relative_gap(2.867, 2.87)
        g2 = compute_relative_gap(18.0, 20.0)
    except NotImplementedError:
        g1, g2 = 999.0, 999.0

    check("gap between 2.867 and 2.87 is ~0.10%",
          lambda: abs(g1 - 0.1045) < 0.01,
          "expected gap around 0.10%")
    check("gap between 18.0 and 20.0 is exactly 10.0%",
          lambda: abs(g2 - 10.0) < 1e-4,
          "expected gap 10.0%")

    print("\n--- Testing Exercise 2.3: classify_reproduction ---")
    try:
        st_succ, _ = classify_reproduction(0.5, 0.2, 0.8)
        st_expl, _ = classify_reproduction(7.2, 1.0, 2.0)
        st_inve, _ = classify_reproduction(12.5, 3.0, 4.0)
    except NotImplementedError:
        st_succ, st_expl, st_inve = "", "", ""

    check("classifies <= 5% as SUCCESS",
          lambda: st_succ == "SUCCESS",
          "all gaps <= 5% is SUCCESS")
    check("classifies 5-10% as ACCEPTABLE_WITH_EXPLANATION",
          lambda: st_expl == "ACCEPTABLE_WITH_EXPLANATION",
          "max gap 7.2% is ACCEPTABLE_WITH_EXPLANATION")
    check("classifies > 10% as NEEDS_INVESTIGATION",
          lambda: st_inve == "NEEDS_INVESTIGATION",
          "max gap 12.5% is NEEDS_INVESTIGATION")

    print("\n--- Testing Exercise 2.4: generate_reproduction_report ---")
    with tempfile.TemporaryDirectory() as tmp:
        rep_file = Path(tmp) / "results" / "reproduction_report.md"
        sample_reproduced = {"name": "ours", "pesq": 2.867, "stoi": 0.940, "sisdr": 18.80}
        sample_paper = {"pesq": 2.87, "stoi": 0.940, "sisdr": 18.83}

        try:
            stat, max_g = generate_reproduction_report(sample_reproduced, sample_paper, rep_file)
            has_report = rep_file.exists() and "SUCCESS" in rep_file.read_text(encoding="utf-8")
        except NotImplementedError:
            stat, has_report = "", False

        check("generates markdown report with SUCCESS status",
              lambda: stat == "SUCCESS" and has_report,
              "expected reproduction_report.md created with SUCCESS")

    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))
