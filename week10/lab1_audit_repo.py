"""
Week 10 - Lab 1: Automated Repository Hygiene & Deliverables Audit   run:  python lab1_audit_repo.py

In academic research and engineering teams, repository hygiene is just as important as model code.
A messy repository with uncommitted secrets, giant binary files, missing .gitignore rules,
or unorganized deliverables causes reproducibility failures and wastes team storage.

In this final lab, you:
  1. Implement an automated repository linter that detects oversized files (> 10 MB)
  2. Verify that .gitignore includes critical patterns for checkpoints, caches, and environments
  3. Audit the presence and completeness of all 10-week milestones and reports
  4. Generate a comprehensive audit score card in results/repo_audit_report.md
"""
import os
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


# ---------------------------------------------------------------- Exercise 1.1
def audit_file_sizes(root_dir, max_mb=10.0):
    """Scan directory recursively for files exceeding `max_mb`.

    Ignores the `.git` directory.

    Args:
      root_dir: path to directory to scan
      max_mb: maximum allowed file size in Megabytes (default 10.0)

    Returns:
      list of (rel_path: str, size_mb: float) for files exceeding max_mb
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.2
def audit_gitignore_rules(gitignore_path, required_keywords=("*.tar", "__pycache__", ".venv")):
    """Check that .gitignore exists and contains essential hygiene patterns.

    Args:
      gitignore_path: path to .gitignore file
      required_keywords: tuple of required keywords or patterns

    Returns:
      (is_valid: bool, missing_keywords: list)
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.3
def audit_course_deliverables(root_dir, required_items):
    """Check that required deliverables exist across the 10-week curriculum.

    Args:
      root_dir: student repository root path
      required_items: list of relative paths to check

    Returns:
      (completion_rate: float, found_items: list, missing_items: list)
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.4
def generate_audit_report(size_violations, gitignore_status, deliverables_status, out_path):
    """Format audit results into results/repo_audit_report.md."""
    # TODO
    raise NotImplementedError


# ================================================================ Test Suite
if __name__ == "__main__":
    import tempfile

    print("--- Testing Exercise 1.1: audit_file_sizes ---")
    with tempfile.TemporaryDirectory() as tmp:
        p_small = Path(tmp) / "small.txt"
        p_small.write_text("hello" * 100, encoding="utf-8")
        p_big = Path(tmp) / "big.bin"
        p_big.write_bytes(b"\x00" * 2048)  # 2 KB

        try:
            # Test with 1 KB threshold
            viols = audit_file_sizes(tmp, max_mb=0.001)  # ~1 KB
        except NotImplementedError:
            viols = []

        check("identifies files exceeding size threshold",
              lambda: len(viols) == 1 and viols[0][0] == "big.bin",
              "expected big.bin to be flagged as violation")

    print("\n--- Testing Exercise 1.2: audit_gitignore_rules ---")
    with tempfile.TemporaryDirectory() as tmp:
        gi = Path(tmp) / ".gitignore"
        gi.write_text("*.tar\n__pycache__/\n.venv/\n", encoding="utf-8")
        try:
            valid, miss = audit_gitignore_rules(gi, required_keywords=("*.tar", "__pycache__", ".venv"))
        except NotImplementedError:
            valid, miss = False, ["not implemented"]

        check("validates clean .gitignore containing required keywords",
              lambda: valid and len(miss) == 0,
              "expected valid=True and empty missing list")

        # Test missing keyword
        gi_bad = Path(tmp) / "bad.gitignore"
        gi_bad.write_text("*.log\n", encoding="utf-8")
        try:
            bad_valid, bad_miss = audit_gitignore_rules(gi_bad, required_keywords=("*.tar", ".venv"))
        except NotImplementedError:
            bad_valid, bad_miss = True, []

        check("detects missing keywords in deficient .gitignore",
              lambda: not bad_valid and len(bad_miss) == 2,
              "expected bad_valid=False with 2 missing keywords")

    print("\n--- Testing Exercise 1.3: audit_course_deliverables ---")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "report1.md").write_text("# R1", encoding="utf-8")
        (Path(tmp) / "report2.md").write_text("# R2", encoding="utf-8")
        reqs = ["report1.md", "report2.md", "report3.md"]

        try:
            rate, fnd, mss = audit_course_deliverables(tmp, reqs)
        except NotImplementedError:
            rate, fnd, mss = 0.0, [], reqs

        check("computes deliverable completion rate accurately (2/3 = 66.7%)",
              lambda: abs(rate - (2.0 / 3.0)) < 0.01 and "report3.md" in mss,
              "expected 66.7% completion with report3.md missing")

    print("\n--- Testing Exercise 1.4: generate_audit_report ---")
    with tempfile.TemporaryDirectory() as tmp:
        out_f = Path(tmp) / "results" / "repo_audit_report.md"
        try:
            rep = generate_audit_report([], (True, []), (1.0, ["rep1.md"], []), out_f)
            has_rep = out_f.exists() and "Ready for Acceptance" in out_f.read_text(encoding="utf-8")
        except NotImplementedError:
            rep, has_rep = "", False

        check("writes audit markdown report with PASS status",
              lambda: has_rep and "Final Repository Hygiene" in rep,
              "expected markdown report with PASS status")

    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))
