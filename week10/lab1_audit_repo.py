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
import sys
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
    """Check which of the required deliverables exist under root_dir.

    Args:
      root_dir: your group's repository folder
      required_items: list of paths relative to root_dir

    Returns:
      (completion_rate, found_items, missing_items): completion_rate is the fraction found, from 0.0 to 1.0
      (2 of 3 found -> 0.667); found_items and missing_items keep the order of required_items
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.4
def generate_audit_report(size_violations, gitignore_status, deliverables_status, out_path):
    """Write the audit as Markdown to out_path and return its text.

    size_violations: from audit_file_sizes(); gitignore_status: (is_valid, missing) from audit_gitignore_rules();
    deliverables_status: (rate, found, missing) from audit_course_deliverables().
    The report starts with the title "# Final Repository Hygiene Audit", lists the files over 10 MB, the
    missing .gitignore patterns and the missing deliverables, and ends with the verdict
    "PASS (Ready for Acceptance)" when there is nothing to list, otherwise "NOT READY" and what to fix.
    """
    # TODO
    raise NotImplementedError


# ================================================================ Test Suite
REQUIRED = ["README.md", ".gitignore", "configs/cfg_train_vctk_full.yaml",
            "results/week3_table.md", "results/training_curves.csv", "results/reproduction_report.md",
            "results/efficiency_table.md", "results/stress_test_report.md", "results/comparison_table.md",
            "demo/index.html"]


if __name__ == "__main__":
    import tempfile

    print("--- Testing Exercise 1.1: audit_file_sizes ---")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "small.txt").write_text("hello" * 100, encoding="utf-8")
        (Path(tmp) / "big.bin").write_bytes(b"\x00" * 2048)  # 2 KB
        try:
            viols = audit_file_sizes(tmp, max_mb=0.001)        # about 1 KB
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
        try:
            rate, fnd, mss = audit_course_deliverables(tmp, ["report1.md", "report2.md", "report3.md"])
        except NotImplementedError:
            rate, fnd, mss = 0.0, [], ["report1.md", "report2.md", "report3.md"]

        check("computes deliverable completion rate accurately (2 of 3 = 0.667)",
              lambda: abs(rate - (2.0 / 3.0)) < 0.01 and "report3.md" in mss,
              "a fraction from 0.0 to 1.0, with report3.md missing")

    print("\n--- Testing Exercise 1.4: generate_audit_report ---")
    with tempfile.TemporaryDirectory() as tmp:
        out_f = Path(tmp) / "results" / "repo_audit_report.md"
        try:
            rep = generate_audit_report([], (True, []), (1.0, ["rep1.md"], []), out_f)
            has_rep = out_f.exists() and "PASS (Ready for Acceptance)" in out_f.read_text(encoding="utf-8")
            rep_bad = generate_audit_report([("ckpt.tar", 25.0)], (True, []), (0.5, ["a"], ["b"]),
                                            Path(tmp) / "bad.md")
        except NotImplementedError:
            rep, has_rep, rep_bad = "", False, ""

        check("writes the report: title, and PASS only when nothing is missing",
              lambda: has_rep and "Final Repository Hygiene Audit" in rep
              and "NOT READY" in rep_bad and "ckpt.tar" in rep_bad,
              "title '# Final Repository Hygiene Audit'; last line PASS (Ready for Acceptance) or NOT READY")

    if len(sys.argv) > 1:                        # python lab1_audit_repo.py <your group's repository>
        repo = Path(sys.argv[1]).expanduser()
        print(f"\n--- Your repository: {repo} ---")
        try:
            report = generate_audit_report(audit_file_sizes(repo), audit_gitignore_rules(repo / ".gitignore"),
                                           audit_course_deliverables(repo, REQUIRED),
                                           Path("results") / "repo_audit_report.md")
            print(report.splitlines()[-1])
            print("  saved results/repo_audit_report.md")
        except NotImplementedError:
            print("  finish the exercises first")
    else:
        print("\n  Your group's repository: python lab1_audit_repo.py <its folder>")

    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))
