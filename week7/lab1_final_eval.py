"""
Week 7 - Lab 1: Final Test Set Evaluation                           run:  python lab1_final_eval.py

This is Acceptance Week (◆).
Throughout Weeks 4–6, you trained GTCRN and scored checkpoints strictly on the validation set.
Now you perform the final evaluation on the official, untouched VCTK-DEMAND test set (824 pairs).

In this lab, you:
  1. Verify the integrity of the test set against official course SHA256 fingerprints
  2. Format and execute inference and se_eval on the test set
  3. Extract the final reproduced test scores (PESQ, STOI, SI-SDR)
"""
import csv
import hashlib
from pathlib import Path

_passed = []

OFFICIAL_FINGERPRINTS = {
    "clean": "f161abab16d8391e",
    "noisy": "205263899f4f980b",
}


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
def compute_folder_fingerprint(folder_path):
    """Compute an 8-byte hex fingerprint of all .wav files in a folder.

    Algorithm (matches prepare_vctk_demand.py):
      1. Find all .wav files, sorted by filename
      2. Hash their sorted names and sizes with SHA256
      3. Return the first 16 hex characters (8 bytes)
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.2
def verify_test_fingerprints(test_dir, expected=OFFICIAL_FINGERPRINTS):
    """Verify that test/clean and test/noisy match the expected course fingerprints.

    Returns:
      (all_ok: bool, details: dict)
      details = {"clean": (computed, expected, ok), "noisy": (computed, expected, ok)}
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.3
def format_test_eval_run(checkpoint_path, out_dir, test_clean, test_noisy, name):
    """Generate commands to run inference and se_eval on the final test set."""
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.4
def load_reproduced_numbers(eval_summary_csv, model_name):
    """Read results/eval_summary.csv and extract the row for model_name on vctk-demand-test.

    Returns:
      dict with keys: 'name', 'dataset', 'pesq', 'stoi', 'sisdr'
      or None if not found.
    """
    # TODO
    raise NotImplementedError


# ================================================================ Test Suite
if __name__ == "__main__":
    import tempfile

    print("--- Testing Exercise 1.1 & 1.2: Fingerprint Verification ---")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        (td / "clean").mkdir()
        (td / "noisy").mkdir()
        (td / "clean" / "file1.wav").write_bytes(b"dummy1")
        (td / "clean" / "file2.wav").write_bytes(b"dummy2")

        try:
            fp_clean = compute_folder_fingerprint(td / "clean")
            fp_empty = compute_folder_fingerprint(td / "noisy")
        except NotImplementedError:
            fp_clean, fp_empty = "", ""

        check("computes 16-character hex fingerprint",
              lambda: len(fp_clean) == 16 and all(c in "0123456789abcdef" for c in fp_clean),
              "fingerprint should be 16 hex chars")
        check("empty directory produces different hash than populated",
              lambda: fp_clean != fp_empty,
              "different contents produce distinct hashes")

        test_expected = {"clean": fp_clean, "noisy": "9999999999999999"}
        try:
            is_ok, rep = verify_test_fingerprints(td, expected=test_expected)
        except NotImplementedError:
            is_ok, rep = False, {}

        check("detects matching clean and mismatched noisy",
              lambda: rep.get("clean", (0, 0, False))[2] is True and rep.get("noisy", (0, 0, True))[2] is False,
              "clean should match, noisy should fail")

    print("\n--- Testing Exercise 1.3: format_test_eval_run ---")
    try:
        inf_cmd, ev_cmd = format_test_eval_run("exp/best.tar", "out/test_enh", "~/data/test/clean", "~/data/test/noisy", "gtcrn-reproduced")
    except NotImplementedError:
        inf_cmd, ev_cmd = "", ""

    check("infer command specifies adapter setrain_model and test set path",
          lambda: "--adapter setrain_model" in inf_cmd and "~/data/test/noisy" in inf_cmd,
          "infer command must use setrain_model adapter")
    check("eval command targets vctk-demand-test",
          lambda: "--dataset vctk-demand-test" in ev_cmd and "--name gtcrn-reproduced" in ev_cmd,
          "eval command must target dataset vctk-demand-test")

    print("\n--- Testing Exercise 1.4: load_reproduced_numbers ---")
    with tempfile.TemporaryDirectory() as tmp:
        csv_file = Path(tmp) / "eval_summary.csv"
        csv_file.write_text(
            "name,dataset,pesq,stoi,estoi,sisdr\n"
            "gtcrn-val,vctk-demand-val,2.810,0.938,0.840,18.35\n"
            "gtcrn-reproduced,vctk-demand-test,2.868,0.941,0.847,18.81\n",
            encoding="utf-8"
        )
        try:
            row = load_reproduced_numbers(csv_file, "gtcrn-reproduced")
            missing = load_reproduced_numbers(csv_file, "non-existent")
        except NotImplementedError:
            row, missing = None, 1

        check("loads target row from eval_summary",
              lambda: row is not None and row["pesq"] == 2.868 and row["sisdr"] == 18.81,
              "expected reproduced row loaded correctly")
        check("returns None when model is not found",
              lambda: missing is None,
              "missing model should return None")

    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))
