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
    """The course fingerprint of a folder of .wav files - the same algorithm as prepare_vctk_demand.py:
      1. go through the .wav files sorted by file name
      2. for each file, feed  name + "\\t" + (the SHA256 hex digest of the file's bytes) + "\\n"  (UTF-8)
         into one running SHA256
      3. return the first 16 hex characters of that SHA256
    Return "" if the folder does not exist.
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
    """The two commands that score the chosen checkpoint on the test set, as strings:
      infer_cmd: python infer_dir.py --adapter setrain_model --checkpoint <checkpoint_path> --noisy <test_noisy> --out <out_dir>
      eval_cmd:  se_eval --enhanced <out_dir> --clean <test_clean> --name <name> --dataset vctk-demand-test

    Returns:
      (infer_cmd, eval_cmd)
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.4
def load_reproduced_numbers(eval_summary_csv, model_name):
    """From se_eval's results/eval_summary.csv, the LAST row with this name on vctk-demand-test (the file only
    grows, so the newest score is the last one). se_eval's columns are named PESQ, STOI and SI-SDR.

    Returns:
      {'name': str, 'dataset': str, 'pesq': float, 'stoi': float, 'sisdr': float}, or None if there is no such row
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
        inf_cmd, ev_cmd = format_test_eval_run("exp/run/checkpoints/model_150.tar", "out/test_enh",
                                               "~/data/test/clean", "~/data/test/noisy", "gtcrn-reproduced")
    except NotImplementedError:
        inf_cmd, ev_cmd = "", ""

    check("infer command specifies adapter setrain_model and test set path",
          lambda: "--adapter setrain_model" in inf_cmd and "~/data/test/noisy" in inf_cmd,
          "infer command must use setrain_model adapter")
    check("eval command targets vctk-demand-test",
          lambda: "--dataset vctk-demand-test" in ev_cmd and "--name gtcrn-reproduced" in ev_cmd,
          "eval command must target dataset vctk-demand-test")

    print("\n--- Testing Exercise 1.4: load_reproduced_numbers ---")
    header = ("date,name,dataset,n_files,PESQ,STOI,ESTOI,SI-SDR,DNSMOS_OVRL,DNSMOS_SIG,DNSMOS_BAK,DNSMOS_P808,"
              "pesq_failed,cut_files,max_len_diff,enhanced_dir,clean_dir,se_eval_version,note")
    example = ["2026-11-20,gtcrn-ep150,vctk-demand-val,1572,2.810,0.938,0.840,18.35" + "," * 11,
               "2026-11-27,gtcrn-reproduced,vctk-demand-test,824,2.801,0.939,0.845,18.60" + "," * 11,
               "2026-11-28,gtcrn-reproduced,vctk-demand-test,824,2.868,0.941,0.847,18.81" + "," * 11]
    with tempfile.TemporaryDirectory() as tmp:
        csv_file = Path(tmp) / "eval_summary.csv"
        csv_file.write_text(header + "\n" + "\n".join(example) + "\n", encoding="utf-8")
        try:
            row = load_reproduced_numbers(csv_file, "gtcrn-reproduced")
            missing = load_reproduced_numbers(csv_file, "non-existent")
        except NotImplementedError:
            row, missing = None, 1

        check("loads the newest test-set row (the last one) from se_eval's columns",
              lambda: row is not None and row["pesq"] == 2.868 and row["sisdr"] == 18.81,
              "se_eval's columns are PESQ, STOI, SI-SDR; the file only grows, so take the last matching row")
        check("returns None when model is not found",
              lambda: missing is None,
              "missing model should return None")

    print("\n--- Your test set ---")
    test_dir = Path.home() / "data" / "VCTK-DEMAND" / "test"
    try:
        real_ok, real = verify_test_fingerprints(test_dir)
    except NotImplementedError:
        real_ok, real = False, {}
    for side, (got, expected, _) in real.items():
        print(f"  {side}: {got or '(no folder)'}  (course: {expected})")
    check("your test set has the course fingerprints",
          lambda: real_ok,
          f"the data in {test_dir} must come from tools/se_eval/prepare_vctk_demand.py (Week 3)")

    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))
