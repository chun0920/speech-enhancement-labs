"""
Week 6 - Lab 1: Milestone Checkpoint Evaluator                      run:  python lab1_checkpoint_eval.py

During the 200-epoch training schedule, checkpoints are saved every 10 epochs (model_010.tar,
model_020.tar, ...) alongside best_model_<epoch>.tar.

Before deciding which checkpoint to take to Week 7's final test set evaluation, you must
evaluate training milestones on the VALIDATION set (val/):
  1. Detect and index all available saved checkpoints in call order
  2. Format automated inference and se_eval evaluation commands
  3. Parse results/eval_summary.csv to track metric progression (PESQ, STOI, SI-SDR)
  4. Implement checkpoint selection criteria (validation peak vs final epoch)
"""
import csv
import re
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
def find_available_checkpoints(checkpoint_dir):
    """Find all saved model checkpoints in checkpoint_dir.

    Matches:
      - 'model_010.tar', 'model_020.tar', etc.
      - 'best_model_050.tar', etc. (listed so you see it; never evaluate it - it holds the last epoch, Week 5)

    Returns:
      sorted list of tuples: (epoch: int, is_best: bool, path: Path)
      sorted by epoch ascending, with regular checkpoints before best if same epoch.
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.2
def format_eval_command(checkpoint_path, out_dir, clean_dir, noisy_dir, name, dataset="vctk-demand-val"):
    """The two commands that score one checkpoint on the validation set, as strings:
      infer_cmd: python infer_dir.py --adapter setrain_model --checkpoint <checkpoint_path> --noisy <noisy_dir> --out <out_dir>
      eval_cmd:  se_eval --enhanced <out_dir> --clean <clean_dir> --name <name> --dataset <dataset>

    Returns:
      (infer_cmd, eval_cmd)
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.3
def parse_eval_summary(csv_path):
    """Read se_eval's results/eval_summary.csv and return its rows as a list of dicts, in file order.

    se_eval names its columns PESQ, STOI, ESTOI and SI-SDR. For every row that has a PESQ value:
      {"name": str, "dataset": str, "pesq": float, "stoi": float, "estoi": float, "sisdr": float}
    Skip rows without PESQ (DNSMOS-only rows). Return [] if the file does not exist.
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.4
def select_best_checkpoint(evaluated_records, dataset="vctk-demand-val"):
    """Pick the best checkpoint from the records of parse_eval_summary(), using ONLY the rows of `dataset`
    (the validation set - never the test set) and skipping the row named "noisy" (no model).
    Best = highest PESQ; on a tie, the higher SI-SDR.

    Returns:
      (best_record, rationale): rationale is one sentence with the record's name and its PESQ with 3 decimals,
      e.g. "gtcrn-ep150: highest validation PESQ 2.810 (SI-SDR 18.35 dB)".
      If no record qualifies: (None, "no checkpoints evaluated")
    """
    # TODO
    raise NotImplementedError


# ================================================================ Test Suite
if __name__ == "__main__":
    import tempfile

    print("--- Testing Exercise 1.1: find_available_checkpoints ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for fname in ("model_010.tar", "model_050.tar", "model_020.tar", "best_model_050.tar", "ignore_me.txt"):
            (tmp_dir / fname).touch()
        try:
            cps = find_available_checkpoints(tmp_dir)
        except NotImplementedError:
            cps = []

        check("finds all 4 checkpoints",
              lambda: len(cps) == 4,
              "expected 4 .tar checkpoints (the .txt file is not one)")
        check("sorts by epoch in ascending order",
              lambda: [c[0] for c in cps] == [10, 20, 50, 50],
              "epochs must be sorted 10, 20, 50, 50")
        check("flags best_model correctly, after the regular checkpoint of the same epoch",
              lambda: cps[2][1] is False and cps[3][1] is True,
              "model_050.tar first (is_best=False), then best_model_050.tar (is_best=True)")

    print("\n--- Testing Exercise 1.2: format_eval_command ---")
    try:
        inf_cmd, ev_cmd = format_eval_command(
            "checkpoints/model_050.tar", "out/val_050",
            "~/data/val/clean", "~/data/val/noisy", "gtcrn-ep050"
        )
    except NotImplementedError:
        inf_cmd, ev_cmd = "", ""

    check("infer command specifies adapter setrain_model",
          lambda: "--adapter setrain_model" in inf_cmd and "--checkpoint checkpoints/model_050.tar" in inf_cmd,
          "infer_dir requires --adapter setrain_model")
    check("eval command specifies name and dataset",
          lambda: "se_eval" in ev_cmd and "--name gtcrn-ep050" in ev_cmd and "--dataset vctk-demand-val" in ev_cmd,
          "se_eval command must include --name and --dataset")

    print("\n--- Testing Exercise 1.3: parse_eval_summary ---")
    header = ("date,name,dataset,n_files,PESQ,STOI,ESTOI,SI-SDR,DNSMOS_OVRL,DNSMOS_SIG,DNSMOS_BAK,DNSMOS_P808,"
              "pesq_failed,cut_files,max_len_diff,enhanced_dir,clean_dir,se_eval_version,note")

    def row(name, dataset, n, pesq="", stoi="", estoi="", sisdr="", ovrl=""):     # one line in se_eval's format
        return ",".join(["2026-11-20", name, dataset, n, pesq, stoi, estoi, sisdr, ovrl] + [""] * 10)

    example = [row("noisy", "vctk-demand-val", "1572", "1.473", "0.839", "0.643", "6.23"),
               row("gtcrn-ep010", "vctk-demand-val", "1572", "2.201", "0.880", "0.730", "14.50"),
               row("gtcrn-ep050", "vctk-demand-val", "1572", "2.650", "0.925", "0.810", "17.20"),
               row("gtcrn-ep100", "vctk-demand-val", "1572", "2.782", "0.935", "0.835", "18.10"),
               row("gtcrn-ep150", "vctk-demand-val", "1572", "2.810", "0.938", "0.840", "18.35"),
               row("gtcrn-ep150", "vctk-demand-test", "824", "2.900", "0.941", "0.847", "18.81"),
               row("noisy", "dns3-blind-track1", "700", ovrl="2.347")]
    with tempfile.TemporaryDirectory() as tmp:
        csv_file = Path(tmp) / "eval_summary.csv"
        csv_file.write_text(header + "\n" + "\n".join(example) + "\n", encoding="utf-8")
        try:
            records = parse_eval_summary(csv_file)
        except NotImplementedError:
            records = []

        check("parses the 6 rows that have PESQ (skips the DNSMOS-only row)",
              lambda: len(records) == 6,
              "se_eval's columns are PESQ, STOI, ESTOI, SI-SDR; skip rows whose PESQ is empty")
        check("converts se_eval's metric columns to float",
              lambda: records[2]["pesq"] == 2.650 and records[2]["sisdr"] == 17.20,
              "keys pesq / stoi / estoi / sisdr, values float")

    print("\n--- Testing Exercise 1.4: select_best_checkpoint ---")
    try:
        best_entry, why = select_best_checkpoint(records)
    except NotImplementedError:
        best_entry, why = None, ""

    check("picks the best checkpoint on the validation set (not the higher test-set row)",
          lambda: best_entry is not None and best_entry["name"] == "gtcrn-ep150"
          and best_entry["dataset"] == "vctk-demand-val" and best_entry["pesq"] == 2.810,
          "only rows of the validation set count; the test set is never used to choose")
    check("gives the name and its PESQ in the rationale",
          lambda: "gtcrn-ep150" in why and "2.810" in why,
          "e.g. gtcrn-ep150: highest validation PESQ 2.810 (SI-SDR 18.35 dB)")

    real = Path("results") / "eval_summary.csv"
    if real.exists():
        try:
            print(f"\n  Your rows in {real}: {select_best_checkpoint(parse_eval_summary(real))[1]}")
        except NotImplementedError:
            pass

    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))
