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
      - 'best_model_085.tar', etc.

    Returns:
      sorted list of tuples: (epoch: int, is_best: bool, path: Path)
      sorted by epoch ascending, with regular checkpoints before best if same epoch.
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.2
def format_eval_command(checkpoint_path, out_dir, clean_dir, noisy_dir, name, dataset="vctk-demand-val"):
    """Generate command-line strings for running infer_dir and se_eval.

    Returns a tuple of two command strings:
      (infer_cmd, eval_cmd)
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.3
def parse_eval_summary(csv_path):
    """Read results/eval_summary.csv and return rows as a list of dicts.

    Each dict contains:
      {"name": str, "dataset": str, "pesq": float, "stoi": float, "estoi": float, "sisdr": float}
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.4
def select_best_checkpoint(evaluated_records):
    """Select the best checkpoint from evaluated validation records based on peak PESQ.

    Args:
      evaluated_records: list of dicts from parse_eval_summary()

    Returns:
      (best_record: dict, rationale: str)
      If empty, returns (None, "no checkpoints evaluated")
    """
    # TODO
    raise NotImplementedError


# ================================================================ Test Suite
if __name__ == "__main__":
    import tempfile

    print("--- Testing Exercise 1.1: find_available_checkpoints ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / "model_010.tar").touch()
        (tmp_dir / "model_050.tar").touch()
        (tmp_dir / "model_020.tar").touch()
        (tmp_dir / "best_model_048.tar").touch()
        (tmp_dir / "ignore_me.txt").touch()

        try:
            cps = find_available_checkpoints(tmp_dir)
        except NotImplementedError:
            cps = []

        check("finds all 4 checkpoints",
              lambda: len(cps) == 4,
              "expected 4 tar checkpoints detected")
        check("sorts by epoch in ascending order",
              lambda: [c[0] for c in cps] == [10, 20, 48, 50],
              "epochs must be sorted 10, 20, 48, 50")
        check("flags best_model correctly",
              lambda: cps[2][1] is True and cps[0][1] is False,
              "best_model_048 must have is_best=True")

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
    with tempfile.TemporaryDirectory() as tmp:
        csv_file = Path(tmp) / "eval_summary.csv"
        csv_file.write_text(
            "name,dataset,pesq,stoi,estoi,sisdr\n"
            "gtcrn-ep010,vctk-demand-val,2.201,0.880,0.730,14.50\n"
            "gtcrn-ep050,vctk-demand-val,2.650,0.925,0.810,17.20\n"
            "gtcrn-ep100,vctk-demand-val,2.782,0.935,0.835,18.10\n"
            "gtcrn-best,vctk-demand-val,2.810,0.938,0.840,18.35\n",
            encoding="utf-8"
        )
        try:
            records = parse_eval_summary(csv_file)
        except NotImplementedError:
            records = []

        check("parses all 4 rows from CSV",
              lambda: len(records) == 4,
              "expected 4 parsed records")
        check("converts metrics to float correctly",
              lambda: records[1]["pesq"] == 2.650 and records[1]["sisdr"] == 17.20,
              "metric values must be floats")

    print("\n--- Testing Exercise 1.4: select_best_checkpoint ---")
    try:
        best_entry, why = select_best_checkpoint(records)
    except NotImplementedError:
        best_entry, why = None, ""

    check("selects checkpoint with peak validation PESQ (gtcrn-best)",
          lambda: best_entry is not None and best_entry["name"] == "gtcrn-best" and best_entry["pesq"] == 2.810,
          "peak PESQ is 2.810 from gtcrn-best")
    check("provides informative selection rationale",
          lambda: "gtcrn-best" in why and "2.810" in why,
          "rationale should state model name and score")

    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))
