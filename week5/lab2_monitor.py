"""
Week 5 - Lab 2: Training Curve Monitor & Convergence Analysis       run:  python lab2_monitor.py

During a 200-epoch training run (~10 hours on lab112), you need to monitor training health:
  - Is the learning rate warming up and decaying along the cosine schedule?
  - Is the training loss falling steadily without gradient divergence?
  - Is validation loss tracking training loss (healthy generalization) or diverging (overfitting)?
  - Which epoch achieved the peak validation PESQ?

This lab builds the monitoring toolkit:
  1. Analytical learning rate calculator matching SEtrain's WarmupLR scheduler
  2. Training log parser for console output and TensorBoard scalars
  3. Automated convergence analyzer and CSV curve exporter for results/training_curves.csv
"""
import math
import re
import sys
from pathlib import Path

RESULTS = Path("results")
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
def compute_lr(step, warmup_steps=25000, decay_until_step=250000, max_lr=1e-3, min_lr=1e-6):
    """Compute the expected learning rate at a given training step:
         - If step < warmup_steps:
             Linear warmup: max_lr * step / warmup_steps
         - If step >= decay_until_step:
             Returns min_lr
         - If warmup_steps <= step < decay_until_step:
             Cosine annealing decay:
               decay_ratio = (step - warmup_steps) / (decay_until_step - warmup_steps)
               coeff = 0.5 * (1.0 + cos(pi * decay_ratio))
               return min_lr + coeff * (max_lr - min_lr)
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 2.2
def parse_training_log(text):
    """Parse SEtrain log output text to extract per-epoch metrics.

    SEtrain log lines typically contain:
      train[1/200][2026-10-21-18:35] train_loss=7.291
      validate[1/200][2026-10-21-18:36] valid_loss=5.640, pesq=1.416

    Returns:
      records: dict keyed by epoch int:
        {
          epoch: {
            "epoch": int,
            "train_loss": float,
            "val_loss": float,
            "val_pesq": float
          }
        }
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 2.3
def analyze_convergence(records):
    """Analyze the parsed training records and return a summary dict:
         - "best_pesq_epoch": epoch int with highest val_pesq
         - "best_pesq": maximum val_pesq float
         - "lowest_val_loss_epoch": epoch int with lowest val_loss
         - "lowest_val_loss": minimum val_loss float
         - "gap_at_best": val_loss - train_loss at best_pesq_epoch
         - "is_healthy": True if both train_loss and val_loss decreased overall
    """
    # TODO
    raise NotImplementedError


def write_curves_csv(records, out_path, steps_per_epoch=1250):
    """(Given) Write parsed training curves to CSV with step-calculated learning rates."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["epoch,step,train_loss,val_loss,val_pesq,lr"]
    for ep in sorted(records.keys()):
        r = records[ep]
        step = ep * steps_per_epoch
        lr = compute_lr(step)
        t_l = f"{r['train_loss']:.4f}" if r.get("train_loss") is not None else ""
        v_l = f"{r['val_loss']:.4f}" if r.get("val_loss") is not None else ""
        v_p = f"{r['val_pesq']:.4f}" if r.get("val_pesq") is not None else ""
        lines.append(f"{ep},{step},{t_l},{v_l},{v_p},{lr:.6e}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ================================================================ Test Suite
if __name__ == "__main__":
    print("--- Testing Exercise 2.1: compute_lr ---")
    try:
        lr_0 = compute_lr(0)
        lr_warmup_half = compute_lr(12500)
        lr_peak = compute_lr(25000)
        lr_mid_decay = compute_lr(137500)
        lr_end = compute_lr(250000)
        lr_post_end = compute_lr(300000)
    except NotImplementedError:
        lr_0, lr_warmup_half, lr_peak = None, None, None
        lr_mid_decay, lr_end, lr_post_end = None, None, None

    check("step 0 starts at 0.0 lr",
          lambda: lr_0 == 0.0,
          "warmup starts from 0 at step 0")
    check("linear warmup at midpoint (step 12500) is exactly 5e-4",
          lambda: lr_warmup_half is not None and math.isclose(lr_warmup_half, 5e-4, rel_tol=1e-5),
          "linear interpolation from 0 to 1e-3 reaches 5e-4 halfway")
    check("reaches max_lr 1e-3 at end of warmup (step 25000)",
          lambda: lr_peak is not None and math.isclose(lr_peak, 1e-3, rel_tol=1e-5),
          "at step 25000, lr should equal max_lr")
    check("cosine decay at midpoint (step 137500) is ~5.005e-4",
          lambda: lr_mid_decay is not None and 4.9e-4 < lr_mid_decay < 5.1e-4,
          "at the midpoint of cosine decay, coeff is 0.5")
    check("decays to min_lr 1e-6 at step 250000",
          lambda: lr_end is not None and math.isclose(lr_end, 1e-6, rel_tol=1e-5),
          "at decay_until_step, lr should reach min_lr")
    check("stays at min_lr after step 250000",
          lambda: lr_post_end is not None and math.isclose(lr_post_end, 1e-6, rel_tol=1e-5),
          "steps beyond decay_until_step must not drop below min_lr")

    print("\n--- Testing Exercise 2.2: parse_training_log ---")
    sample_log = """
2026-10-21 18:32:00 INFO: Starting training
   train[1/200][2026-10-21-18:35] train_loss=7.291
validate[1/200][2026-10-21-18:36] valid_loss=5.640, pesq=1.416
   train[2/200][2026-10-21-18:38] train_loss=3.826
validate[2/200][2026-10-21-18:39] valid_loss=3.365, pesq=1.619
   train[3/200][2026-10-21-18:41] train_loss=2.028
validate[3/200][2026-10-21-18:42] valid_loss=2.109, pesq=1.840
   train[4/200][2026-10-21-18:44] train_loss=1.650
validate[4/200][2026-10-21-18:45] valid_loss=1.820, pesq=1.950
"""
    try:
        parsed = parse_training_log(sample_log)
    except NotImplementedError:
        parsed = {}

    check("parses all 4 epochs from sample log",
          lambda: len(parsed) == 4,
          "expected 4 epochs parsed into dictionary")
    check("correctly extracts train_loss, val_loss, and pesq for epoch 3",
          lambda: parsed.get(3) == {"epoch": 3, "train_loss": 2.028, "val_loss": 2.109, "val_pesq": 1.840},
          "epoch 3 must match the Week 4 measured values")

    print("\n--- Testing Exercise 2.3: analyze_convergence ---")
    try:
        analysis = analyze_convergence(parsed)
    except NotImplementedError:
        analysis = {}

    check("identifies epoch 4 as best validation PESQ (1.950)",
          lambda: analysis.get("best_pesq_epoch") == 4 and math.isclose(analysis.get("best_pesq", 0), 1.950),
          "best_pesq_epoch should be 4 with pesq 1.950")
    check("identifies epoch 4 as lowest validation loss (1.820)",
          lambda: analysis.get("lowest_val_loss_epoch") == 4 and math.isclose(analysis.get("lowest_val_loss", 0), 1.820),
          "lowest_val_loss_epoch should be 4")
    check("assesses training trajectory as healthy",
          lambda: analysis.get("is_healthy") is True,
          "both train and val loss decreased, so is_healthy is True")

    if len(sys.argv) > 1:                        # python lab2_monitor.py <your train_vctk_full.log>
        log = Path(sys.argv[1]).expanduser()
        try:
            real = parse_training_log(log.read_text(encoding="utf-8", errors="replace"))
            summary_real = analyze_convergence(real) if real else {}
        except NotImplementedError:
            real, summary_real = {}, {}
        if real:
            csv_file = write_curves_csv(real, RESULTS / "training_curves.csv")
            print(f"\n  {log}: {len(real)} epochs -> {csv_file}")
            print(f"  {summary_real}")
        else:
            print(f"\n  {log}: no epochs found - finish the exercises, and give the log of train.py")
    else:
        print("\n  Your own curves: python lab2_monitor.py ~/baselines/SEtrain/train_vctk_full.log")

    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))
