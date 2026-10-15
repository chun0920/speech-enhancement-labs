"""Helpers shared by all labs this week - no need to edit this file."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen on the server -> figures are written to PNG files

OUT = Path("out")
OUT.mkdir(exist_ok=True)
DATA = Path("data")

_passed = []


def section(title):
    print(f"\n--- {title} ---")


def check(name, fn, hint=""):
    """Run fn() and print PASS if it returns True, otherwise FAIL plus a hint."""
    try:
        ok = bool(fn())
    except NotImplementedError:
        ok, hint = False, "not written yet - look for the # TODO"
    except Exception as e:  # show the error but keep going
        ok, hint = False, f"{type(e).__name__}: {e}  {hint}".strip()
    _passed.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok and hint:
        print(f"         hint: {hint}")
    return ok


def summary():
    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))
