"""
Week 3 - Lab 2: The evaluation table                    run:  python lab2_table.py --model gtcrn-official

This is what Week 3 hands in (◆ acceptance week): one table with your numbers next to the paper's,
the gap between them, how the numbers were made, and the listening samples.

The table is BUILT from results/eval_summary.csv (what se_eval measured) and paper_numbers.csv (what
you copied from the paper) - never typed by hand, so it cannot disagree with the measurements.
It also records what is needed to make the same numbers again: the checkpoint and its SHA256, the
device, the data fingerprints and the se_eval version.

Before running this file (README, Steps 5-6): run infer_dir.py on the whole test set, then score the
enhanced files with se_eval using --name gtcrn-official (and --listen for the samples).
"""
import argparse
import json
from pathlib import Path

from lab1_calibrate import METRICS, RESULTS, _passed, check, read_csv, read_latest, relative_gap, summary


# ---------------------------------------------------------------- Exercise 2.1
def fmt(value, metric):
    """A table cell: SI-SDR with 2 decimals, every other metric with 3, and "—" when value is None.
    fmt(18.796, "SI-SDR") -> "18.80"   fmt(2.8667, "PESQ") -> "2.867"   fmt(None, "PESQ") -> "—"
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 2.2
def markdown_table(header, rows):
    """A Markdown table as one string: the header line, a line of ---, then one line per row.
    markdown_table(["a", "b"], [["1", "2"]]) ->
        | a | b |
        |---|---|
        | 1 | 2 |
    (lines joined with "\\n", no newline at the very end)
    """
    # TODO
    raise NotImplementedError


def provenance(model_name, dataset):
    """(Given) Where the numbers came from: inference log, data SOURCE.txt, se_eval version."""
    out = []
    log = Path("out/enhanced/inference_log.json")
    if log.exists():
        info = json.loads(log.read_text())
        out += [f"- checkpoint: `{info['checkpoint']}` (SHA256 `{info['checkpoint_sha256'][:16]}…`), "
                f"{info['parameters']:,} parameters",
                f"- inference: {info['n_files']} files on {info['device']} {info['gpu']}, torch {info['torch']}, "
                f"RTF {info['rtf']} ({info['date']})"]
    rows = [r for r in read_csv(RESULTS / "eval_summary.csv") if r["name"] == model_name and r["dataset"] == dataset]
    if rows:
        out.append(f"- scored with se_eval {rows[-1]['se_eval_version']} on {rows[-1]['date']}: {rows[-1]['n_files']} files, "
                   f"enhanced `{rows[-1]['enhanced_dir']}`, clean `{rows[-1]['clean_dir'] or '— (no reference)'}`")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the --name you gave se_eval for the model, e.g. gtcrn-official")
    ap.add_argument("--paper-name", default="gtcrn", help="the model's row name in paper_numbers.csv")
    ap.add_argument("--dataset", default="vctk-demand-test")
    ap.add_argument("--data", default=str(Path.home() / "data" / "VCTK-DEMAND"), help="the data folder (for SOURCE.txt)")
    args = ap.parse_args()

    print("--- Exercise 2.1 and 2.2 ---")
    check('fmt(18.796, "SI-SDR") == "18.80"', lambda: fmt(18.796, "SI-SDR") == "18.80")
    check('fmt(2.8667, "PESQ") == "2.867" and fmt(None, "PESQ") == "—"', lambda: fmt(2.8667, "PESQ") == "2.867" and fmt(None, "PESQ") == "—")
    check("markdown_table(['a', 'b'], [['1', '2']])", lambda: markdown_table(["a", "b"], [["1", "2"]]) == "| a | b |\n|---|---|\n| 1 | 2 |",
          'lines joined with "\\n"')
    if not all(_passed):                     # the next part uses your functions: finish them first
        summary()
        raise SystemExit(1)

    print("\n--- The table ---")
    summary_rows = read_csv(RESULTS / "eval_summary.csv")
    paper = read_csv("paper_numbers.csv")
    ours_noisy, ours_model = read_latest(summary_rows, "noisy", args.dataset), read_latest(summary_rows, args.model, args.dataset)
    paper_noisy, paper_model = read_latest(paper, "noisy", args.dataset), read_latest(paper, args.paper_name, args.dataset)
    check(f'se_eval rows for "noisy" and "{args.model}" on {args.dataset}', lambda: ours_noisy and ours_model,
          "score both with se_eval, with exactly these --name and --dataset")
    check(f'paper rows for "noisy" and "{args.paper_name}" in paper_numbers.csv', lambda: paper_noisy and paper_model)
    if not (ours_noisy and ours_model and paper_noisy and paper_model):
        raise SystemExit(1)
    metrics = [m for m in METRICS if m in ours_model]
    source = {(r["name"], r["dataset"]): r.get("source", "") for r in paper}
    rows = [["Noisy (no processing)", "ours"] + [fmt(ours_noisy.get(m), m) for m in metrics],
            ["Noisy (no processing)", f"paper: {source.get(('noisy', args.dataset), '')}"] + [fmt(paper_noisy.get(m), m) for m in metrics],
            [f"{args.model}", "ours"] + [fmt(ours_model.get(m), m) for m in metrics],
            [f"{args.paper_name}", f"paper: {source.get((args.paper_name, args.dataset), '')}"] + [fmt(paper_model.get(m), m) for m in metrics],
            ["gap, ours vs paper", ""] + [f"{100 * relative_gap(ours_model[m], paper_model[m]):.1f} %"
                                          if m in paper_model else "—" for m in metrics]]
    table = markdown_table(["system", "numbers from"] + metrics, rows)
    print(table)

    data_source = Path(args.data).expanduser() / "SOURCE.txt"
    samples = sorted((RESULTS / "samples").glob(f"{args.model}/*_enhanced.wav"))
    doc = [f"# Week 3 evaluation table: {args.model} on {args.dataset}", "", table, "",
           "## How these numbers were made", ""] + provenance(args.model, args.dataset)
    if data_source.exists():
        doc += [f"- data ({data_source}):"] + [f"  - {l.strip()}" for l in data_source.read_text().splitlines() if "fingerprint" in l]
    doc += ["", "## Listening samples", "",
            f"{len(samples)} utterances in `results/samples/{args.model}/` (noisy, enhanced, and clean if the test set has references):", ""]
    doc += [f"- {p.name.replace('_enhanced.wav', '')}" for p in samples]
    (RESULTS / "week3_table.md").write_text("\n".join(doc) + "\n", encoding="utf-8")
    print(f"\n  saved {RESULTS / 'week3_table.md'}")
    check("listening samples are there (se_eval --listen)", lambda: len(samples) >= 5,
          "run se_eval again with --listen <tools/se_eval/listen_lists/...> --noisy <noisy dir>")
    summary()
