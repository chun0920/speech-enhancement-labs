# Week 9 — Mismatched SNRs, the comparison table, and a demo page

Weeks 7 and 8 measured your model under the conditions of the paper: the VCTK-DEMAND test set, scored with
se_eval. This week answers three other questions:

1. **Mismatched conditions**: the training set was mixed at 0, 5, 10 and 15 dB SNR. What does GTCRN do at
   −5 dB, which it has never seen, and where does it help most?
2. **The comparison table**: both groups scored with the same se_eval, so every number can go into one table.
3. **The demo page**: the same 10 test utterances for every system (se_eval's listening list, from Week 3),
   side by side, so a listener can hear the differences the numbers describe.

Hand-ins for Prof. Hung (Week 10): the comparison table and the demo page.

---

## What's in the folder

| File | Purpose |
|---|---|
| `README.md` | this guide |
| `check_env.py` | checks this week's setup |
| `lab1_mismatched.py` | **your work**: mix speech and noise at an exact SNR, score a model at several SNRs, write the report |
| `lab2_make_demo.py` | **your work**: waveform pictures and the HTML demo page |
| `ANSWERS.md` | questions |

You also need `my_model.py` from Week 8 and, for your own trained model, `setrain_model.py` from Week 4.

---

## Step 0 — Setup

```bash
cp -r ~/speech-enhancement-labs/week9 ~/week9_<yourname> && cd ~/week9_<yourname>
cp ~/week8_<yourname>/my_model.py ~/week4_<yourname>/setrain_model.py .
conda activate se
SE=~/speech-enhancement-labs/tools/se_eval                          # skip these two lines if they are
alias se_eval="uv run --project $SE $SE/se_eval.py"                  # already in your ~/.bashrc (Week 3)
python check_env.py
```

---

## Step 1 — GTCRN at SNRs it was not trained on (Lab 1, ≈ 60 min)

**1a.** Finish the three TODOs of `lab1_mismatched.py` and run it until you see **All done!** (6 checks). It
tests your functions on toy signals.

**1b.** Now use them on GTCRN. The noise: in every VCTK-DEMAND test file, `noisy − clean` is exactly the DEMAND
noise that was mixed in. We take the 10 utterances of the listening list, keep their noise, and mix it again at
−5, 0, 5, 10 and 15 dB (SNR measured by mean power). Save this as `stress_gtcrn.py` and run `python stress_gtcrn.py`:

```python
# Week 9, Step 1b: GTCRN at -5 ... 15 dB SNR (the training set: 0-15 dB)
from pathlib import Path

import numpy as np
import soundfile as sf
from pesq import pesq

import my_model                    # the official checkpoint; for your own model see below
from lab1_mismatched import evaluate_snr_stress, format_stress_test_report

TEST = Path.home() / "data" / "VCTK-DEMAND" / "test"
LIST = Path.home() / "speech-enhancement-labs" / "tools" / "se_eval" / "listen_lists" / "vctk_demand_test.txt"
ids = [l.split()[0] for l in LIST.read_text().splitlines() if l.strip() and not l.startswith("#")]
clean = [sf.read(TEST / "clean" / f"{i}.wav")[0] for i in ids]
noise = [sf.read(TEST / "noisy" / f"{i}.wav")[0] - c for i, c in zip(ids, clean)]   # noisy - clean

model = my_model.build_model("cpu")
results = evaluate_snr_stress(lambda x: my_model.enhance(model, np.asarray(x, dtype=np.float32), "cpu"),
                              clean, noise, snr_levels=(-5, 0, 5, 10, 15),
                              metric_fn=lambda ref, est: pesq(16000, ref, est, "wb"))
for snr, r in results.items():
    print(f"{snr:+3d} dB   noisy PESQ {r['noisy_score']:.2f}   enhanced {r['enh_score']:.2f}   gain {r['delta']:+.2f}")
format_stress_test_report(results, "results/stress_test_report.md")
print("saved results/stress_test_report.md")
```

It runs on the CPU (about 11 s on an AMD Ryzen 5 7500F). **Your own model** (Week 7) instead of the official checkpoint: replace the
line `import my_model ...` with these two, with your run folder and epoch:

```python
import setrain_model as my_model
my_model.CHECKPOINT = Path.home() / "exp" / "gtcrn_vctk_full_<date>" / "checkpoints" / "model_<epoch>.tar"
```

Then write your observations under "Observations" in `results/stress_test_report.md`: where the gain is largest,
and what happens at −5 dB, outside the training range.

---

## Step 2 — The comparison table (≈ 30 min)

Every system both groups scored is a line in some `eval_summary.csv`, with the same columns. Ask the other group
for theirs (the one from their Week 7 folder) and save it as `results/eval_summary_other.csv`. Then save this as
`make_table.py`, put your name in the two paths, and run `python make_table.py`. It uses your Week 3 functions, so
the table is built from the measurements, never typed by hand:

```python
# Week 9, Step 2: one table from every group's eval_summary.csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "week3_<yourname>"))            # Week 3 Labs 1 and 2
from lab1_calibrate import read_csv, read_latest
from lab2_table import fmt, markdown_table

DATASET = "vctk-demand-test"
METRICS = ["PESQ", "STOI", "ESTOI", "SI-SDR"]
files = {"ours": [Path.home() / "week3_<yourname>" / "results" / "eval_summary.csv",
                  Path.home() / "week7_<yourname>" / "results" / "eval_summary.csv"],
         "other group": [Path("results") / "eval_summary_other.csv"]}
rows = []
for group, paths in files.items():
    for path in paths:
        for r in read_csv(path):
            r["name"] = f"{group}: {r['name']}"
            rows.append(r)
names = list(dict.fromkeys(r["name"] for r in rows if r["dataset"] == DATASET))     # in order, no repeats
table = markdown_table(["system"] + METRICS,
                       [[n] + [fmt(read_latest(rows, n, DATASET).get(m), m) for m in METRICS] for n in names])
print(table)
Path("results").mkdir(exist_ok=True)
Path("results/comparison_table.md").write_text(f"# All systems on {DATASET}, scored with se_eval\n\n{table}\n")
print("saved results/comparison_table.md")
```

Both groups' `noisy` rows must be identical — the same files, the same se_eval. If they are not, find out why
before you use the table. Add the paper's row from your Week 7 `paper_numbers.csv`, and parameters and RTF from
Week 8, if your group wants them in the same table.

---

## Step 3 — The demo page (Lab 2, ≈ 45 min)

**3a.** Finish the three TODOs of `lab2_make_demo.py` and run it until you see **All done!** (4 checks).

**3b.** The audio: se_eval saves the 10 listening-list utterances of every system it scores with `--listen`
(Week 3). Put one folder per system into `results/samples/`:

```bash
mkdir -p results/samples
cp -r ~/week3_<yourname>/results/samples/gtcrn-official results/samples/
se_eval --enhanced ~/week7_<yourname>/out/test_enhanced --clean ~/data/VCTK-DEMAND/test/clean \
        --name gtcrn-reproduced --dataset vctk-demand-test \
        --listen $SE/listen_lists/vctk_demand_test.txt --noisy ~/data/VCTK-DEMAND/test/noisy
```

and copy the other group's `results/samples/<their system>/` folder next to them. Then save this as
`make_demo.py` and run `python make_demo.py`:

```python
# Week 9, Step 3b: the demo page from se_eval's listening samples
import shutil
from pathlib import Path

import soundfile as sf

from lab2_make_demo import build_html_demo, downsample_waveform_envelope, generate_svg_waveform

LOG = Path.home() / "data" / "VCTK-DEMAND" / "logfiles" / "log_testset.txt"     # noise type and SNR per file
info = {l.split()[0]: (l.split()[1], float(l.split()[2])) for l in LOG.read_text().splitlines() if l.strip()}
cards = []
for system in sorted(p for p in Path("results/samples").iterdir() if p.is_dir()):
    for enh in sorted(system.glob("*_enhanced.wav")):
        uid = enh.name.replace("_enhanced.wav", "")
        card = {"id": f"{system.name}: {uid}", "noise_type": info[uid][0], "snr": f"{info[uid][1]:g} dB"}
        for kind, key in (("noisy", "noisy"), ("enhanced", "enh"), ("clean", "clean")):
            src = system / f"{uid}_{kind}.wav"
            dst = Path("demo") / "audio" / system.name / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            card[f"{key}_audio"] = dst.relative_to("demo").as_posix()
            card[f"{key}_svg"] = generate_svg_waveform(downsample_waveform_envelope(sf.read(src)[0]))
        cards.append(card)
build_html_demo(cards, "demo/index.html")
print(f"saved demo/index.html: {len(cards)} cards")
```

Open `demo/index.html` in a browser (copy the whole `demo/` folder to your laptop if the lab computer has no
screen) and listen. The same sentence appears once per system.

---

## Step 4 — Questions (`ANSWERS.md`)

Answer Q1–Q6 in `ANSWERS.md`, from your own results.

---

## Hand-in

Your folder `week9_<yourname>/` with:
- `lab1_mismatched.py` (6/6) and `lab2_make_demo.py` (4/4) finished
- `stress_gtcrn.py`, `make_table.py`, `make_demo.py`
- `results/stress_test_report.md` with your observations
- **`results/comparison_table.md`** and **`demo/`** (the whole folder, with `audio/`) — for Week 10
- `ANSWERS.md`
