# se_eval — the evaluation both groups share

One script, one locked environment, one protocol. Every number in the lab's comparison table (Week 9) comes
from here, so both groups' numbers can be put side by side. Do not edit your copy: if something is wrong,
tell your mentor, the script gets fixed once for everybody, and the version printed in every result changes.

| File | What it is |
|---|---|
| `se_eval.py` | scores a folder of enhanced WAV files: PESQ, STOI, ESTOI, SI-SDR, and DNSMOS |
| `prepare_vctk_demand.py` | downloads VCTK-DEMAND and makes the course's 16 kHz version, with fingerprints |
| `prepare_dns3_blind.py` | downloads the DNS Challenge 3 blind test set (700 noisy recordings) |
| `listen_lists/vctk_demand_test.txt`, `listen_lists/dns3_blind_track1.txt` | the 10 utterances / recordings every group saves as listening samples |
| `splits/vctk_demand_val.txt` | the 1,572 validation utterances (taken out of the training set) |
| `dnsmos/` | Microsoft's DNSMOS models (CC BY 4.0, see `NOTICE.txt`) |
| `pyproject.toml`, `uv.lock`, `.python-version` | the exact environment |

## Set up (once per computer)

```bash
SE=~/speech-enhancement-labs/tools/se_eval
cd $SE && uv sync && cd -
alias se_eval="uv run --project $SE $SE/se_eval.py"          # add this line and the SE= line to ~/.bashrc
```

No PyTorch is needed: the script only reads WAV files. `uv sync` compiles `pesq`, so `gcc` must be installed.

## Use

```bash
se_eval --enhanced out/enhanced --clean ~/data/VCTK-DEMAND/test/clean --name gtcrn-official --dataset vctk-demand-test
se_eval --enhanced out/enhanced --dnsmos --name gtcrn-official --dataset dns3-blind-track1          # no clean references
se_eval ... --listen $SE/listen_lists/vctk_demand_test.txt --noisy ~/data/VCTK-DEMAND/test/noisy   # + listening samples
```

Each run appends **one line** to `results/eval_summary.csv` (never overwritten — a re-run adds a new line) and
writes the per-file scores to `results/<name>_per_file.csv`. `--note` stores one line with the summary
(which checkpoint, which epoch). Keep `--dataset` spelled the same way every time: the tables look rows up by it.

## The protocol

| Metric | How |
|---|---|
| PESQ | ITU-T P.862.2 wide-band, `pesq` package, 16 kHz |
| STOI, ESTOI | `pystoi`, given the 16 kHz signals (it resamples to 10 kHz inside) |
| SI-SDR | in dB, both signals made zero-mean first — the "SISNR" of the GTCRN paper |
| DNSMOS | P.835 SIG / BAK / OVRL and P.808, Microsoft's `dnsmos_local.py` algorithm (9.01 s windows, 1 s hop, the non-personalized polynomial fit) |

- Every file must be **16 kHz mono**; nothing is resampled silently.
- Enhanced and clean files are matched **by file name**; a missing file stops the run.
- If an enhanced file has a different length, both signals are cut to the shorter one and the count is reported.
  A correct inference keeps the input's length.

## Calibration (2026-10-21)

On the course's VCTK-DEMAND test set (`prepare_vctk_demand.py`), unprocessed noisy files and GTCRN's official
VCTK checkpoint (`my_model.py` of Week 3):

| | PESQ | STOI | ESTOI | SI-SDR |
|---|---|---|---|---|
| noisy — se_eval | 1.968 | 0.921 | 0.787 | 8.45 |
| noisy — GTCRN paper, Table 1 | 1.97 | 0.921 | — | 8.45 |
| GTCRN official checkpoint — se_eval | 2.867 | 0.940 | 0.848 | 18.80 |
| GTCRN — paper, Table 1 | 2.87 | 0.940 | — | 18.83 |

The same checkpoint on a 16 kHz version made with torchaudio's default resampler scores PESQ 2.853 instead
of 2.867: that is why everybody uses the files made by `prepare_vctk_demand.py`.

## Data sources

- VCTK-DEMAND: C. Valentini-Botinhao, "Noisy speech database for training speech enhancement algorithms and
  TTS models," University of Edinburgh, 2017, https://doi.org/10.7488/ds/2117 — check its licence before any
  use outside the course.
- DNS Challenge 3 blind test set: github.com/microsoft/DNS-Challenge, branch `interspeech2021/adddata`,
  CC BY 4.0. C. K. A. Reddy et al., "Interspeech 2021 Deep Noise Suppression Challenge," Interspeech 2021.
