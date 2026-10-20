# Week 2 — Seeing and Scoring Noise: STFT, Spectrograms, SI-SDR / PESQ / STOI

Last week noise was something you could hear and measure with one number, the SNR. This week you
look *inside* the signal — where in time and frequency the noise sits — and learn to score speech the
way every enhancement paper does.

When you finish you can:

1. cut audio into frames, apply a window, and read an FFT (which bin is which frequency, and why windows matter)
2. compute an STFT, draw a spectrogram, and invert it back to the exact waveform
3. explain where noise lives in time–frequency, and why phase matters
4. compute **SI-SDR** by hand and call **PESQ** and **STOI** correctly
5. measure the *ceiling* of mask-based enhancement with oracle masks — the number next week's network will chase

## What's in the folder

| File | Purpose | Time |
|---|---|---|
| `README.md` | this guide | |
| `requirements.txt`, `check_env.py` | two new packages + environment check | 10 min |
| `lab1_frames_fft.py` | Lab 1 — frames, windows, one FFT | 35 min |
| `lab2_spectrogram.py` | Lab 2 — STFT, spectrogram, inverse STFT, phase | 35 min |
| `lab3_metrics.py` | Lab 3 — SI-SDR, PESQ, STOI | 40 min |
| `lab4_oracle_mask.py` | Lab 4 — mini-project: oracle masks | 40 min |
| `common.py` | helpers used by the labs (leave it alone) | |
| `ANSWERS.md` | your answers to the questions | |
| `data/speech_sample.wav` | the clean speech everyone uses (see `data/README.txt`) | |
| `out/` | created by the labs; all your results land here | |

The labs work exactly like Week 1: replace each `raise NotImplementedError` under a `# TODO`, run the
file, and repeat until the last line says **All done!** Lab 4 imports your `stft` / `istft` from Lab 2
and your `evaluate` from Lab 3, so do the labs in order.

## Setup (≈ 10 min)

Same environment as Week 1:

```bash
conda activate se
pip install -r requirements.txt     # adds pesq and pystoi
python check_env.py
```

Every line should start with `[ OK ]`, and the last check runs PESQ and STOI on the sample clip.

## Why everyone uses the same clip

All labs use `data/speech_sample.wav` and noise generated from a fixed random seed. Your numbers
should therefore match your classmates' (and your mentor's). If yours differ, something in your
code differs — that is a useful thing to know.

Want to try your own noise? Labs 3 and 4 accept `--noise data/noise.wav` (or `--noise white`).
Hand in the default run.

## Order

```bash
python lab1_frames_fft.py
python lab2_spectrogram.py
python lab3_metrics.py
python lab4_oracle_mask.py
python lab4_oracle_mask.py --clean-phase    # experiment for question Q10
```

Listening is part of the work again: `out/phase/`, `out/oracle/` and `out/lab2_noisy_5dB.wav` are all
needed for `ANSWERS.md`. Copy `out/` to your laptop with `scp -r` as in Week 1, or click the files
in VS Code Remote-SSH.

## Hand-in (before next week)

A folder `week2_<yourname>/` with:

1. the four finished `lab*.py` files (each one ends with **All done!**)
2. the whole `out/` folder — especially `lab1_resolution.png`, `lab2_spectrograms.png`,
   `lab3_metrics.csv`, `week2_oracle.csv`, `week2_oracle_cleanphase.csv`, `week2_oracle.png`
3. `ANSWERS.md`, filled in

**Next week:** a small network gets only the noisy spectrogram and has to *guess* the mask. Your
Lab 4 table is the ceiling it is aiming for.

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install pesq` fails mentioning `gcc`, `cc`, `Python.h` or `Visual C++` | pip compiles pesq during install and needs a C compiler (the lab server has one; Windows laptops usually don't). Install a pre-built copy instead: `conda install -c conda-forge pesq -y`, then `pip install -r requirements.txt` again |
| `pesq.NoUtterancesError` / `No utterances detected` | the estimate you passed is (almost) silent — check that your function returns the signal, and the argument order |
| `check_env.py`: `CUDA error: no kernel image is available` | PyTorch is too old for the lab GPU (RTX 50-series): `pip uninstall -y torch`, then `pip install torch --index-url https://download.pytorch.org/whl/cu130` |
| `RuntimeError: stft requires the return_complex parameter` | add `return_complex=True` to `torch.stft` |
| reconstruction error is large in Lab 2 | `istft` must use the same `n_fft`, `hop` and Hann window as `stft` |
| `ModuleNotFoundError: lab2_spectrogram` | run the labs from inside this folder (`cd week2`) |
| numbers differ from a classmate's | compare code line by line; both runs use the same clip and the same noise seed |
| anything else | screenshot the whole error and send it to your mentor |
