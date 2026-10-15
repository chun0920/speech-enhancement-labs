# Week 1 — Audio as Data: Tensors, a Training Loop, WAV Files, SNR

This week you build the toolbox for the rest of the course. Everything runs in a terminal on the
lab's Linux server (no Jupyter).

When you finish you can:

1. run PyTorch from the terminal, on the GPU if there is one
2. move audio around as tensors shaped **`[batch, channel, time]`**
3. write the **5-step training loop** and train a one-layer "rumble remover"
4. open a WAV file and know exactly what is inside: channels, integers, sample rate, bit depth
5. build a small **paired dataset** (noisy input → clean target) at chosen SNRs, with a manifest

## What's in the folder

| File | Purpose | Time |
|---|---|---|
| `README.md` | this guide | |
| `requirements.txt`, `check_env.py` | install + environment check | 20 min |
| `lab1_tensors.py` | Lab 1 — tones, fades, batches, normalization | 35 min |
| `lab2_train_loop.py` | Lab 2 — train a filter to remove low-frequency rumble | 35 min |
| `lab3_audio_io.py` | Lab 3 — channels, int16 samples, resampling, saving without clipping | 40 min |
| `lab4_noisy_mix.py` | Lab 4 — mini-project: a paired noisy dataset + manifest | 40 min |
| `common.py` | helpers used by the labs (leave it alone) | |
| `ANSWERS.md` | your answers to the questions | |
| `data/` | put audio files here | |
| `out/` | created by the labs; all your results land here | |

## How a lab works

Each `lab*.py` has functions containing `# TODO` and `raise NotImplementedError`.

1. Open the file (`nano`, `vim`, or VS Code with Remote-SSH).
2. Read the function's docstring and replace `raise NotImplementedError` with your code.
3. Run it, e.g. `python lab1_tensors.py`.
4. `[PASS]` means correct. `[FAIL]` comes with a hint. Edit and run again until the last line says **All done!**

Run each file once **before** you write anything: Lab 1 starts with a demo that prints every tensor
operation you will need.

> Stuck for more than 10 minutes? Ask — that's what the lab session is for.

## Setup (≈ 20 min)

**1. Make a conda environment** (ask your mentor if `conda` is missing)

```bash
conda create -n se python=3.11 -y
conda activate se
```

**2. Find the CUDA version of the server**

```bash
nvidia-smi
```

It is printed top-right, e.g. `CUDA Version: 13.2`, next to the GPU name.

**3. Install PyTorch** — pick the line that matches (your mentor will confirm):

```bash
# lab server (RTX 5060 Ti, CUDA Version 13.x) - or any machine whose nvidia-smi shows CUDA 13.0 or newer
pip install torch --index-url https://download.pytorch.org/whl/cu130
# no GPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Do **not** use an older `cu12x` build on the lab server: it installs without complaint, but RTX 50-series
GPUs need a newer build, and the first GPU operation fails with `no kernel image is available`.
Other machines: <https://pytorch.org/get-started/locally/>

**4. Install the rest**

```bash
pip install -r requirements.txt
```

**5. Check**

```bash
python check_env.py
```

Every line should start with `[ OK ]`.

## Audio files

Put these in `data/`:

- `clean_speech.wav` — a clean recording of one person speaking for 2–5 s. Your mentor can give you
  one, or record yourself and convert it to WAV. No file? Lab 3 generates a synthetic stand-in.
- `noise.wav` — optional: any noise recording (fan, street, café, rain). Without it Lab 4 uses pink noise.

### Listening on a server

The server has no speakers. Copy the results to your laptop — run this **on the laptop**:

```bash
scp -r your_username@SERVER_ADDRESS:~/week1/out ./out
```

(`~/week1` is wherever this folder lives.) In VS Code Remote-SSH you can also click a `.wav` or
`.png` in the file explorer.

**Listening is part of the work.** Several questions in `ANSWERS.md` can only be answered with headphones.

## Order

```bash
python lab1_tensors.py
python lab2_train_loop.py
python lab3_audio_io.py        # writes out/speech_16k.wav, which Lab 4 needs
python lab4_noisy_mix.py
```

After Lab 2 passes, run the two experiments that `ANSWERS.md` asks about:

```bash
python lab2_train_loop.py --forget-zero-grad
python lab2_train_loop.py --kernel 5
```

(They overwrite `out/lab2_*` — run the plain `python lab2_train_loop.py` again at the end so the
hand-in has the default result.)

## Hand-in (before next week)

A folder `week1_<yourname>/` with:

1. the four finished `lab*.py` files (each one ends with **All done!**)
2. the whole `out/` folder — especially `out/pairs/` (8 WAV files + `manifest.csv`),
   `week1_pairs.png`, `lab2_training.png`, `lab3_spectrum.png`
3. `ANSWERS.md`, filled in

**Next week:** the short-time Fourier transform. You will look at your noisy pairs as spectrograms
and score them with PESQ, STOI and SI-SDR.

## Troubleshooting

| Problem | Fix |
|---|---|
| `conda: command not found` | ask your mentor, or try `source ~/miniconda3/bin/activate` |
| `ModuleNotFoundError: No module named 'torch'` | every new terminal needs `conda activate se` again |
| `check_env.py` shows no GPU on a GPU server | you installed the CPU build: redo step 3 with the `cu130` line |
| `CUDA error: no kernel image is available for execution on the device` | PyTorch is too old for this GPU: `pip uninstall -y torch`, then step 3 with the `cu130` line |
| `OSError: sndfile library not found` | `conda install -c conda-forge libsndfile` |
| `LibsndfileError: Format not recognised` | your file is m4a/mp3/aac — convert it: `ffmpeg -i input.m4a data/clean_speech.wav` |
| `out/speech_16k.wav not found` in Lab 4 | run Lab 3 first |
| anything else | screenshot the whole error and send it to your mentor |
