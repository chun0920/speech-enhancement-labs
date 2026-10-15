# Week 3 — Your First Enhancement Network: Learning to Guess the Mask

In Week 2 the oracle mask cleaned the noisy sentence beautifully — but it cheated: it was computed
from the clean speech. This week a neural network has to *guess* the mask from the noisy
spectrogram alone. You will train two networks of the same size, test them on the Week 2 sentence,
and find out how close they get to the oracle — and where they fail completely.

When you finish you can:

1. feed a network with **dynamic mixing**: endless (noisy, clean) pairs mixed on the fly with a `Dataset` and a `DataLoader`
2. build a mask network in PyTorch and train it with the same 5-step loop as Week 1
3. run a **fair comparison**: two models with the same size, data, loss and steps — one change (memory)
4. evaluate on a held-out test, report the result as a **fraction of the oracle ceiling**, and test on a noise the models never heard

## What's in the folder

| File | Purpose | Time |
|---|---|---|
| `README.md` | this guide | |
| `requirements.txt`, `check_env.py` | same packages as Week 2 + environment check | 5 min |
| `lab1_data.py` | Lab 1 — random crops, a Dataset that mixes on the fly, normalized features | 35 min |
| `lab2_mlp.py` | Lab 2 — a network that judges one frame at a time; train it | 35 min |
| `lab3_gru.py` | Lab 3 — the same size of network, with a memory; train it | 40 min |
| `lab4_evaluate.py` | Lab 4 — mini-project: the exam on unseen speech and unseen noise | 40 min |
| `common.py` | helpers: everything from Weeks 1–2 (mixing, STFT, IRM, PESQ/STOI/SI-SDR) | |
| `ANSWERS.md` | your answers to the questions | |
| `data/` | 117 training clips, the test sentence, a babble noise (see `data/README.txt`) | |
| `out/` | created by the labs; trained models, figures and sound files land here | |

The labs work like Weeks 1 and 2: replace each `raise NotImplementedError` under a `# TODO`, run the
file, and repeat until the last line says **All done!** The labs build on each other — Lab 2 imports
your Dataset and features from Lab 1, Lab 3 reuses Lab 2's training code, and Lab 4 loads the models
that Labs 2 and 3 saved — so do them in order.

## Setup (≈ 5 min)

Same environment as Weeks 1 and 2 — nothing new to install:

```bash
conda activate se
pip install -r requirements.txt
python check_env.py
```

Every line should start with `[ OK ]`, including the GPU line and the two lines about `data/`.

## Training time

Labs 2 and 3 each train a network for 2000 steps. On the lab server's GPU that takes under a
minute for the MLP and about a minute for the GRU (a little longer when others use the GPU too).
On a computer **without** a GPU it takes
15–20 minutes each.

While you are still fixing your code, train for only a few steps so you don't wait each time:

```bash
python lab2_mlp.py --steps 100
```

The checks that compare scores need the full 2000 steps — do one full run of each lab at the end.

## Order

```bash
python lab1_data.py
python lab2_mlp.py          # saves out/mlp.pt
python lab3_gru.py          # saves out/gru.pt
python lab4_evaluate.py     # loads both models
```

Listen to `out/lab1_examples/` and `out/exam/` — several questions in `ANSWERS.md` need your ears.
Copy `out/` to your laptop with `scp -r` as in Week 1, or click the files in VS Code Remote-SSH.

## Your numbers vs. your classmates'

Training starts from fixed random seeds, so everyone's networks end up very similar — but not
bit-for-bit identical: GPUs add tiny rounding differences. Expect your scores to match a
classmate's to about ±0.05 in PESQ, ±0.02 in STOI and ±0.5 dB in SI-SDR. The "noisy" and "oracle" rows
in Lab 4 use no training, so those must match Week 2 exactly.

## Hand-in (before next week)

A folder `week3_<yourname>/` with:

1. the four finished `lab*.py` files (each one ends with **All done!** after a full 2000-step run)
2. the whole `out/` folder — especially `lab2_mlp.png`, `lab3_gru.png`, `week3_results.csv`,
   `week3_results.png`, `week3_masks.png` and `out/exam/`
3. `ANSWERS.md`, filled in

## Troubleshooting

| Problem | Fix |
|---|---|
| `check_env.py`: `CUDA error: no kernel image is available` | PyTorch is too old for the lab GPU (RTX 50-series): `pip uninstall -y torch`, then `pip install torch --index-url https://download.pytorch.org/whl/cu130` |
| `check_env.py`: the `data/train` line fails | the `data/` folder is missing or incomplete — unzip the whole `week3` folder again |
| `out/mlp.pt not found` in Lab 4 | run `lab2_mlp.py` and `lab3_gru.py` first |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` or `input.size(-1) must be equal to input_size` | the frequencies must be the last axis before `nn.Linear` / `nn.GRU`: `feats.transpose(1, 2)` |
| `RuntimeError: Expected all tensors to be on the same device` | in `train_step`, everything comes from `noisy` and `clean`, which are already on the GPU — don't create new tensors on the CPU |
| `CUDA out of memory` | someone else is using the GPU heavily — wait, or train on the CPU with fewer steps |
| training is very slow | check that `check_env.py` shows the GPU; with no GPU, use `--steps 500` while testing |
| the loss does not go down | the target must be the IRM of the *clean* and the *noise* (`noisy - clean`), and the five training steps must be in the Week 1 order |
| anything else | screenshot the whole error and send it to your mentor |
