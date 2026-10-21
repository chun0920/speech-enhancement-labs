# Week 5 — Full Training on lab112 & Loss Curve Dynamics

In Week 4, you walked through GTCRN's code, verified every tensor shape, adapted the dataset for
VCTK-DEMAND, and ran a short 3-epoch trial run. Now comes the real training run: **200 epochs**
(250,000 steps, ~10 hours on lab112).

While the model trains on the server in the background, you will dissect the objective function
that drives it: **HybridLoss**. You will implement power-law spectral compression ($c = 0.3$),
complex spectrogram loss ($\lambda_{RI} = 30$), magnitude loss ($\lambda_{Mag} = 70$), and
time-domain SI-SNR loss, verifying scale invariance and gradient backpropagation. Then you will
monitor the full run's learning rate schedule and loss convergence curves.

When you finish you can:

1. Launch and manage long-running GPU training jobs on remote servers using `tmux` or `nohup`
2. Explain why speech enhancement combines time-domain and frequency-domain losses
3. Implement power-law compressed spectral loss and verify its psychoacoustic rationale
4. Verify time-domain scale-invariant SNR (SI-SNR) and understand why scale invariance matters
5. Track and diagnose training health: warmup, cosine decay, loss convergence, and overfitting
6. Master's students: Report 2 — the mathematical and experimental role of each loss component

---

## What's in the folder

| File | Purpose |
|---|---|
| `README.md` | this guide |
| `check_env.py` | checks this week's setup |
| `cfg_train_vctk_full.yaml` | SEtrain configuration for 200 epochs on VCTK-DEMAND |
| `lab1_loss.py` | **your work**: implement and verify HybridLoss (compression, spectral, SI-SNR) |
| `lab2_monitor.py` | **your work**: learning rate schedule, log parser, and convergence analysis |
| `report2_template.md` | master's students: Report 2 (Loss components) |
| `ANSWERS.md` | analytical questions on loss terms and training dynamics |

You also need `my_model.py`, `infer_dir.py`, and `dataloader_vctk.py` from your Week 4 folder (Step 0).

---

## Step 0 — Setup and prerequisites

Copy last week's files into your week 5 working folder:

```bash
cp -r ~/speech-enhancement-labs/week5 ~/week5_<yourname> && cd ~/week5_<yourname>
cp ~/week4_<yourname>/my_model.py ~/week4_<yourname>/infer_dir.py ~/week4_<yourname>/dataloader_vctk.py .
conda activate se
python check_env.py
```

Ensure `check_env.py` shows all `[ OK ]`.

---

## Step 1 — Launch 200-Epoch Training on lab112

In Week 4, you verified that the 200-epoch schedule takes ~10 hours on lab112 (about 3 minutes per epoch).
Because SSH connections can disconnect, **always run long training jobs inside a persistent session** (`tmux` or `nohup`).

### 1.1 Copy the 200-epoch configuration
```bash
cp cfg_train_vctk_full.yaml ~/baselines/SEtrain/configs/
```

### 1.2 Note on the STFT Window (Engineering Decision)
In Week 4 (Question Q2), you saw that GTCRN's official checkpoint scored **2.867 PESQ** with square-root Hann
(`torch.hann_window(512).pow(0.5)`) vs **2.837 PESQ** with plain Hann.
To ensure your trained model matches the paper's 2.87 result in Week 7:
In `~/baselines/SEtrain/models/gtcrn_end2end.py`, change line 309:
```python
# From:
'window': torch.hann_window(self.win_len).to(device), 'onesided': True}
# To:
'window': torch.hann_window(self.win_len).pow(0.5).to(device), 'onesided': True}
```
Record this change in `ANSWERS.md`.

### 1.3 Check GPU status and launch
Check that GPU 0 has sufficient free memory (`nvidia-smi`):
```bash
nvidia-smi
```
> [!NOTE]
> The server may be shared with other researchers. Keep `batch_size: 8` (allocated VRAM ~2.2 GB) so multiple jobs coexist cleanly.

Start the training job with `nohup` (or in a `tmux` window):
```bash
cd ~/baselines/SEtrain
nohup python train.py -C configs/cfg_train_vctk_full.yaml -D 0 > train_vctk_full.log 2>&1 &
echo $! > train.pid
```

Monitor live progress:
```bash
tail -f train_vctk_full.log
```
Checkpoints are saved every 10 epochs (`model_010.tar`, `model_020.tar`, ...) and the peak validation checkpoint is tracked as `best_model_<epoch>.tar`.

---

## Step 2 — Dissecting HybridLoss (Lab 1, ≈ 45 min)

While the training runs on the server, work on your local environment to dissect the loss function:

```bash
python lab1_loss.py
```

Complete the four TODOs:
1. `power_compress()`: Power-law compression $X_c = X / (|X| + \epsilon)^{1 - c}$ with $c = 0.3$.
2. `spectral_loss()`: Weighted combination of Real+Imag MSE ($\lambda_{RI} = 30$) and Magnitude MSE ($\lambda_{Mag} = 70$).
3. `sisnr_loss()`: Time-domain Scale-Invariant SNR projection and logarithmic loss.
4. `hybrid_loss()`: Complete forward pass from waveform to scalar loss and backpropagation gradient check.

Run until all 9 checks show `[PASS]` and you see **All done!**.

---

## Step 3 — Training Monitor & Convergence Analysis (Lab 2, ≈ 30 min)

```bash
python lab2_monitor.py
```

Complete the three exercises:
1. `compute_lr()`: Calculates the exact learning rate for any step according to the linear warmup (steps 1–25,000) and cosine decay (steps 25,000–250,000).
2. `parse_training_log()`: Extracts `train_loss`, `val_loss`, and `val_pesq` across epochs from training logs.
3. `analyze_convergence()`: Identifies the epoch with peak validation PESQ, lowest validation loss, and checks overall convergence health.

When your actual lab112 training has completed 10–20+ epochs, you can feed your real `train_vctk_full.log` into `lab2_monitor.py` to generate your real `results/training_curves.csv`.

---

## Step 4 — Master's Students: Report 2

Master's students: fill in `report2_template.md` (approx. 10-minute presentation to Prof. Hung).
Topic: **Loss 各項的作用 (The Role of Each Loss Component)**:
- Why both time-domain and frequency-domain?
- Psychoacoustics of $c = 0.3$ compression
- Phase sensitivity ($\mathcal{L}_{RI}$) vs spectral envelope ($\mathcal{L}_{Mag}$)
- Waveform fidelity ($\text{SI-SNR}$)
- Observed training curve trends from your run

---

## Step 5 — Analytical Questions (`ANSWERS.md`)

Answer questions Q1–Q8 in `ANSWERS.md`:
- Q1–Q4: The math, weighting, and psychoacoustic properties of `HybridLoss`.
- Q5–Q6: Learning rate schedule dynamics (warmup rationale, cosine decay).
- Q7–Q8: Diagnosing training stability, overfitting, and remote process management.

---

## Hand-in

Your folder `week5_<yourname>/` containing:
- Both lab scripts finished (`lab1_loss.py` 9/9, `lab2_monitor.py` 11/11, both ending with **All done!**)
- `results/training_curves.csv` (generated from training log or monitor)
- `ANSWERS.md` filled in
- Master's students: `report2_template.md` filled in

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | GPU VRAM full | Check `nvidia-smi`; ensure no zombie Python processes (`pkill -f train.py`) |
| Training stops immediately | Config syntax error or missing dataloader | Check `train_vctk_full.log`; verify `dataloader_vctk.py` is in `~/baselines/SEtrain/` |
| `loss is NaN` in Lab 1 | Missing `clamp(min=eps)` before division or log10 | Ensure `eps = 1e-12` is clamped when computing magnitude and noise norm |
| SI-SNR loss not scale invariant | Residual calculation altered | Projection formula must be $(\langle s, \hat{s} \rangle / \|s\|^2) s$ |
| Learning rate warmup stays at 0 | Integer division | Use float division `max_lr * step / warmup_steps` |
