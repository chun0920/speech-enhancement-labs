"""
Week 3 - Lab 2: A first mask network - one frame at a time        run:  python lab2_mlp.py

The network gets the normalized log power of the noisy spectrogram (Lab 1) and must output a
mask between 0 and 1 for every time-frequency bin. Training target: the ideal ratio mask.

    noisy --STFT--> Y --features--> network --> mask M      loss = mean( (M - IRM)^2 )

The first network is a multilayer perceptron (MLP) that looks at ONE frame (32 ms) at a time:

    257 numbers  ->  Linear -> ReLU -> Linear -> ReLU -> Linear  ->  257 numbers  ->  sigmoid (0...1)

It never sees the frame before or after. Keep that in mind when you look at its mistakes.

Training uses the same 5 steps as Week 1 - only the data and the model are bigger. Training runs
on the GPU if there is one (under a minute on the lab server; 15-20 minutes on a CPU).

Your job: fill in the TODOs, then run this file.
Shorter test run while debugging:   python lab2_mlp.py --steps 100
"""
import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (DEVICE, N_FREQ, OUT, TEST_SPEECH, check, count_params, evaluate, ideal_ratio_mask, istft,
                    load_audio, make_noise, mix_at_snr, section, show_tf, stft, summary)
from lab1_data import NoisyMixDataset, log_spec_features, train_files


class FrameMLP(torch.nn.Module):
    """Sees one frame at a time: 257 log-power values in, 257 mask values out."""

    def __init__(self, n_freq=N_FREQ, hidden=704):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_freq, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_freq))

    # ---------------------------------------------------------------- Exercise 2.1
    def forward(self, feats):
        """feats [B, F, T] -> mask [B, F, T], every value between 0 and 1.

        nn.Linear works on the LAST axis, so the frequencies must be last:
          1. x = feats.transpose(1, 2)           [B, F, T] -> [B, T, F]
          2. x = self.net(x)                     each of the T frames goes through the MLP on its own
          3. x = torch.sigmoid(x)                squash to 0 ... 1
          4. return x.transpose(1, 2)            back to [B, F, T]
        """
        # TODO
        raise NotImplementedError


# ---------------------------------------------------------------- Exercise 2.2
def train_step(model, noisy, clean, optimizer):
    """One training step on a batch. noisy, clean: [B, T] tensors on the right device. Return the loss (a float).

      1. Y = stft(noisy);  S = stft(clean);  N = stft(noisy - clean)
      2. target = ideal_ratio_mask(S, N)
      3. mask = model(log_spec_features(Y))
      4. loss = torch.nn.functional.mse_loss(mask, target)
      5. the Week 1 steps: optimizer.zero_grad(), loss.backward(), optimizer.step()
      6. return loss.item()
    """
    # TODO
    raise NotImplementedError


def train(model, steps, batch_size=32, lr=1e-3, seed=0):
    """(Given) Train with dynamic mixing for `steps` batches. Returns the list of losses."""
    torch.manual_seed(seed)
    model.to(DEVICE).train()
    workers = min(4, os.cpu_count() or 1)   # several CPU cores mix the next batches while the GPU trains
    loader = torch.utils.data.DataLoader(NoisyMixDataset(train_files(), seed=seed), batch_size=batch_size, num_workers=workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses, t0 = [], time.time()
    for step, (noisy, clean) in enumerate(loader):
        if step == steps:
            break
        losses.append(train_step(model, noisy.to(DEVICE), clean.to(DEVICE), optimizer))
        if step % 200 == 0 or step == steps - 1:
            print(f"  step {step:5d}/{steps}   loss {np.mean(losses[-50:]):.4f}   {time.time() - t0:5.0f} s")
    model.eval()
    return losses


def enhance_test(model, noisy):
    """(Given) Run a trained model on one 1-D numpy signal. Returns (enhanced signal, mask)."""
    with torch.no_grad():
        Y = stft(torch.from_numpy(noisy).to(DEVICE)[None])
        mask = model(log_spec_features(Y))
        return istft(mask * Y, len(noisy))[0].cpu().numpy(), mask[0].cpu()


def test_condition(snr=0):
    """(Given) The Week 2 test: the same sentence and the same pink noise. Returns (clean, noisy, noise)."""
    clean = load_audio(TEST_SPEECH)
    noisy, noise = mix_at_snr(clean, make_noise(len(clean), tilt=1.0), snr)
    return clean, noisy, noise


def plot_training(losses, mask, oracle, path, title):
    """(Given) Loss curve + the mask the network predicts next to the oracle mask."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
    ax[0].plot(np.convolve(losses, np.ones(50) / 50, mode="valid"))
    ax[0].set(title=f"{title}: training loss (smoothed)", xlabel="step", ylabel="MSE vs IRM", yscale="log")
    show_tf(ax[1], mask, f"{title}: predicted mask (0 dB test)", vmin=0, vmax=1, cmap="gray")
    show_tf(ax[2], oracle, "ideal ratio mask (the answer)", vmin=0, vmax=1, cmap="gray")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()

    model = FrameMLP()
    section("Model")
    print(f"  FrameMLP: {count_params(model):,} trainable weights | device: {DEVICE}")

    section("Exercise 2.1  forward")
    feats = torch.randn(2, N_FREQ, 50)
    try:
        out = model(feats)
    except NotImplementedError:
        check("forward() is written", lambda: False, "finish the TODO in Exercise 2.1")
        summary()
        raise SystemExit(1)
    check("output shape = input shape [2, 257, 50]", lambda: out.shape == feats.shape, "transpose to [B, T, F], run the net, transpose back")
    check("every value is between 0 and 1", lambda: out.min() >= 0 and out.max() <= 1, "torch.sigmoid")
    check("each frame is processed on its own", lambda: torch.allclose(model(feats[:, :, 10:11]), out[:, :, 10:11], atol=1e-6),
          "the MLP must not mix frames - only transpose, never reshape across time")

    section(f"Exercise 2.2  training ({args.steps} steps)")
    try:
        losses = train(model, args.steps)
    except NotImplementedError:
        check("train_step() is written", lambda: False, "finish the TODO in Exercise 2.2 (and Lab 1, which makes the data)")
        summary()
        raise SystemExit(1)
    check("loss went down by at least 30 %", lambda: np.mean(losses[-50:]) < 0.7 * np.mean(losses[:50]),
          "zero_grad / backward / step, and the target must be the IRM")
    torch.save(model.state_dict(), OUT / "mlp.pt")

    section("Test: the Week 2 sentence + pink noise at 0 dB")
    clean, noisy, noise = test_condition(0)
    enhanced, mask = enhance_test(model, noisy)
    oracle = ideal_ratio_mask(stft(torch.from_numpy(clean)), stft(torch.from_numpy(noise)))
    rows = [{"input": "noisy", **evaluate(clean, noisy)}, {"input": "MLP", **evaluate(clean, enhanced)}]
    for r in rows:
        print(f"  {r['input']:<6} PESQ {r['PESQ']:.2f}   STOI {r['STOI']:.3f}   SI-SDR {r['SI-SDR']:6.2f} dB")
    print("  (Week 2 oracle IRM at 0 dB: PESQ 2.90, STOI 0.94, SI-SDR 11.54 dB)")
    check("the MLP improves STOI and gains over 8 dB of SI-SDR", lambda: rows[1]["STOI"] > rows[0]["STOI"] and rows[1]["SI-SDR"] > rows[0]["SI-SDR"] + 8,
          "a full 2000-step run gains about 8.5 dB. Much less: is the target the IRM of clean and NOISE (noisy - clean)?")
    json.dump({**rows[1], "train_loss": float(np.mean(losses[-200:]))}, open(OUT / "mlp_test.json", "w"))
    plot_training(losses, mask, oracle, OUT / "lab2_mlp.png", "MLP")
    print("  saved out/mlp.pt, out/mlp_test.json and out/lab2_mlp.png (question Q3)")
    summary()
