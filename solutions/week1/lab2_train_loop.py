"""
Week 1 - Lab 2: Train a rumble remover                    run:  python lab2_train_loop.py

Every model you will meet in this course - a single filter or a big network - is
trained with the same 5-step loop:

    1. optimizer.zero_grad()            throw away the gradients from the last step
    2. output = model(input)            forward pass
    3. loss = loss_fn(output, target)   how wrong is the output?
    4. loss.backward()                  compute new gradients
    5. optimizer.step()                 nudge the weights downhill

The model here is ONE Conv1d layer: a learnable filter with 63 weights.
Input  = four tones + low-frequency RUMBLE (think wind, traffic, an air conditioner).
Target = the four clean tones.
Can 63 numbers learn to remove the rumble?

Your job: fill in the TODO in train(), then run this file.

Experiments for ANSWERS.md, after everything passes:
    python lab2_train_loop.py --forget-zero-grad
    python lab2_train_loop.py --kernel 5
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from scipy.signal import lfilter

from common import OUT, check, section, summary

SR = 16000
FREQS = [500, 1000, 2000, 4000]


def make_data(device):
    """(Given) clean: 4 tones, shape [4, 1, 16000].  noisy: clean + rumble."""
    t = torch.arange(SR) / SR
    clean = torch.stack([0.5 * torch.sin(2 * torch.pi * f * t) for f in FREQS]).unsqueeze(1)
    # Rumble = white noise through a "leaky integrator": every sample keeps 99.5 % of the
    # previous one, so the signal drifts slowly and almost all its energy sits below ~200 Hz.
    white = np.random.randn(len(FREQS), 1, SR)
    rumble = lfilter([1.0], [1.0, -0.995], white, axis=-1)
    rumble = torch.from_numpy(0.5 * rumble / rumble.std()).float()
    return clean.to(device), (clean + rumble).to(device)


def snr_db(ref, est):
    """(Given) Signal-to-noise ratio of an estimate in dB. Higher = closer to ref."""
    return 10 * torch.log10(ref.pow(2).sum() / (est - ref).pow(2).sum()).item()


def train(model, noisy, clean, steps=300, lr=1e-2, zero_grad=True):
    """The 5-step loop. When zero_grad is False, SKIP step 1 (that's an experiment)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    losses = []
    for step in range(steps):
        # TODO: the 5 steps (skip step 1 when zero_grad is False).
        #       Name the model output `enhanced` and the loss `loss`.
        if zero_grad:
            optimizer.zero_grad()
        enhanced = model(noisy)
        loss = loss_fn(enhanced, clean)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step:3d}   loss {loss.item():.5f}")
    return losses


def plot(losses, noisy, enhanced, clean, model, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    ax[0].plot(losses)
    ax[0].set(title="Training loss", xlabel="step", ylabel="MSE", yscale="log")

    n = 800  # the first 50 ms of the 500 Hz clip
    ax[1].plot(noisy[0, 0, :n].cpu(), lw=0.8, alpha=0.6, label="noisy")
    ax[1].plot(enhanced[0, 0, :n].cpu(), lw=1.2, label="enhanced")
    ax[1].plot(clean[0, 0, :n].cpu(), "k--", lw=1, label="clean")
    ax[1].set(title="Waveform (500 Hz clip)", xlabel="sample")
    ax[1].legend(loc="upper right")

    # What did the filter learn? Its gain at every frequency:
    w = model.weight.detach().cpu().numpy().ravel()
    f = np.fft.rfftfreq(8192, 1 / SR)
    H = 20 * np.log10(np.abs(np.fft.rfft(w, 8192)) + 1e-8)
    ax[2].semilogx(f[1:], H[1:])
    ax[2].axvspan(20, 200, color="0.85", label="most rumble is here")
    for fr in FREQS:
        ax[2].axvline(fr, color="r", ls=":", lw=1)
    ax[2].set(title="Learned filter (red = tone frequencies)", xlabel="Hz (log scale)", ylabel="gain (dB)", xlim=(20, SR / 2))
    ax[2].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    ap.add_argument("--kernel", type=int, default=63, help="filter length (odd number)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--forget-zero-grad", action="store_true", help="experiment: skip optimizer.zero_grad()")
    args = ap.parse_args()
    experiment = args.forget_zero_grad or args.kernel != 63 or args.lr != 1e-2 or args.steps != 300

    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clean, noisy = make_data(device)
    model = torch.nn.Conv1d(1, 1, kernel_size=args.kernel, padding=args.kernel // 2, bias=False).to(device)

    section("Model")
    print(" ", model)
    print("  trainable weights:", sum(p.numel() for p in model.parameters()))
    print("  input shape:", tuple(noisy.shape), "-> output shape:", tuple(model(noisy).shape))
    print(f"  device: {device} | lr: {args.lr} | kernel: {args.kernel} | zero_grad: {not args.forget_zero_grad}")

    section("Exercise 2.1  training loop")
    try:
        losses = train(model, noisy, clean, steps=args.steps, lr=args.lr, zero_grad=not args.forget_zero_grad)
    except (NotImplementedError, NameError, UnboundLocalError) as e:
        check("train() runs", lambda: False, f"finish the TODO in train()  ({type(e).__name__})")
        summary()
        raise SystemExit(1)

    with torch.no_grad():  # evaluation only: no gradients needed
        enhanced = model(noisy)
    snr_in, snr_out = snr_db(clean, noisy), snr_db(clean, enhanced)
    print(f"\n  SNR noisy    : {snr_in:6.1f} dB")
    print(f"  SNR enhanced : {snr_out:6.1f} dB   (improvement {snr_out - snr_in:+.1f} dB)")
    if experiment:
        print("  (experiment settings - FAIL lines below can be expected; that's what you are studying)")
    check("final loss is below 1/10 of the first loss", lambda: losses[-1] < 0.1 * losses[0], "check zero_grad / backward / step")
    check("SNR improved by more than 10 dB", lambda: snr_out - snr_in > 10)

    plot(losses, noisy, enhanced, clean, model, OUT / "lab2_training.png")
    sf.write(OUT / "lab2_noisy.wav", np.clip(noisy[0, 0].cpu().numpy(), -1, 1), SR)
    sf.write(OUT / "lab2_enhanced.wav", np.clip(enhanced[0, 0].cpu().numpy(), -1, 1), SR)
    print("\n  saved out/lab2_training.png, out/lab2_noisy.wav, out/lab2_enhanced.wav")
    print("  -> look at the figure, then listen to noisy vs enhanced")
    summary()
