"""
Week 2 - Lab 2: The spectrogram - and back again             run:  python lab2_spectrogram.py

The short-time Fourier transform (STFT) is Lab 1 done for every frame at once:

    waveform [T]  --STFT-->  complex matrix X [F, frames]  --inverse STFT-->  waveform [T]

    F = n_fft // 2 + 1 frequency bins
    |X|       magnitude: how much energy at this time and frequency
    angle(X)  phase:     where the wave is in its cycle - the fine timing inside the frame

Settings for the rest of the course (16 kHz audio):
    n_fft = 512 (32 ms frame)    hop = 128 (8 ms, 75 % overlap)    window = Hann

Almost every enhancement model works on X: look at the noisy spectrogram, change it,
turn it back into a waveform with the inverse STFT. This lab builds both directions.

Your job: fill in the TODOs, then run this file.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import HOP, N_FFT, OUT, SR, check, load_speech, make_noise, mix_at_snr, save_wav, section, show_tf, summary


# ---------------------------------------------------------------- Exercise 2.1
def stft(x, n_fft=N_FFT, hop=HOP):
    """Complex STFT of a 1-D float tensor -> complex tensor [n_fft // 2 + 1, frames].

    Use torch.stft with  n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft),
    return_complex=True.  (Leave everything else at its default.)
    """
    # TODO
    return torch.stft(x, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft), return_complex=True)


# ---------------------------------------------------------------- Exercise 2.2
def power_db(X):
    """Power spectrogram in dB:  10 * log10( |X|^2 + 1e-10 ).  Real tensor, same shape as X."""
    # TODO
    return 10 * torch.log10(X.abs() ** 2 + 1e-10)


# ---------------------------------------------------------------- Exercise 2.3
def istft(X, length, n_fft=N_FFT, hop=HOP):
    """Inverse STFT -> 1-D float tensor with exactly `length` samples.

    Use torch.istft with the SAME n_fft, hop and window as stft(), plus length=length.
    """
    # TODO
    return torch.istft(X, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft), length=length)


if __name__ == "__main__":
    clean = torch.from_numpy(load_speech())
    T = len(clean)
    print(f"clean speech: {T} samples = {T / SR:.2f} s")

    section("Exercise 2.1  stft")
    try:
        S = stft(clean)
    except NotImplementedError:
        check("stft() is written", lambda: False, "finish the TODO in Exercise 2.1")
        summary()
        raise SystemExit(1)
    print(f"  waveform {tuple(clean.shape)}  ->  STFT {tuple(S.shape)}  {S.dtype}")
    check("257 frequency bins", lambda: S.shape[0] == N_FFT // 2 + 1)
    check("1 + T // hop frames", lambda: S.shape[1] == 1 + T // HOP,
          "torch.stft pads n_fft // 2 on both ends (center=True), so it is not Lab 1's 1 + (T - n_fft) // hop")
    check("complex numbers", lambda: S.is_complex(), "return_complex=True")
    t1k = 0.5 * torch.sin(2 * torch.pi * 1000 * torch.arange(SR) / SR)
    check("a 1000 Hz tone lands in bin 32", lambda: stft(t1k).abs().pow(2).sum(dim=1).argmax().item() == 32, "bin k = k * 16000 / 512 Hz")

    section("Exercise 2.2  power in dB")
    check("real, same shape as X", lambda: not power_db(S).is_complex() and power_db(S).shape == S.shape, "use |X| = X.abs()")
    check("equals 20 * log10 |X| where |X| is not tiny", lambda: torch.allclose(
        power_db(S)[S.abs() > 1e-3], 20 * torch.log10(S.abs()[S.abs() > 1e-3]), atol=1e-3), "10 * log10(|X|**2 + 1e-10)")

    section("Exercise 2.3  istft - and back to a waveform")
    try:
        back = istft(S, T)
    except NotImplementedError:
        check("istft() is written", lambda: False, "finish the TODO in Exercise 2.3")
        summary()
        raise SystemExit(1)
    err = (back - clean).abs().max().item()
    print(f"  largest error after STFT -> iSTFT: {err:.2e}")
    check("exactly `length` samples", lambda: len(back) == T and len(istft(stft(clean[:16001]), 16001)) == 16001, "pass length=length")
    check("perfect reconstruction (error < 1e-5)", lambda: err < 1e-5, "same n_fft, hop and window as stft()")

    section("Where is the noise? Pink noise at +5 dB SNR")
    noisy, noise = (torch.from_numpy(a) for a in mix_at_snr(clean.numpy(), make_noise(T, "pink"), 5))
    Y, N = stft(noisy), stft(noise)
    louder = (N.abs() > S.abs()).float().mean().item()
    print(f"  the noise is louder than the speech in {100 * louder:.0f} % of all time-frequency bins")
    print("  ... even though, over the whole file, the speech is 5 dB louder than the noise.")
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    top = power_db(Y).max().item()
    for ax, X, name in zip(axes, (S, N, Y), ("clean speech", "noise", "noisy = speech + noise (5 dB)")):
        im = show_tf(ax, power_db(X), name, vmin=top - 80, vmax=top)
        ax.label_outer()
    fig.colorbar(im, ax=axes, label="dB", shrink=0.6)
    fig.savefig(OUT / "lab2_spectrograms.png", dpi=110); plt.close(fig)
    save_wav(OUT / "lab2_noisy_5dB.wav", noisy.numpy())

    section("Short vs long frames on real speech")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, n_fft in zip(axes, (128, 2048)):
        X = stft(clean, n_fft=n_fft, hop=32)
        show_tf(ax, power_db(X), f"n_fft = {n_fft} ({1000 * n_fft // SR} ms frames)", hop=32)
        ax.set_ylim(0, 4000)
    fig.tight_layout(); fig.savefig(OUT / "lab2_resolution.png", dpi=110); plt.close(fig)

    section("Is phase important?")
    mag = S.abs()
    torch.manual_seed(0)
    versions = {
        "original phase": S,
        "zero phase": mag.to(torch.complex64),
        "random phase": mag * torch.exp(1j * 2 * torch.pi * torch.rand(mag.shape)),
        "phase of the noisy mix (0 dB)": mag * torch.exp(1j * stft(torch.from_numpy(
            mix_at_snr(clean.numpy(), make_noise(T, "pink"), 0)[0])).angle()),
    }
    for i, (name, X) in enumerate(versions.items()):
        path = OUT / "phase" / f"{i}_{name.split(' (')[0].replace(' ', '_')}.wav"
        save_wav(path, istft(X, T).numpy())
        print(f"  clean magnitude + {name:<30} -> {path}")
    print("  Same magnitude in all four. Listen to them (question Q5).")

    print("\n  saved out/lab2_spectrograms.png, out/lab2_resolution.png, out/phase/*.wav")
    summary()
