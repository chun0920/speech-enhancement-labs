"""Helpers shared by all labs this week - no need to edit this file.

Everything you built in Weeks 1 and 2 (mixing at an SNR, STFT / iSTFT, the ideal ratio mask,
PESQ / STOI / SI-SDR) is here, ready to use.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen on the server -> figures are written to PNG files

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

OUT = Path("out")
OUT.mkdir(exist_ok=True)
DATA = Path("data")
TRAIN_DIR = DATA / "train"                 # 117 clean LibriSpeech clips for training (see data/README.txt)
TEST_SPEECH = DATA / "speech_sample.wav"   # the Week 2 sentence - never used for training
BABBLE = DATA / "noise_babble.flac"        # six people talking at once - never used for training
SR = 16000
N_FFT, HOP = 512, 128                      # 32 ms Hann frames, 8 ms hop - same as Week 2
N_FREQ = N_FFT // 2 + 1                    # 257 frequency bins
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_passed = []


def section(title):
    print(f"\n--- {title} ---")


def check(name, fn, hint=""):
    """Run fn() and print PASS if it returns True, otherwise FAIL plus a hint."""
    try:
        ok = bool(fn())
    except NotImplementedError:
        ok, hint = False, "not written yet - look for the # TODO"
    except Exception as e:  # show the error but keep going
        ok, hint = False, f"{type(e).__name__}: {e}  {hint}".strip()
    _passed.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok and hint:
        print(f"         hint: {hint}")
    return ok


def summary():
    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))


# ------------------------------------------------------------------ audio (Week 1)
def load_audio(path):
    """Mono float32 at 16 kHz (all files shipped this week are already 16 kHz)."""
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR, f"{path} is {sr} Hz, expected {SR}"
    return x.mean(axis=1) if x.ndim == 2 else x


def make_noise(length, tilt=1.0, rng=None):
    """Stationary noise with a spectral tilt, standard deviation 1.

    tilt = 0 -> white (flat)   1 -> pink (-3 dB/octave)   2 -> brown (-6 dB/octave, a rumble)
    """
    rng = np.random.default_rng(0) if rng is None else rng
    spec = np.fft.rfft(rng.standard_normal(length))
    f = np.fft.rfftfreq(length)
    spec[0] = 0
    spec[1:] /= f[1:] ** (tilt / 2)
    n = np.fft.irfft(spec, length)
    return (n / n.std()).astype(np.float32)


def level_db(x):
    """Level in dB: 20 * log10(RMS)."""
    return 20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


def mix_at_snr(clean, noise, snr_db):
    """Scale the noise so that clean + noise has the requested SNR. Returns (noisy, scaled_noise)."""
    scaled = noise * np.float32(10 ** ((level_db(clean) - snr_db - level_db(noise)) / 20))
    return clean + scaled, scaled


def save_wav(path, x):
    """Save a 16 kHz, 16-bit WAV; scales down only if it would clip."""
    x = np.asarray(x, dtype=np.float32)
    peak = np.abs(x).max()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x * (0.99 / peak if peak > 0.99 else 1.0), SR, subtype="PCM_16")


# ------------------------------------------------------------------ spectrograms (Week 2)
def stft(x):
    """[T] or [B, T] float tensor -> complex [N_FREQ, frames] or [B, N_FREQ, frames]."""
    return torch.stft(x, N_FFT, HOP, window=torch.hann_window(N_FFT, device=x.device), return_complex=True)


def istft(X, length):
    """Inverse of stft(): complex [..., N_FREQ, frames] -> float [..., length]."""
    return torch.istft(X, N_FFT, HOP, window=torch.hann_window(N_FFT, device=X.device), length=length)


def ideal_ratio_mask(S, N):
    """The Week 2 IRM: sqrt(|S|^2 / (|S|^2 + |N|^2)), between 0 and 1."""
    s2, n2 = S.abs() ** 2, N.abs() ** 2
    return torch.sqrt(s2 / (s2 + n2 + 1e-12))


def show_tf(ax, img, title, vmin=None, vmax=None, cmap="magma"):
    """Draw a [freq, frames] image with time (s) and frequency (Hz) axes."""
    img = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    vmax = img.max() if vmax is None else vmax
    vmin = vmax - 80 if vmin is None else vmin
    im = ax.imshow(img, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[0, img.shape[1] * HOP / SR, 0, SR / 2])
    ax.set(title=title, xlabel="time (s)", ylabel="Hz")
    return im


# ------------------------------------------------------------------ metrics (Week 2)
def si_sdr(ref, est, eps=1e-8):
    ref = ref.astype(np.float64) - ref.mean()
    est = est.astype(np.float64) - est.mean()
    target = np.dot(est, ref) / np.dot(ref, ref) * ref
    return 10 * np.log10((np.sum(target ** 2) + eps) / (np.sum((est - target) ** 2) + eps))


def evaluate(ref, est):
    """{"PESQ", "STOI", "SI-SDR"} of a 1-D numpy estimate against the clean reference."""
    from pesq import pesq
    from pystoi import stoi
    return {"PESQ": pesq(SR, ref, est, "wb"), "STOI": stoi(ref, est, SR, extended=False), "SI-SDR": si_sdr(ref, est)}


def print_table(rows):
    """Print a list of dicts as a table."""
    keys = list(rows[0])
    print("  " + "".join(f"{k:>10}" for k in keys))
    for r in rows:
        print("  " + "".join(f"{r[k]:>10}" if isinstance(r[k], str) else f"{r[k]:>10.2f}" for k in keys))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
