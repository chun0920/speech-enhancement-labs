"""Helpers shared by all labs this week - no need to edit this file."""
from math import gcd
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen on the server -> figures are written to PNG files

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

OUT = Path("out")
OUT.mkdir(exist_ok=True)
DATA = Path("data")
SR = 16000
N_FFT, HOP = 512, 128  # STFT settings for the rest of the course: 32 ms Hann frames, 8 ms hop
SAMPLE = DATA / "speech_sample.wav"  # LibriSpeech clip shipped with the labs (see data/README.txt)

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


# ------------------------------------------------------------- audio helpers (from Week 1)
def load_speech(path=None):
    """Clean speech as mono float32 at 16 kHz. Default: the LibriSpeech sample in data/."""
    x, sr = sf.read(path or SAMPLE, dtype="float32")
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x.astype(np.float32)


def make_noise(length, kind="pink", seed=0):
    """Reproducible noise with standard deviation 1: kind = 'white', 'pink', or the path of a WAV file."""
    rng = np.random.default_rng(seed)
    if kind == "white":
        n = rng.standard_normal(length)
    elif kind == "pink":  # equal energy per octave - sounds like steady rain
        spec = np.fft.rfft(rng.standard_normal(length))
        f = np.fft.rfftfreq(length)
        spec[0] = 0
        spec[1:] /= np.sqrt(f[1:])
        n = np.fft.irfft(spec, length)
    else:
        n = load_speech(kind)  # any WAV -> mono 16 kHz
        n = np.tile(n, int(np.ceil(length / len(n))) + 1)
        start = rng.integers(0, len(n) - length + 1)
        n = n[start:start + length]
    return (n / n.std()).astype(np.float32)


def level_db(x):
    """Level in dB: 20 * log10(RMS)."""
    return 20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


def mix_at_snr(clean, noise, snr_db):
    """Scale the noise so that clean + noise has the requested SNR. Returns (noisy, scaled_noise)."""
    scaled = noise * np.float32(10 ** ((level_db(clean) - snr_db - level_db(noise)) / 20))
    return clean + scaled, scaled


def show_tf(ax, img, title, hop=HOP, vmin=None, vmax=None, cmap="magma"):
    """Draw a [freq, frames] image (e.g. a spectrogram in dB) with time (s) and frequency (Hz) axes."""
    img = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    if vmax is None:
        vmax = img.max()
    if vmin is None:
        vmin = vmax - 80  # show the top 80 dB
    im = ax.imshow(img, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[0, img.shape[1] * hop / SR, 0, SR / 2])
    ax.set(title=title, xlabel="time (s)", ylabel="Hz")
    return im


def save_wav(path, x):
    """Save a 16 kHz, 16-bit WAV; scales down only if it would clip."""
    x = np.asarray(x, dtype=np.float32)
    peak = np.abs(x).max()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x * (0.99 / peak if peak > 0.99 else 1.0), SR, subtype="PCM_16")
