"""
Week 1 - Lab 3: What is inside a WAV file?                run:  python lab3_audio_io.py

Input: data/clean_speech.wav (from your mentor, or a recording of your own voice).
Any other file works too:   python lab3_audio_io.py --wav data/my_voice.wav
If the file is missing, a synthetic "speech-like" placeholder is generated so you can continue.

SHAPE TRAP: soundfile gives (time,) for mono and (time, channels) for stereo - time FIRST.
PyTorch models want channels first: [C, T]. When in doubt, print the shape.

Your job: fill in the TODOs, then run this file.
"""
import argparse
from math import gcd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, welch

from common import DATA, OUT, check, section, summary

TARGET_SR = 16000


def make_placeholder(path):
    """(Given) Fake 'speech' - 48 kHz, stereo, 16-bit - so the lab works without a recording."""
    print(f"  {path} not found -> generating a synthetic placeholder (48 kHz, stereo)")
    path.parent.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    sr0 = 48000
    t = np.arange(int(2.5 * sr0)) / sr0
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.7 * t)                       # pitch that wobbles, like a voice
    phase = 2 * np.pi * np.cumsum(f0) / sr0
    voiced = sum(np.sin(k * phase) / k for k in range(1, 30))         # harmonics -> "vowels"
    spec = np.fft.rfft(rng.standard_normal(len(t)))
    spec[np.fft.rfftfreq(len(t), 1 / sr0) < 4000] = 0
    hiss = np.fft.irfft(spec, len(t))                                 # noise above 4 kHz -> "s" / "sh"
    vowel_on = np.clip(np.sin(2 * np.pi * 2.0 * t), 0, None)
    hiss_on = np.clip(-np.sin(2 * np.pi * 2.0 * t), 0, None) ** 2
    mono = 0.35 * voiced * vowel_on / np.abs(voiced).max() + 0.08 * hiss * hiss_on / hiss.std()
    sf.write(path, np.stack([mono, 0.7 * mono], axis=1), sr0, subtype="PCM_16")


# ---------------------------------------------------------------- Exercise 3.1
def to_channels_first(audio):
    """Turn the array from sf.read into a float32 array shaped [C, T] (channels first).

        mono   (T,)    -> (1, T)       hint: audio[None, :]   or   audio.reshape(1, -1)
        stereo (T, C)  -> (C, T)       hint: audio.T
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 3.2
def int16_to_float(x_int):
    """Convert raw 16-bit samples to float32 in [-1, 1).

    Inside a 16-bit WAV every sample is an integer from -32768 to 32767;
    sf.read(path, dtype="int16") hands you those integers unchanged.
    Models - and sf.read(path, dtype="float32") - use floats in [-1, 1): divide by 32768.
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 3.3
def resample(x, sr, target_sr):
    """Resample the 1-D signal x from sr to target_sr. Return float32.

    Use scipy.signal.resample_poly(x, up, down), where up/down = target_sr/sr in lowest terms.
    Hint: g = gcd(sr, target_sr);  up = target_sr // g;  down = sr // g
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 3.4
def safe_save(path, x, sr, peak=0.99):
    """Save the 1-D float signal x as a 16-bit WAV WITHOUT clipping. Return the gain you used.

    A 16-bit WAV can only store values in [-1, 1]; anything louder is cut flat - that is
    clipping, and it sounds harsh. So:
      - if max(|x|) > peak:  gain = peak / max(|x|)   (the loudest sample becomes `peak`)
      - otherwise:           gain = 1.0                (leave it alone)
    then  sf.write(path, x * gain, sr, subtype="PCM_16")  and return gain.
    """
    # TODO
    raise NotImplementedError


def quantization_snr(x, path, subtype):
    """(Given) Save x with the given sample format, read it back, return the SNR of the round trip in dB."""
    sf.write(path, x, TARGET_SR, subtype=subtype)
    back, _ = sf.read(path, dtype="float32")
    return 10 * np.log10(np.sum(x ** 2) / np.sum((back - x) ** 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=str(DATA / "clean_speech.wav"))
    args = ap.parse_args()
    wav = Path(args.wav)
    if not wav.exists():
        make_placeholder(wav)

    section("File info")
    info = sf.info(wav)
    print(info)
    audio, sr = sf.read(wav, dtype="float32")
    print("sf.read gives:", type(audio).__name__, audio.dtype, "shape", audio.shape, "| sample rate", sr)

    section("Exercise 3.1  channels first")
    try:
        x = to_channels_first(audio)
        print(f"  {audio.shape}  ->  {x.shape}")
    except NotImplementedError:
        check("to_channels_first() is written", lambda: False, "finish the TODO in Exercise 3.1")
        summary()
        raise SystemExit(1)
    check("shape[0] = number of channels", lambda: x.ndim == 2 and x.shape[0] == info.channels, "mono -> audio[None, :],  stereo -> audio.T")
    check("shape[1] = number of samples", lambda: x.shape[1] == info.frames)
    check("dtype is float32", lambda: x.dtype == np.float32)
    check("row 0 is the first (left) channel", lambda: np.array_equal(x[0], audio if audio.ndim == 1 else audio[:, 0]))
    speech = x.mean(axis=0)  # average the channels -> mono
    print(f"  {x.shape[0]} channel(s), {x.shape[1]} samples = {x.shape[1] / sr:.2f} s")

    plt.figure(figsize=(12, 2.5))
    plt.plot(np.arange(len(speech)) / sr, speech, lw=0.5)
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.title(f"{wav.name} ({sr} Hz, mono mix)")
    plt.tight_layout(); plt.savefig(OUT / "lab3_waveform.png", dpi=110); plt.close()

    section("Exercise 3.2  the integers inside the file")
    raw, _ = sf.read(wav, dtype="int16")
    print(f"  raw samples: dtype {raw.dtype}, from {raw.min()} to {raw.max()}")
    check("[-32768, -16384, 0, 16384] -> [-1, -0.5, 0, 0.5]", lambda: int16_to_float(np.array([-32768, -16384, 0, 16384], dtype=np.int16)).tolist() == [-1, -0.5, 0, 0.5],
          "divide by 32768 (not 32767, not 65536)")
    check("dtype is float32", lambda: int16_to_float(raw).dtype == np.float32, "add .astype(np.float32)")
    check("same numbers as sf.read(dtype='float32')", lambda: np.abs(int16_to_float(raw) - audio).max() < 1e-4)
    print(f"  forget this step and a model expecting [-1, 1] gets numbers up to {np.abs(raw.astype(np.int32)).max()}")

    section("Exercise 3.3  resample")
    try:
        speech16 = resample(speech, sr, TARGET_SR)
    except NotImplementedError:
        check("resample() is written", lambda: False, "finish the TODO in Exercise 3.3")
        summary()
        raise SystemExit(1)
    print(f"  {sr} Hz: {len(speech)} samples  ->  {TARGET_SR} Hz: {len(speech16)} samples")
    check("length matches 16 kHz", lambda: abs(len(speech16) - round(len(speech) * TARGET_SR / sr)) <= 1, "up and down may be swapped")
    check("dtype is float32", lambda: speech16.dtype == np.float32, "add .astype(np.float32)")
    check("16 kHz -> 8 kHz halves the length", lambda: abs(len(resample(speech16, TARGET_SR, 8000)) - len(speech16) / 2) <= 1)

    section("Exercise 3.4  save without clipping")
    try:
        g_quiet = safe_save(OUT / "lab3_quiet.wav", 0.5 * speech16 / np.abs(speech16).max(), TARGET_SR)
        g_loud = safe_save(OUT / "lab3_loud.wav", 5 * speech16 / np.abs(speech16).max(), TARGET_SR)  # peak 5: far too loud
    except NotImplementedError:
        check("safe_save() is written", lambda: False, "finish the TODO in Exercise 3.4")
        summary()
        raise SystemExit(1)
    loud_back, loud_sr = sf.read(OUT / "lab3_loud.wav", dtype="float32")
    check("a quiet signal is left alone (gain 1.0)", lambda: g_quiet == 1.0)
    check("a signal peaking at 5 gets gain 0.99 / 5", lambda: abs(g_loud - 0.99 / 5) < 1e-6, "gain = peak / max(|x|)")
    check("no clipping in the file", lambda: np.abs(loud_back).max() <= 0.99 + 1e-4)
    check("16-bit WAV at 16 kHz", lambda: sf.info(OUT / "lab3_loud.wav").subtype == "PCM_16" and loud_sr == TARGET_SR, 'subtype="PCM_16"')
    gain = safe_save(OUT / "speech_16k.wav", speech16, TARGET_SR)
    print(f"  saved out/speech_16k.wav (gain {gain:.3f}) - Lab 4 uses this file")
    speech16 = speech16 * gain

    section("Telephone quality: 16 kHz -> 8 kHz -> 16 kHz")
    phone = resample(resample(speech16, TARGET_SR, 8000), 8000, TARGET_SR)[:len(speech16)]
    safe_save(OUT / "phone_quality.wav", phone, TARGET_SR)
    f, p_orig = welch(speech16, TARGET_SR, nperseg=1024)
    _, p_phone = welch(phone, TARGET_SR, nperseg=1024)
    hi = f > 4500
    kept = p_phone[hi].sum() / max(p_orig[hi].sum(), 1e-20)
    print(f"  energy above 4.5 kHz that survives the round trip: {100 * kept:.3f} %")
    check("almost nothing above 4.5 kHz survives (< 1 %)", lambda: kept < 0.01)
    plt.figure(figsize=(8, 3.5))
    plt.plot(f, 10 * np.log10(p_orig + 1e-20), label="16 kHz")
    plt.plot(f, 10 * np.log10(p_phone + 1e-20), label="16k -> 8k -> 16k")
    plt.axvline(4000, color="k", ls=":", lw=1)
    plt.xlabel("Frequency (Hz)"); plt.ylabel("power (dB)"); plt.title("Average spectrum"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "lab3_spectrum.png", dpi=110); plt.close()
    print("  saved out/phone_quality.wav and out/lab3_spectrum.png (question Q6)")

    section("Break it on purpose: 8-bit audio")
    for subtype, name in [("PCM_16", "lab3_16bit.wav"), ("PCM_U8", "lab3_8bit.wav")]:
        print(f"  {subtype:<6}: round-trip SNR = {quantization_snr(speech16, OUT / name, subtype):5.1f} dB   -> out/{name}")
    print("  Listen to out/lab3_8bit.wav, then answer Q7.")
    summary()
