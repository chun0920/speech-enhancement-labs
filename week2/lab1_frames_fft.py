"""
Week 2 - Lab 1: Frames, windows and the FFT                run:  python lab1_frames_fft.py

Speech never sits still: a vowel lasts about 100 ms, a "t" only a few ms. So we never take one
Fourier transform of a whole recording. Instead we

    1. cut the signal into short, overlapping FRAMES (20-40 ms: short enough that the sound is
       roughly steady inside one frame),
    2. multiply every frame by a smooth WINDOW that fades it in and out,
    3. take the FFT of every frame.

Put the results side by side and you have a spectrogram (Lab 2).

    frame_len   samples per frame       25 ms at 16 kHz = 400 samples
    hop         step between frames     10 ms at 16 kHz = 160 samples
    n_fft       FFT size                FFT bin k sits at  k * sr / n_fft  Hz

Your job: fill in the TODOs, then run this file.
"""
import matplotlib.pyplot as plt
import torch

from common import OUT, SR, check, section, summary

# ---------------------------------------------------------------- Exercise 1.1
# Work this out on paper FIRST, then type your number:
#   how many complete 25 ms frames, with a 10 ms hop, fit in 10 s of 16 kHz audio?
#   (frame_len = 400 samples, hop = 160 samples, T = 160000 samples)
MY_FRAME_COUNT = None  # TODO: replace None with your number


def num_frames(T, frame_len, hop):
    """Number of complete frames in a signal of T samples (assume T >= frame_len).

    The first frame starts at sample 0, each next one starts `hop` samples later,
    and the last one must end inside the signal.
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.2
def frame_signal(x, frame_len, hop):
    """Cut the 1-D tensor x into overlapping frames -> tensor [num_frames, frame_len].

    Frame i is x[i * hop : i * hop + frame_len].
    Hint: x.unfold(0, frame_len, hop) does exactly this. (A for-loop + torch.stack works too.)
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.3
def spectrum_db(frame, window, n_fft=512, sr=SR):
    """Magnitude spectrum of ONE frame in dB. Returns (freqs_hz, mag_db), each of shape [n_fft // 2 + 1].

    1. X = torch.fft.rfft(frame * window, n=n_fft)      (n=n_fft pads the frame with zeros)
    2. mag_db = 20 * log10(|X| + 1e-8)                  (|X| is X.abs())
    3. freqs_hz: bin k sits at k * sr / n_fft Hz        hint: torch.arange(n_fft // 2 + 1)
    """
    # TODO
    raise NotImplementedError


def tone(freqs, n, amp=0.5):
    """(Given) Sum of sines at the given frequencies, n samples long."""
    t = torch.arange(n) / SR
    return sum(amp * torch.sin(2 * torch.pi * f * t) for f in freqs)


def far_leakage_db(freqs, mag_db):
    """(Given) Loudest level between 3 and 8 kHz, relative to the peak. Lower = less leakage."""
    return (mag_db[freqs > 3000].max() - mag_db.max()).item()


if __name__ == "__main__":
    section("Exercise 1.1  how many frames?")
    print(f"  your paper answer: {MY_FRAME_COUNT}")
    check("num_frames(160000, 400, 160) is correct", lambda: num_frames(160000, 400, 160) == 998,
          "the first frame uses 400 samples; every extra frame needs 160 more")
    check("num_frames works for any length", lambda: [num_frames(T, 400, 160) for T in (400, 559, 560, 12345)] == [1, 1, 2, 75])
    check("your paper answer agrees", lambda: MY_FRAME_COUNT == num_frames(160000, 400, 160),
          "work it out by hand - this is the one check you can't copy from a formula in the code")

    section("Exercise 1.2  frame_signal")
    x = torch.arange(160000.)  # stands in for 10 s of audio
    check("10 s -> shape [998, 400]", lambda: tuple(frame_signal(x, 400, 160).shape) == (998, 400), "x.unfold(0, frame_len, hop)")
    check("frame 2 is x[320:720]", lambda: torch.equal(frame_signal(x, 400, 160)[2], x[320:720]))
    check("frame count agrees with num_frames()", lambda: len(frame_signal(torch.zeros(12345), 400, 160)) == num_frames(12345, 400, 160))

    section("Exercise 1.3  spectrum of one frame")
    hann = torch.hann_window(400)
    try:
        freqs, mag = spectrum_db(tone([1000], 400), hann)
    except NotImplementedError:
        check("spectrum_db() is written", lambda: False, "finish the TODO in Exercise 1.3")
        summary()
        raise SystemExit(1)
    check("257 frequency bins for n_fft = 512", lambda: freqs.shape == (257,) and mag.shape == (257,), "rfft gives n_fft // 2 + 1 bins")
    check("bin 32 sits at 1000 Hz", lambda: abs(freqs[32].item() - 1000) < 1e-3, "k * sr / n_fft")
    check("a 1000 Hz tone peaks at 1000 Hz", lambda: abs(freqs[mag.argmax()].item() - 1000) < 1e-3)
    check("mag_db is in dB (Hann-windowed peak around +34 dB)", lambda: 32 < mag.max().item() < 36, "20 * log10(|X| + 1e-8) of the WINDOWED frame")

    section("Why the window? A 1030 Hz tone, cut out two ways")
    frame = tone([1030], 400)
    f_rect, m_rect = spectrum_db(frame, torch.ones(400), n_fft=4096)
    f_hann, m_hann = spectrum_db(frame, hann, n_fft=4096)
    leak_rect, leak_hann = far_leakage_db(f_rect, m_rect), far_leakage_db(f_hann, m_hann)
    print(f"  loudest leakage above 3 kHz:  rectangular {leak_rect:6.1f} dB   Hann {leak_hann:6.1f} dB   (relative to the peak)")
    check("Hann leaks at least 30 dB less than rectangular", lambda: leak_hann < leak_rect - 30)
    plt.figure(figsize=(9, 3.5))
    plt.plot(f_rect, m_rect - m_rect.max(), lw=1, label="rectangular (just cut it out)")
    plt.plot(f_hann, m_hann - m_hann.max(), lw=1, label="Hann window")
    plt.ylim(-120, 5); plt.xlabel("Hz"); plt.ylabel("dB re. peak"); plt.title("One 25 ms frame of a 1030 Hz tone")
    plt.legend(); plt.tight_layout(); plt.savefig(OUT / "lab1_windows.png", dpi=110); plt.close()

    section("Frame length vs frequency detail: tones at 1000 Hz and 1040 Hz")
    plt.figure(figsize=(9, 3.5))
    two_tones = tone([1000, 1040], 2 * SR)
    for ms in (25, 100):
        n = SR * ms // 1000
        frames = frame_signal(two_tones, n, n // 4)
        power = torch.stack([10 ** (spectrum_db(fr, torch.hann_window(n), n_fft=16384)[1] / 10) for fr in frames]).mean(dim=0)
        f = spectrum_db(frames[0], torch.hann_window(n), n_fft=16384)[0]
        m = 10 * torch.log10(power / power.max())  # average over all frames, in dB re. the peak
        dip = m[(f - 1020).abs().argmin()].item()
        print(f"  {ms:3d} ms frame: level halfway between the tones = {dip:6.1f} dB"
              + ("   -> two separate peaks" if dip < -6 else "   -> one merged bump"))
        plt.plot(f, m, label=f"{ms} ms frame")
    for fr in (1000, 1040):
        plt.axvline(fr, color="r", ls=":", lw=1)
    plt.xlim(850, 1200); plt.ylim(-60, 3); plt.xlabel("Hz"); plt.ylabel("dB re. peak"); plt.legend()
    plt.title("Two tones 40 Hz apart (red), averaged over all frames"); plt.tight_layout(); plt.savefig(OUT / "lab1_resolution.png", dpi=110); plt.close()
    print("\n  saved out/lab1_windows.png and out/lab1_resolution.png (questions Q1, Q2)")
    summary()
