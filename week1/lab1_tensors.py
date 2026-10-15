"""
Week 1 - Lab 1: Audio as tensors                          run:  python lab1_tensors.py

Digital audio is just a long row of numbers (samples). At a sample rate of 16 kHz,
one second of sound = 16000 numbers.

Every speech model in this course reads a batch of waveforms shaped [B, C, T]:
    B = batch     how many clips at once     e.g. 4
    C = channels  1 = mono                   e.g. 1
    T = time      number of samples          e.g. 16000

Your job: fill in every function marked TODO, run this file, and repeat until
every check says PASS.
"""
import soundfile as sf
import torch

from common import OUT, check, section, summary

SR = 16000  # samples per second


def demo():
    """Run the file once before writing anything and read this output - it has all the tools you need."""
    x = torch.tensor([[1., -4., 3.],
                      [2., 5., -6.]])
    print("x =", x.tolist())
    print("shape:", tuple(x.shape), "| dtype:", x.dtype, "| device:", x.device)
    print("x[:, :2]                :", x[:, :2].tolist(), "  <- first two columns")
    print("x[:, -1]                :", x[:, -1].tolist(), "  <- last column")
    print("x.abs()                 :", x.abs().tolist())
    print("x.abs().amax(dim=1)     :", x.abs().amax(dim=1).tolist(), "  <- biggest |value| in each row")
    print("x.abs().amax(dim=1, keepdim=True).shape :", tuple(x.abs().amax(dim=1, keepdim=True).shape))
    print("x / torch.tensor([[4.], [2.]])          :", (x / torch.tensor([[4.], [2.]])).tolist(),
          "  <- [2,3] / [2,1]: row 0 divided by 4, row 1 by 2")
    print("x.reshape(-1)           :", x.reshape(-1).tolist(), "  <- -1 means 'work it out yourself'")
    print("x.reshape(3, 2)         :", x.reshape(3, 2).tolist())
    print("torch.linspace(0, 1, 5) :", torch.linspace(0, 1, 5).tolist())
    print("... .flip(0)            :", torch.linspace(0, 1, 5).flip(0).tolist())
    print("torch.arange(6)[-2:]    :", torch.arange(6)[-2:].tolist(), "  <- last 2. Careful: [-0:] is EVERYTHING")


# ---------------------------------------------------------------- Exercise 1.1
def make_tone(freq, sr=SR, duration=1.0, amp=0.5, fade=0.01):
    """Return a sine tone of shape [int(sr * duration)] with a short fade at both ends.

        x(t) = amp * sin(2 * pi * freq * t),    t in seconds

    A tone that starts or stops abruptly makes an audible click. To avoid it, the first
    F = int(sr * fade) samples are multiplied by a ramp going linearly from 0 to 1, and the
    last F samples by a ramp going from 1 back to 0. If F is 0, return the plain sine.

    Hints:  torch.arange(N) / sr           -> the time (s) of every sample
            ramp = torch.linspace(0, 1, F)  ;  ramp.flip(0) goes from 1 to 0
            env = torch.ones(N);  env[:F] = ... ;  env[-F:] = ... ;  return x * env
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.2
def chop(x, seg_len):
    """Cut a long 1-D signal into back-to-back segments -> shape [B, 1, seg_len].

    B = len(x) // seg_len. Samples left over at the end are dropped.
    Example: chop(torch.arange(10), 3) -> [[[0, 1, 2]], [[3, 4, 5]], [[6, 7, 8]]]   (9 is dropped)

    This is how long recordings become a training batch.
    Hint: keep the first B * seg_len samples -> .reshape(B, seg_len) -> .unsqueeze(1)
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.3
def peak_normalize(x, peak=0.9):
    """Scale EACH clip so that its largest absolute sample equals `peak`.

    x has shape [B, C, T]. Take the max of |x| over the TIME axis only -> shape [B, C, 1],
    then divide; broadcasting does the rest. The output has the same shape as x.
    Hint: x.abs().amax(dim=-1, keepdim=True)
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    section("Demo")
    demo()

    section("Exercise 1.1  make_tone")
    check("shape is [16000]", lambda: tuple(make_tone(440).shape) == (SR,), "N = int(sr * duration); t = torch.arange(N) / sr")
    check("0.25 s gives 4000 samples", lambda: tuple(make_tone(440, duration=0.25).shape) == (4000,))
    check("peak is 0.5", lambda: abs(make_tone(440).abs().max().item() - 0.5) < 1e-3, "multiply by amp")
    check("the middle is a plain sine", lambda: abs(make_tone(440)[2000].item() - 0.5 * torch.sin(torch.tensor(2 * torch.pi * 440 * 2000 / SR)).item()) < 1e-3,
          "x = amp * sin(2 * pi * freq * t)")
    check("fades in: first 5 ms is quieter", lambda: make_tone(440)[:80].abs().max().item() < 0.26, "multiply the first F samples by torch.linspace(0, 1, F)")
    check("fades out: last 5 ms is quieter", lambda: make_tone(440)[-80:].abs().max().item() < 0.26, "the last F samples use the ramp flipped")
    check("fade=0 gives the plain sine", lambda: make_tone(440, fade=0)[:80].abs().max().item() > 0.49,
          "if F == 0 skip the ramp (careful: env[-0:] is the WHOLE array)")

    section("Exercise 1.2  chop")
    sig = torch.arange(40000.)  # stands in for 2.5 s of audio at 16 kHz
    check("chop(arange(10), 3) has shape [3, 1, 3]", lambda: tuple(chop(torch.arange(10.), 3).shape) == (3, 1, 3),
          "B = len(x) // seg_len, and drop the leftover samples")
    check("values stay in time order", lambda: chop(torch.arange(10.), 3)[:, 0].tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    check("2.5 s in 0.5 s segments -> [5, 1, 8000]", lambda: tuple(chop(sig, 8000).shape) == (5, 1, 8000))
    check("segment 3 starts at sample 24000", lambda: torch.equal(chop(sig, 8000)[3, 0, :4], sig[24000:24004]))

    section("Exercise 1.3  peak_normalize")
    torch.manual_seed(0)
    batch = torch.randn(4, 1, SR) * torch.tensor([0.1, 0.3, 0.6, 1.2]).view(4, 1, 1)  # 4 clips, 4 volumes
    print("  peaks before:", [round(v, 2) for v in batch.abs().amax(dim=-1).flatten().tolist()])
    check("shape is unchanged [4, 1, 16000]", lambda: tuple(peak_normalize(batch).shape) == (4, 1, SR))
    check("every clip now peaks at 0.9", lambda: torch.allclose(peak_normalize(batch).abs().amax(dim=-1), torch.full((4, 1), 0.9)),
          "take the max over time only (dim=-1, keepdim=True), not over the whole batch")
    check("only the volume changed, not the waveform", lambda: torch.allclose(
        peak_normalize(batch) / peak_normalize(batch).abs().amax(-1, keepdim=True), batch / batch.abs().amax(-1, keepdim=True), atol=1e-6))

    section("GPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("  device:", device)
    try:
        print("  peaks after, computed on", device, ":", [round(v, 2) for v in peak_normalize(batch.to(device)).abs().amax(dim=-1).flatten().tolist()])
        print("  (same code, different device - that's all it takes)")
    except NotImplementedError:
        print("  finish Exercise 1.3 first")

    try:
        gap = torch.zeros(int(0.15 * SR))
        hard = torch.cat([torch.cat([make_tone(880, duration=0.11, fade=0), gap]) for _ in range(5)])
        soft = torch.cat([torch.cat([make_tone(880, duration=0.11, fade=0.01), gap]) for _ in range(5)])
        sf.write(OUT / "lab1_beeps_hard.wav", hard.numpy(), SR)
        sf.write(OUT / "lab1_beeps_fade.wav", soft.numpy(), SR)
        print("\n  saved out/lab1_beeps_hard.wav and out/lab1_beeps_fade.wav -> listen to both (question Q1)")
    except NotImplementedError:
        pass

    summary()
