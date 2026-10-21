"""
Week 3 - Lab 1: Training data that never repeats                run:  python lab1_data.py

This week a network learns to do what the oracle did in Week 2 - but from the NOISY input alone.
To learn that, it needs many thousands of (noisy, clean) examples.

We have only 117 clean clips (data/train/, about 10 minutes of speech). Instead of saving a fixed
set of noisy files like in Week 1, every example is mixed ON THE FLY:

    a random clip  ->  a random 2 s piece of it  ->  random noise  ->  a random SNR  ->  mix

so the network practically never sees the same example twice. This is called dynamic mixing.

PyTorch organizes this with two classes:
    Dataset     dataset[i] returns example number i
    DataLoader  asks the Dataset for many examples and stacks them into a batch

The network will not look at the waveform but at a FEATURE of the spectrogram: the log power,
normalized so that playing the same sentence louder gives the same numbers.

Your job: fill in the TODOs, then run this file.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (DEVICE, OUT, SR, TRAIN_DIR, check, ideal_ratio_mask, level_db, load_audio, make_noise,
                    mix_at_snr, save_wav, section, show_tf, stft, summary)

SEG = 2 * SR             # every training example is 2 s long
SNR_RANGE = (-5, 15)     # training SNRs are drawn uniformly from this range (dB)


# ---------------------------------------------------------------- Exercise 1.1
def random_crop(x, length, rng):
    """Return a random piece of x that is exactly `length` samples long.

    - if x is longer:  pick a random start in [0, len(x) - length] with rng.integers(...)
                       and return x[start:start + length]
    - otherwise:       return x padded with zeros at the end:  np.pad(x, (0, length - len(x)))
    """
    # TODO
    raise NotImplementedError


class NoisyMixDataset(torch.utils.data.Dataset):
    """Endless supply of (noisy, clean) pairs, mixed on the fly from the clean clips in data/train/."""

    def __init__(self, files, seg=SEG, snr_range=SNR_RANGE, size=100_000, seed=0):
        self.clips = [load_audio(f) for f in files]   # all clips in memory (~40 MB)
        self.seg, self.snr_range, self.size, self.seed = seg, snr_range, size, seed

    def __len__(self):
        return self.size   # "virtual" size: there is no fixed list of examples

    # ---------------------------------------------------------------- Exercise 1.2
    def __getitem__(self, i):
        """Example number i: returns (noisy, clean), two float32 tensors of shape [seg].

        rng = np.random.default_rng(...) is given: the same i always gives the same example.
          1. clean = random_crop(one random clip, self.seg, rng)     hint: rng.integers(len(self.clips))
          2. noise = make_noise(self.seg, tilt=rng.uniform(0, 2), rng=rng)    white ... pink ... brown
          3. snr   = rng.uniform(*self.snr_range)
          4. noisy, _ = mix_at_snr(clean, noise, snr)
          5. return torch.from_numpy(noisy), torch.from_numpy(clean)
        """
        rng = np.random.default_rng([self.seed, i])
        # TODO
        raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.3
def log_spec_features(Y):
    """What the network sees: the log power of the noisy STFT, normalized per example.

    Y is complex, shape [B, F, T].
      1. f = torch.log(|Y|^2 + 1e-8)                       (|Y| is Y.abs())
      2. subtract the mean and divide by the standard deviation of EACH example,
         taken over frequency AND time together:  dim=(1, 2), keepdim=True   (add 1e-5 to the std)
    Returns a real tensor [B, F, T].
    """
    # TODO
    raise NotImplementedError


def train_files():
    """(Given) The training clips, sorted."""
    return sorted(TRAIN_DIR.glob("*.flac"))


if __name__ == "__main__":
    files = train_files()
    print(f"{len(files)} clean training clips in {TRAIN_DIR}, "
          f"{sum(len(load_audio(f)) for f in files) / SR / 60:.1f} minutes in total | device: {DEVICE}")

    section("Exercise 1.1  random_crop")
    rng = np.random.default_rng(0)
    x = np.arange(10, dtype=np.float32)
    check("crop of a long signal has the right length", lambda: len(random_crop(np.arange(100.), 30, rng)) == 30,
          "x[start:start + length]")
    check("the crop is a continuous piece of x", lambda: np.all(np.diff(random_crop(np.arange(100.), 30, rng)) == 1))
    check("different starts appear", lambda: len({random_crop(np.arange(100.), 30, rng)[0] for _ in range(50)}) > 10,
          "start = rng.integers(0, len(x) - length + 1)")
    check("a short signal is padded with zeros at the end", lambda: random_crop(x, 13, rng).tolist() == list(range(10)) + [0, 0, 0],
          "np.pad(x, (0, length - len(x)))")

    section("Exercise 1.2  the dataset")
    ds = NoisyMixDataset(files)
    try:
        noisy, clean = ds[0]
    except NotImplementedError:
        check("__getitem__ is written", lambda: False, "finish the TODOs in Exercises 1.1 and 1.2")
        summary()
        raise SystemExit(1)
    check("returns two float32 tensors of 2 s", lambda: noisy.shape == clean.shape == (SEG,) and noisy.dtype == torch.float32,
          "torch.from_numpy(...) of float32 arrays, both seg samples long")
    check("the same index gives the same example", lambda: torch.equal(ds[7][0], ds[7][0]))
    check("different indices give different examples", lambda: not torch.equal(ds[7][0], ds[8][0]))
    snrs = [level_db(c.numpy()) - level_db((n - c).numpy()) for n, c in (ds[i] for i in range(200))]
    print(f"  SNR of 200 examples: min {min(snrs):.1f} dB, mean {np.mean(snrs):.1f} dB, max {max(snrs):.1f} dB")
    check("SNRs stay inside -5 ... 15 dB and cover the range", lambda: min(snrs) > -5.01 and max(snrs) < 15.01 and max(snrs) - min(snrs) > 15,
          "snr = rng.uniform(*self.snr_range)")

    section("A batch from the DataLoader")
    loader = torch.utils.data.DataLoader(ds, batch_size=16)
    noisy_b, clean_b = next(iter(loader))
    print(f"  noisy batch {tuple(noisy_b.shape)}, clean batch {tuple(clean_b.shape)}")
    check("batch shape is [16, 32000]", lambda: tuple(noisy_b.shape) == (16, SEG))

    section("Exercise 1.3  features")
    Y = stft(noisy_b)
    try:
        feats = log_spec_features(Y)
    except NotImplementedError:
        check("log_spec_features() is written", lambda: False, "finish the TODO in Exercise 1.3")
        summary()
        raise SystemExit(1)
    print(f"  STFT {tuple(Y.shape)} complex  ->  features {tuple(feats.shape)} real")
    check("real, same shape as the STFT", lambda: not feats.is_complex() and feats.shape == Y.shape, "use Y.abs()")
    check("each example has mean 0 and std 1", lambda: torch.allclose(feats.mean(dim=(1, 2)), torch.zeros(16), atol=1e-4)
          and torch.allclose(feats.std(dim=(1, 2)), torch.ones(16), atol=1e-3), "dim=(1, 2), keepdim=True")
    check("10x louder input gives (almost) the same features", lambda: (log_spec_features(stft(10 * noisy_b)) - feats).abs().mean() < 1e-3,
          "normalize per example: subtract its mean, divide by its std")

    section("What the network gets, and what it should answer")
    target = ideal_ratio_mask(stft(clean_b), stft(noisy_b - clean_b))
    k = 0
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    show_tf(axes[0], 10 * torch.log10(stft(clean_b[k]).abs() ** 2 + 1e-10), "clean (not given to the network)")
    show_tf(axes[1], feats[k], "INPUT: normalized log power of noisy", vmin=-3, vmax=3)
    show_tf(axes[2], target[k], "TARGET: ideal ratio mask", vmin=0, vmax=1, cmap="gray")
    fig.tight_layout(); fig.savefig(OUT / "lab1_example.png", dpi=110); plt.close(fig)
    for j in range(3):
        save_wav(OUT / "lab1_examples" / f"noisy_{j}.wav", noisy_b[j].numpy())
    print("  saved out/lab1_example.png and out/lab1_examples/noisy_*.wav - listen to a few (question Q1)")
    summary()
