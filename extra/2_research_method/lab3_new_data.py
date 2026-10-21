"""
Week 4 - Lab 3: Write it down first, then build the new training data    run:  uv run lab3_new_data.py

Lab 2 told you what the baseline gets wrong. Before training anything new, write down in design.md
what you will change, why, what you expect, and which result would make you drop the idea (README
explains each part). A plan written after the results is not a plan, it is a story.

The change this week is the one most of you proposed in Week 3 (question Q8). The baseline only ever
trained on stationary noise, so it learned "whatever changes over time is speech" - and babble IS
speech. So give it speech-like interference during training: babble made of OTHER training speakers.
Never a test speaker, never one of the six babble talkers of the exam, never the target's own voice.

Only this one thing changes: the same 117 clips, SNR range, network, loss, steps and seeds. Half of
the examples (p_babble = 0.5) get babble of 3 to 8 talkers instead of stationary noise. With
p_babble = 0 the new Dataset must give exactly the Week 3 examples, bit for bit - that is the proof
that nothing else changed.

The last part is the 10-example test: before any real run, the new pipeline must be able to
memorize 10 fixed examples. If it cannot, the code is broken - and you find out in 15 seconds
instead of after a full run.

Your job: fill in design.md (parts 1-4), fill in the TODOs here, then run this file.
"""
import time
from pathlib import Path

import numpy as np
import torch

from common import (DATA, DEVICE, OUT, SNR_RANGE, TEST_DIR, MaskGRU, NoisyMixDataset, all_passed, check,
                    level_db, make_noise, mix_at_snr, random_crop, save_wav, section, summary, train_files, train_step)

P_BABBLE = 0.5        # share of the training examples that get babble instead of stationary noise
TALKERS = (3, 8)      # babble of 3 to 8 people


# ---------------------------------------------------------------- Exercise 3.1
def speaker_of(path):
    """LibriSpeech file names are speaker-chapter-utterance:  'data/train/84-121123-0001.flac' -> '84'.

    Path(path).name is the file name without the folders; .split("-") cuts it at every "-".
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 3.2
def make_babble(by_speaker, exclude, n_talkers, length, rng):
    """n_talkers DIFFERENT people talking at once - never the speaker `exclude` - each at the same level.

    by_speaker is a dict {speaker: [clip, clip, ...]}, the clips are 1-D numpy arrays.
      1. others = sorted(s for s in by_speaker if s != exclude)
      2. for each s in rng.choice(others, size=n_talkers, replace=False):       (replace=False: nobody twice)
             clips = by_speaker[s]
             piece = random_crop(clips[rng.integers(len(clips))], length, rng)
             add piece / (piece.std() + 1e-8) to a running total                 (every talker at level 1)
      3. return the total as float32 (start it with np.zeros(length, dtype=np.float32))
    """
    # TODO
    raise NotImplementedError


class BabbleMixDataset(NoisyMixDataset):
    """Week 3's Dataset with ONE change: with probability p_babble, the noise is babble of other training speakers."""

    def __init__(self, files, p_babble=P_BABBLE, talkers=TALKERS, **kwargs):
        super().__init__(files, **kwargs)        # loads the clips exactly like Week 3
        self.speakers = [speaker_of(f) for f in files]
        self.by_speaker = {}
        for s, clip in zip(self.speakers, self.clips):
            self.by_speaker.setdefault(s, []).append(clip)
        self.p_babble, self.talkers = p_babble, talkers

    # ---------------------------------------------------------------- Exercise 3.3
    def __getitem__(self, i):
        """Example number i: (noisy, clean), two float32 tensors of 2 s. Two random generators are given:

          rng   use it EXACTLY as Week 3 does, for everything Week 3 did (NoisyMixDataset in common.py)
          coin  use it ONLY for the two new decisions, so that rng's sequence is never disturbed

          1. t = rng.integers(len(self.clips));   clean = random_crop(self.clips[t], self.seg, rng)
          2. if coin.uniform() < self.p_babble:
                 n = coin.integers(self.talkers[0], self.talkers[1] + 1)          3 ... 8 talkers
                 noise = make_babble(self.by_speaker, self.speakers[t], n, self.seg, coin)
             else:
                 noise = make_noise(self.seg, tilt=rng.uniform(0, 2), rng=rng)    exactly Week 3
          3. noisy, _ = mix_at_snr(clean, noise, rng.uniform(*self.snr_range))    exactly Week 3
          4. return torch.from_numpy(noisy), torch.from_numpy(clean)
        """
        rng = np.random.default_rng([self.seed, i])
        coin = np.random.default_rng([self.seed, i, 1])
        # TODO
        raise NotImplementedError


def overfit_10(dataset, steps=400, lr=3e-3, seed=0):
    """(Given) Train a fresh MaskGRU on the SAME 10 examples over and over. Returns the losses."""
    torch.manual_seed(seed)
    noisy, clean = (torch.stack(x).to(DEVICE) for x in zip(*(dataset[i] for i in range(10))))
    model = MaskGRU().to(DEVICE).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses, t0 = [], time.time()
    for step in range(steps):
        losses.append(train_step(model, noisy, clean, optimizer))
        if step % 100 == 0 or step == steps - 1:
            print(f"    step {step:4d}   loss {losses[-1]:.4f}   {time.time() - t0:3.0f} s")
    return losses


def speakers_in(listing):
    """(Given) Speaker ids in one of the data/*_list.txt files (first line is a title)."""
    return {line.split("\t")[0].split("-")[0] for line in (DATA / listing).read_text().splitlines()[1:] if line.strip()}


if __name__ == "__main__":
    files = train_files()

    section("No voice of the exam is in the training data")
    train_spk = {f.name.split("-")[0] for f in files}
    test_spk = {f.name.split("-")[0] for f in TEST_DIR.glob("*.flac")}
    babble_spk = speakers_in("babble_list.txt")
    print(f"  training: {len(train_spk)} speakers   test: {len(test_spk)} speakers   exam babble: {len(babble_spk)} talkers")
    check("no test speaker is a training speaker", lambda: not train_spk & test_spk)
    check("no exam-babble talker is a training speaker", lambda: not train_spk & babble_spk)
    check("speaker 1272 (the Week 3 sentence) is not a training speaker", lambda: "1272" not in train_spk)

    section("Exercise 3.1  speaker_of")
    check("speaker_of('data/train/84-121123-0001.flac') == '84'", lambda: speaker_of("data/train/84-121123-0001.flac") == "84")
    check("works on Path objects too", lambda: speaker_of(files[0]) == files[0].name.split("-")[0], "Path(path).name")

    section("Exercise 3.2  make_babble")
    sr, t = 16000, np.arange(16000) / 16000
    tones = {s: [np.sin(2 * np.pi * f * t).astype(np.float32)] for s, f in (("A", 100), ("B", 200), ("C", 300), ("D", 400))}

    def level_at(x, hz):
        return np.abs(np.fft.rfft(x))[int(hz * len(x) / sr)]
    try:
        bab = make_babble(tones, "A", 3, 8000, np.random.default_rng(0))
    except NotImplementedError:
        check("make_babble() is written", lambda: False, "finish the TODO in Exercise 3.2")
    except Exception as e:
        check("make_babble() runs", lambda: False, f"{type(e).__name__}: {e}")
    else:
        check("length and dtype", lambda: bab.shape == (8000,) and bab.dtype == np.float32, "np.zeros(length, dtype=np.float32)")
        check('never the excluded speaker ("A" = the 100 Hz tone)', lambda: level_at(bab, 100) < 0.01 * level_at(bab, 200),
              'others = sorted(s for s in by_speaker if s != exclude)')
        check("three DIFFERENT talkers, all at the same level", lambda: np.ptp([level_at(bab, f) for f in (200, 300, 400)])
              < 0.05 * level_at(bab, 200), "replace=False, and divide each piece by its std")
        check("a different rng gives a different babble",
              lambda: not np.allclose(bab, make_babble(tones, "A", 3, 8000, np.random.default_rng(1))))

    section("Exercise 3.3  the new Dataset: only the noise changes")
    week3 = NoisyMixDataset(files, seed=0)
    try:
        same = BabbleMixDataset(files, p_babble=0.0, seed=0)
        always = BabbleMixDataset(files, p_babble=1.0, seed=0)
        new = BabbleMixDataset(files, seed=0)
        n_equal = sum(torch.equal(new[i][0], week3[i][0]) for i in range(400))
    except NotImplementedError:
        check("__getitem__ is written", lambda: False, "finish the TODOs in Exercises 3.1-3.3")
        summary()
        raise SystemExit(1)
    except Exception as e:
        check("the new Dataset runs", lambda: False, f"{type(e).__name__}: {e}")
        summary()
        raise SystemExit(1)
    check("p_babble = 0 gives exactly the Week 3 examples", lambda: all(
        torch.equal(same[i][0], week3[i][0]) and torch.equal(same[i][1], week3[i][1]) for i in range(30)),
        "use rng exactly in the Week 3 order, and coin for nothing but the new decisions")
    check("with babble, the clean target is still the Week 3 one - only the noise changed", lambda: all(
        torch.equal(always[i][1], week3[i][1]) for i in range(30)), "t and the crop come from rng, before the coin is used")
    check("p_babble = 1: every example has new noise", lambda: not any(torch.equal(always[i][0], week3[i][0]) for i in range(30)))
    print(f"  p_babble = 0.5: {400 - n_equal} of 400 examples got babble")
    check("p_babble = 0.5: about half of the examples get babble (40-60 %)", lambda: 160 <= 400 - n_equal <= 240)
    snrs = [level_db(c.numpy()) - level_db((n - c).numpy()) for n, c in (always[i] for i in range(100))]
    check(f"babble examples stay in the SNR range {SNR_RANGE} dB", lambda: min(snrs) > SNR_RANGE[0] - 0.01 and max(snrs) < SNR_RANGE[1] + 0.01)
    for k in range(3):
        noisy, clean = always[k]
        save_wav(OUT / "lab3_examples" / f"{k}_babble_noisy.wav", noisy.numpy())
        save_wav(OUT / "lab3_examples" / f"{k}_clean.wav", clean.numpy())
    print("  saved out/lab3_examples/ - listen: is the target still the loudest voice? (question Q5)")

    section("The 10-example test: can the new pipeline memorize 10 examples?")
    losses = overfit_10(new)
    print(f"  first loss {losses[0]:.4f} -> last {losses[-1]:.4f}  (a full run ends near 0.07: it never sees an example twice)")
    check("loss falls below 1/5 of where it started", lambda: losses[-1] < 0.2 * losses[0],
          "if it cannot memorize 10 examples, something is broken: the target, the features or the training step")

    section("design.md")
    design = Path("design.md").read_text(encoding="utf-8")
    first_four = design.split("## 5.")[0]
    left = first_four.count("[fill in")
    check("parts 1-4 of design.md are filled in", lambda: left == 0, f"{left} '[fill in ...]' left before part 5")
    summary()
