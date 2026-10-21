"""Helpers shared by all labs this week - no need to edit this file.

Everything from Weeks 1-3 is here, finished: mixing at an SNR, STFT / iSTFT, the ideal ratio mask,
PESQ / STOI / SI-SDR, and the whole Week 3 recipe (the dynamic-mixing Dataset, the normalized
log-power features, MaskGRU and its training loop). This week's labs reuse that recipe unchanged.

New this week: the 20-sentence test set, the columns of results/results.csv, and code_version(),
which stamps every number with the version of the code that produced it.
"""
import csv
import hashlib
import os
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen on the server -> figures are written to PNG files

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

OUT = Path("out")
OUT.mkdir(exist_ok=True)
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
RESULTS_CSV = RESULTS / "results.csv"      # one row per number, never delete a row
DATA = Path("data")
TRAIN_DIR = DATA / "train"                 # 117 clean training clips, the same as Week 3
TEST_DIR = DATA / "test"                   # 20 test sentences from 20 speakers - never used for training
TEST_SPEECH = DATA / "speech_sample.wav"   # the Week 2 / Week 3 sentence
BABBLE = DATA / "noise_babble.flac"        # six people talking at once - never used for training
SR = 16000
N_FFT, HOP = 512, 128                      # 32 ms Hann frames, 8 ms hop - same as Weeks 2 and 3
N_FREQ = N_FFT // 2 + 1                    # 257 frequency bins
SEG = 2 * SR                               # every training example is 2 s long
SNR_RANGE = (-5, 15)                       # training SNRs (dB)
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


def all_passed():
    return all(_passed)


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


# ------------------------------------------------------------------ metrics (Week 2)
def si_sdr(ref, est, eps=1e-8):
    ref = ref.astype(np.float64) - ref.mean()
    est = est.astype(np.float64) - est.mean()
    target = np.dot(est, ref) / np.dot(ref, ref) * ref
    return 10 * np.log10((np.sum(target ** 2) + eps) / (np.sum((est - target) ** 2) + eps))


METRICS = ("PESQ", "STOI", "SI-SDR")


def evaluate(ref, est):
    """{"PESQ", "STOI", "SI-SDR"} of a 1-D numpy estimate against the clean reference."""
    from pesq import pesq
    from pystoi import stoi
    return {"PESQ": pesq(SR, ref, est, "wb"), "STOI": stoi(ref, est, SR, extended=False), "SI-SDR": si_sdr(ref, est)}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------ the Week 3 recipe (finished)
def random_crop(x, length, rng):
    """A random piece of x, exactly `length` samples (zero-padded at the end if x is shorter)."""
    if len(x) > length:
        start = rng.integers(0, len(x) - length + 1)
        return x[start:start + length]
    return np.pad(x, (0, length - len(x)))


def train_files():
    """The 117 training clips, sorted."""
    return sorted(TRAIN_DIR.glob("*.flac"))


class NoisyMixDataset(torch.utils.data.Dataset):
    """Week 3: endless (noisy, clean) pairs - a random clip, a random 2 s piece, stationary noise with a
    random tilt, a random SNR. The same index always gives the same example."""

    def __init__(self, files, seg=SEG, snr_range=SNR_RANGE, size=100_000, seed=0):
        self.clips = [load_audio(f) for f in files]
        self.seg, self.snr_range, self.size, self.seed = seg, snr_range, size, seed

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        clean = random_crop(self.clips[rng.integers(len(self.clips))], self.seg, rng)
        noise = make_noise(self.seg, tilt=rng.uniform(0, 2), rng=rng)
        noisy, _ = mix_at_snr(clean, noise, rng.uniform(*self.snr_range))
        return torch.from_numpy(noisy), torch.from_numpy(clean)


def log_spec_features(Y):
    """Week 3: log power of the noisy STFT [B, F, T], normalized per example (mean 0, std 1)."""
    f = torch.log(Y.abs() ** 2 + 1e-8)
    return (f - f.mean(dim=(1, 2), keepdim=True)) / (f.std(dim=(1, 2), keepdim=True) + 1e-5)


class MaskGRU(torch.nn.Module):
    """Week 3: a causal 2-layer GRU, 257 values in and 257 mask values out per frame (856,321 weights)."""

    def __init__(self, n_freq=N_FREQ, hidden=256, layers=2):
        super().__init__()
        self.gru = torch.nn.GRU(n_freq, hidden, num_layers=layers, batch_first=True)
        self.out = torch.nn.Linear(hidden, n_freq)

    def forward(self, feats):
        h, _ = self.gru(feats.transpose(1, 2))
        return torch.sigmoid(self.out(h)).transpose(1, 2)


def train_step(model, noisy, clean, optimizer):
    """Week 3: one step - target IRM, MSE loss, the five steps of Week 1. Returns the loss."""
    Y, S, N = stft(noisy), stft(clean), stft(noisy - clean)
    loss = torch.nn.functional.mse_loss(model(log_spec_features(Y)), ideal_ratio_mask(S, N))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train(model, dataset, steps=2000, seed=0, batch_size=32, lr=1e-3):
    """Week 3 training loop: `steps` batches from `dataset`. Returns the list of losses."""
    torch.manual_seed(seed)
    model.to(DEVICE).train()
    workers = min(4, os.cpu_count() or 1)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses, t0 = [], time.time()
    for step, (noisy, clean) in enumerate(loader):
        if step == steps:
            break
        losses.append(train_step(model, noisy.to(DEVICE), clean.to(DEVICE), optimizer))
        if step % 500 == 0 or step == steps - 1:
            print(f"    step {step:5d}/{steps}   loss {np.mean(losses[-50:]):.4f}   {time.time() - t0:4.0f} s", flush=True)
    model.eval()
    return losses


def enhance(model, noisy):
    """Run a trained model on one 1-D numpy signal; returns the enhanced signal."""
    with torch.no_grad():
        Y = stft(torch.from_numpy(noisy).to(DEVICE)[None])
        return istft(model(log_spec_features(Y)) * Y, len(noisy))[0].cpu().numpy()


def load_gru(path):
    """A MaskGRU with the weights saved in `path`, ready to use."""
    if not Path(path).exists():
        raise SystemExit(f"{path} not found - run the lab that trains it first")
    model = MaskGRU()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()


# ------------------------------------------------------------------ this week's exam
TEST_NOISES = ("pink", "babble")
TEST_SNRS = (0, 5)
SPLITS = [f"{kind}-{snr}dB" for kind in TEST_NOISES for snr in TEST_SNRS]   # pink-0dB, pink-5dB, babble-0dB, babble-5dB


def test_set():
    """The exam: 20 sentences x {pink, babble} x {0, 5 dB} = 80 mixtures, the same every time.

    Each item is a dict: sentence, noise, snr, split, clean, noisy, noise_part (the scaled noise).
    """
    babble = load_audio(BABBLE)
    items = []
    for i, f in enumerate(sorted(TEST_DIR.glob("*.flac"))):
        clean = load_audio(f)
        rng = np.random.default_rng(1000 + i)
        noises = {"pink": make_noise(len(clean), tilt=1.0, rng=rng)}
        start = rng.integers(0, len(babble) - len(clean) + 1)
        noises["babble"] = babble[start:start + len(clean)]
        for kind in TEST_NOISES:
            for snr in TEST_SNRS:
                noisy, noise_part = mix_at_snr(clean, noises[kind], snr)
                items.append({"sentence": f.stem, "noise": kind, "snr": snr, "split": f"{kind}-{snr}dB",
                              "clean": clean, "noisy": noisy, "noise_part": noise_part})
    return items


def oracle(item):
    """The Week 2 ceiling for one test item: noisy x ideal ratio mask (it cheats: it knows the clean speech)."""
    S, N, Y = (stft(torch.from_numpy(item[k])) for k in ("clean", "noise_part", "noisy"))
    return istft(ideal_ratio_mask(S, N) * Y, len(item["noisy"])).numpy()


def score_split_means(process, items):
    """Score process(item) -> enhanced signal on every item. Returns {split: {metric: mean over the 20 sentences}}."""
    per_split = {}
    for item in items:
        per_split.setdefault(item["split"], []).append(evaluate(item["clean"], process(item)))
    return {split: {m: float(np.mean([s[m] for s in scores])) for m in METRICS} for split, scores in per_split.items()}


# ------------------------------------------------------------------ records
FIELDS = ["run_id", "date", "config_path", "dataset", "split", "metric", "value", "seed", "git_commit", "note"]
TEST_SET_NAME = "test-clean-20"


def read_results(path=RESULTS_CSV):
    """All rows of results.csv as a list of dicts (empty list if the file does not exist yet)."""
    if not Path(path).exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def system_of(run_id):
    """'20261020_gru-week3_s1' -> 'gru-week3'."""
    return run_id.split("_", 1)[1].rsplit("_s", 1)[0]


def rows_for(run_id, config_path, dataset, split_means, seed, version, note=""):
    """Turn {split: {metric: value}} into results.csv rows - one row per (split, metric)."""
    day = time.strftime("%Y-%m-%d")
    return [{"run_id": run_id, "date": day, "config_path": config_path, "dataset": dataset, "split": split,
             "metric": m, "value": f"{v:.4f}", "seed": seed, "git_commit": version, "note": note}
            for split, means in split_means.items() for m, v in means.items()]


VERSIONED = ["design.md", "pyproject.toml", "uv.lock"]   # plus every .py file in this folder


def code_version():
    """Which version of the code produces a number.

    In a git repository: the commit, e.g. 'a1b2c3d', with '-dirty' added if a .py file, design.md or the
    environment files have changes that are not committed. 'uncommitted' if nothing is committed yet.
    Without git: 'nogit-' + a fingerprint of those files (it changes whenever one of them changes).
    """
    files = sorted(Path(".").glob("*.py")) + [Path(p) for p in VERSIONED if Path(p).exists()]
    fingerprint = hashlib.sha1(b"".join(p.read_bytes() for p in files)).hexdigest()[:7]

    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True)

    try:
        top = git("rev-parse", "--show-toplevel")
    except OSError:
        return "nogit-" + fingerprint
    if top.returncode != 0:
        return "nogit-" + fingerprint
    if Path(top.stdout.strip()).resolve() != Path.cwd().resolve():
        raise SystemExit(f"This folder is inside another git repository ({top.stdout.strip()}).\n"
                         "Work in your own copy (README > Setup): cp -r the week4 folder out, then git init there.")
    head = git("rev-parse", "--short", "HEAD")
    if head.returncode != 0:
        return "uncommitted"
    dirty = git("status", "--porcelain", "--", *map(str, files)).stdout.strip()
    return head.stdout.strip() + ("-dirty" if dirty else "")


def require_committed(version):
    """Stop before a full run if the code that would produce the numbers is not committed."""
    if version.startswith("nogit-"):
        print(f"  [WARN] git is not available here - numbers are stamped with a fingerprint of your code ({version}).")
        print("         Keep a copy of the folder for every result you report, and tell your mentor.")
        return
    if version == "uncommitted" or version.endswith("-dirty"):
        print(f"\n  Your code has changes that are not committed ({version}).")
        print("  Every number must point to a commit. Commit first, then run again:")
        print('      git add -A && git commit -m "what you changed"')
        raise SystemExit(1)
