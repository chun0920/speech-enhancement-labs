"""
Week 4 - Lab 2: VCTK-DEMAND for SEtrain                        run:  python dataloader_vctk.py   (checks)

SEtrain's own dataloader.py is written for DNS3, where every training file is 10 s long: it simply reads
the first 10 s of each file. VCTK-DEMAND utterances are between about 1 and 15 s long, so a batch of them
cannot be stacked into one tensor as they are. This Dataset returns whole utterances, and its collate_fn
pads the shorter ones of a batch with zeros at the END, up to the longest one.

SEtrain's train.py creates it as   Dataset(**config["train_dataset"])   and uses Dataset.collate_fn if it
exists - so the argument names below must match the config file (cfg_train_vctk.yaml).

When all checks pass, copy this file into ~/baselines/SEtrain/ (README, Step 4).
"""
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class VCTKDataset(Dataset):
    """(noisy, clean) pairs from two folders with the same file names."""

    def __init__(self, clean_dir, noisy_dir, train=True, limit=0, fs=16000):
        self.clean_dir = Path(os.path.expanduser(clean_dir))
        self.noisy_dir = Path(os.path.expanduser(noisy_dir))
        self.train, self.fs = train, fs
        # ---------------------------------------------------------------- Exercise 2.1
        # self.files = the sorted list of file NAMES (not paths) of the *.wav files in noisy_dir.
        # Every name must also exist in clean_dir: if any is missing, raise ValueError and say which.
        # If limit > 0, keep only the first `limit` names (for quick tests).
        # TODO
        raise NotImplementedError
        print(f"[VCTKDataset] {'train' if train else 'valid'}: {len(self.files)} pairs from {self.noisy_dir}")

    def __len__(self):
        return len(self.files)

    # ---------------------------------------------------------------- Exercise 2.2
    def __getitem__(self, idx):
        """(noisy, clean) of file number idx: two 1-D float32 numpy arrays of the same length.
        Read both with sf.read(path, dtype="float32"); if a file's sample rate is not self.fs, raise ValueError.
        """
        # TODO
        raise NotImplementedError

    # ---------------------------------------------------------------- Exercise 2.3
    @staticmethod
    def collate_fn(batch):
        """batch: a list of (noisy, clean) pairs of different lengths. Return (noisy, clean) as two float32
        tensors of shape [B, L], L = the longest length in the batch, shorter ones padded with zeros at the end.
            np.pad(x, (0, L - len(x)))   pads x at the end;   torch.from_numpy(np.stack(list_of_arrays))
        """
        # TODO
        raise NotImplementedError


if __name__ == "__main__":
    import tempfile
    import time
    _passed = []

    def check(name, fn, hint=""):
        try:
            ok = bool(fn())
        except NotImplementedError:
            ok, hint = False, "not written yet - look for the # TODO"
        except Exception as e:
            ok, hint = False, f"{type(e).__name__}: {e}  {hint}".strip()
        _passed.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok and hint:
            print(f"         hint: {hint}")

    print("--- Exercises 2.1-2.3 on three small fake files ---")
    tmp = Path(tempfile.mkdtemp())
    for kind in ("clean", "noisy"):
        (tmp / kind).mkdir()
        for name, n in (("b.wav", 800), ("a.wav", 1600), ("c.wav", 400)):
            sf.write(tmp / kind / name, (0.1 if kind == "clean" else 0.2) * np.ones(n, dtype=np.float32), 16000, subtype="FLOAT")
    try:
        ds = VCTKDataset(tmp / "clean", tmp / "noisy")
        ds2 = VCTKDataset(tmp / "clean", tmp / "noisy", limit=2)
    except NotImplementedError:
        check("__init__ is written", lambda: False, "finish Exercise 2.1 first")
        raise SystemExit(1)
    check("file names, sorted, without folders", lambda: ds.files == ["a.wav", "b.wav", "c.wav"], "sorted(p.name for p in ...)")
    check("limit=2 keeps the first two", lambda: ds2.files == ["a.wav", "b.wav"] and len(ds2) == 2)
    (tmp / "noisy" / "d.wav").write_bytes((tmp / "noisy" / "a.wav").read_bytes())

    def missing_raises():
        try:
            VCTKDataset(tmp / "clean", tmp / "noisy")
        except ValueError:
            return True
        return False
    check("a noisy file without a clean partner raises ValueError", missing_raises)
    (tmp / "noisy" / "d.wav").unlink()
    check("item 1 is (noisy, clean) of b.wav, float32", lambda: ds[1][0].dtype == np.float32 and len(ds[1][0]) == 800
          and abs(ds[1][0][0] - 0.2) < 1e-6 and abs(ds[1][1][0] - 0.1) < 1e-6, "noisy first, then clean")
    try:
        n, c = VCTKDataset.collate_fn([ds[0], ds[2]])
    except NotImplementedError:
        check("__getitem__ and collate_fn are written", lambda: False, "finish Exercises 2.2 and 2.3")
        raise SystemExit(1)
    check("collate_fn: [2, 1600] tensors", lambda: tuple(n.shape) == (2, 1600) and tuple(c.shape) == (2, 1600) and n.dtype == torch.float32)
    check("the short one is padded with zeros at the END", lambda: float(n[1, :400].min()) > 0.19 and float(n[1, 400:].abs().max()) == 0)

    print("\n--- The real data ---")
    root = Path.home() / "data" / "VCTK-DEMAND"
    if (root / "train" / "noisy").exists():
        train = VCTKDataset(root / "train" / "clean", root / "train" / "noisy")
        val = VCTKDataset(root / "val" / "clean", root / "val" / "noisy", train=False)
        check("10,000 training pairs and 1,572 validation pairs", lambda: len(train) == 10000 and len(val) == 1572)
        check("no utterance is in both", lambda: not set(train.files) & set(val.files))
        loader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True, collate_fn=VCTKDataset.collate_fn)
        t0 = time.time()
        noisy, clean = next(iter(loader))
        print(f"  one batch: noisy {tuple(noisy.shape)}, clean {tuple(clean.shape)}  ({time.time() - t0:.2f} s)")
        check("a batch of 8 from the DataLoader", lambda: noisy.shape == clean.shape and noisy.shape[0] == 8)
    else:
        print(f"  {root / 'train'} not found - prepare the training set first (README, Step 1)")
    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))
