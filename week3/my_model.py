"""
Your group's baseline, wrapped so that the course tools can use it.     (Week 3; used again in Weeks 4 and 8)

Every model's official code is different: what forward() takes (a waveform? a complex spectrogram?),
which STFT settings it uses, how the checkpoint is stored. This file hides those differences behind
three functions, so that the course tools work for ANY model:

    build_model(device)            -> the network with its official weights, in eval mode
    enhance(model, noisy, device)  -> the enhanced waveform: 1-D float32 numpy, SAME length as `noisy`
    example_input(seconds, device) -> what forward() takes for `seconds` of audio (Week 4 shapes, Week 8 MACs)

Fill in the TODOs by reading your model's OFFICIAL inference script (infer.py / inference.py / the
README) and copying its steps exactly: the same STFT, the same window, the same checkpoint key. This
week's job is to get the authors' numbers with the authors' weights, so nothing may differ from theirs.

Test it on a few files first:   python infer_dir.py --noisy <noisy dir> --out out/enhanced --limit 5
"""
import sys
from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE = 16000

# ---------------------------------------------------------------- Exercise 1  where things are
# Where you cloned the official repository, which checkpoint (the one trained on YOUR test set's
# training data), and a short name for tables and file names.
REPO = None              # TODO, e.g. Path.home() / "baselines" / "<model>"
CHECKPOINT = None        # TODO, e.g. REPO / "checkpoints" / "<file>"
MODEL_NAME = None        # TODO, e.g. "gtcrn" - lower case, no spaces

# Settings copied from the official inference script (yours may have other names, or none).
N_FFT, HOP = None, None   # TODO if your model works on an STFT; delete if it takes a waveform


# ---------------------------------------------------------------- Exercise 2  the network and its weights
def build_model(device="cpu"):
    """The official network with the official weights, on `device`, in eval mode.

      1. sys.path.insert(0, str(REPO))        so that the repository's own modules can be imported
      2. import the model class and create it exactly as the official script does
      3. ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
         print(ckpt.keys()) once to see where the weights are (often "model" or "state_dict")
      4. model.load_state_dict(<the weights>)   - keep strict=True: a missing key means a wrong model
      5. return model.to(device).eval()
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 3  one file
def enhance(model, noisy, device="cpu"):
    """noisy: 1-D float32 numpy array at 16 kHz. Returns the enhanced signal, 1-D float32 numpy, SAME length.

    Copy the official inference steps. Two things newer PyTorch versions need:
      - torch.stft(..., return_complex=True); if the model wants real/imaginary as the last axis (B, F, T, 2),
        convert with torch.view_as_real(...), and back with torch.view_as_complex(x.contiguous())
      - torch.istft(..., length=len(noisy)) so the output has exactly the input's length
    Run the network inside `with torch.no_grad():` and return .cpu().numpy().
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 4  what forward() takes
def example_input(seconds=4.0, device="cpu"):
    """A random input for `seconds` seconds of audio, in exactly the form model(...) takes, as a TUPLE of tensors
    with a batch axis of 1. Example for a waveform model: (torch.randn(1, int(seconds * SAMPLE_RATE)),)

    Week 4 uses it to print the shape of every layer, Week 8 to count MACs.
    """
    # TODO
    raise NotImplementedError
