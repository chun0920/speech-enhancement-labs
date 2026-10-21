"""
A checkpoint trained with SEtrain, wrapped like my_model.py.                 (Week 4, given - no TODOs)

SEtrain's GTCRN (~/baselines/SEtrain/models/gtcrn_end2end.py) takes the WAVEFORM and does the STFT itself,
with the window SEtrain chose. Its layers and weights are exactly those of the official GTCRN - but the
STFT window is not the one in the official infer.py (README, Step 2). So a checkpoint trained with SEtrain
must be run through SEtrain's own model class, not through my_model.py.

    python infer_dir.py --adapter setrain_model --checkpoint ~/exp/<run>/checkpoints/model_003.tar \\
                        --noisy ~/data/VCTK-DEMAND/val/noisy --out out/val_epoch003
"""
import sys
from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE = 16000
SETRAIN = Path.home() / "baselines" / "SEtrain"
REPO = SETRAIN
CHECKPOINT = None                 # always given with --checkpoint: which epoch of which run
MODEL_NAME = "gtcrn-setrain"


def build_model(device="cpu"):
    sys.path.insert(0, str(SETRAIN))
    from models.gtcrn_end2end import GTCRN
    model = GTCRN(n_fft=512, hop_len=256, win_len=512)        # network_config of the training config
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def enhance(model, noisy, device="cpu"):
    with torch.no_grad():
        out = model(torch.from_numpy(noisy).to(device)[None])[0]   # SEtrain pads the output to the input length
    return out.cpu().numpy().astype(np.float32)


def example_input(seconds=4.0, device="cpu"):
    return (torch.randn(1, int(seconds * SAMPLE_RATE), device=device),)
