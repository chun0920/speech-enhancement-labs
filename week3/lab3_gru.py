"""
Week 3 - Lab 3: Give the network a memory                      run:  python lab3_gru.py

The MLP of Lab 2 judged every 32 ms frame on its own. But much of what tells speech from
noise lives ACROSS frames:
  - the noise here is stationary: its spectrum stays the same, frame after frame
  - speech changes all the time: syllables come and go about 4 times per second,
    harmonics glide up and down, consonants start suddenly

A GRU (gated recurrent unit) reads the frames in time order and carries a memory forward,
so at every frame it knows what the last few hundred milliseconds sounded like.

A fair comparison changes ONE thing. This GRU has about the same number of trainable weights as
the MLP (~0.86 M), is trained on the same data, for the same number of steps, with the same loss.
The only difference is memory. Whatever changes in the scores is caused by that.

The GRU only looks BACKWARD in time (it is causal), like a model that must run live in a
hearing aid or a phone call.

Your job: fill in the TODO, then run this file (after lab2_mlp.py - the comparison needs out/mlp_test.json).
"""
import argparse
import json

import numpy as np
import torch

from common import (DEVICE, N_FREQ, OUT, check, count_params, evaluate, ideal_ratio_mask, section, stft,
                    summary)
from lab2_mlp import FrameMLP, enhance_test, plot_training, test_condition, train


class MaskGRU(torch.nn.Module):
    """Reads the frames in time order and keeps a memory: 257 values in, 257 mask values out per frame."""

    def __init__(self, n_freq=N_FREQ, hidden=256, layers=2):
        super().__init__()
        self.gru = torch.nn.GRU(n_freq, hidden, num_layers=layers, batch_first=True)
        self.out = torch.nn.Linear(hidden, n_freq)

    # ---------------------------------------------------------------- Exercise 3.1
    def forward(self, feats):
        """feats [B, F, T] -> mask [B, F, T], every value between 0 and 1.

          1. x = feats.transpose(1, 2)        [B, T, F]: batch_first=True wants time before features
          2. h, _ = self.gru(x)               h: [B, T, hidden]  (the GRU returns two things; keep the first)
          3. m = torch.sigmoid(self.out(h))   [B, T, F]
          4. return m.transpose(1, 2)         [B, F, T]
        """
        # TODO
        raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()

    model = MaskGRU()
    section("Model")
    n_gru, n_mlp = count_params(model), count_params(FrameMLP())
    print(f"  MaskGRU: {n_gru:,} trainable weights   (FrameMLP: {n_mlp:,})  | device: {DEVICE}")
    check("about the same size as the MLP (within 5 %)", lambda: abs(n_gru - n_mlp) / n_mlp < 0.05)

    section("Exercise 3.1  forward")
    feats = torch.randn(2, N_FREQ, 50)
    try:
        out = model(feats)
    except NotImplementedError:
        check("forward() is written", lambda: False, "finish the TODO in Exercise 3.1")
        summary()
        raise SystemExit(1)
    check("output shape = input shape [2, 257, 50]", lambda: out.shape == feats.shape, "transpose, GRU, Linear, sigmoid, transpose back")
    check("every value is between 0 and 1", lambda: out.min() >= 0 and out.max() <= 1, "torch.sigmoid")
    changed = feats.clone()
    changed[:, :, 30:] += 1.0                                  # change only the LATER frames
    check("causal: changing later frames does not change earlier outputs",
          lambda: torch.allclose(model(changed)[:, :, :30], out[:, :, :30], atol=1e-6))
    earlier = feats.clone()
    earlier[:, :, :20] += 1.0                                  # change only the EARLIER frames
    check("memory: changing earlier frames DOES change later outputs",
          lambda: not torch.allclose(model(earlier)[:, :, 20], out[:, :, 20], atol=1e-6))

    section(f"Training ({args.steps} steps - same data, same loss, same steps as the MLP)")
    try:
        losses = train(model, args.steps)
    except NotImplementedError:
        check("training runs", lambda: False, "finish Lab 2 first - this lab trains with your train_step()")
        summary()
        raise SystemExit(1)
    check("loss went down by at least 30 %", lambda: np.mean(losses[-50:]) < 0.7 * np.mean(losses[:50]))
    torch.save(model.state_dict(), OUT / "gru.pt")

    section("Test: the Week 2 sentence + pink noise at 0 dB")
    clean, noisy, noise = test_condition(0)
    enhanced, mask = enhance_test(model, noisy)
    oracle = ideal_ratio_mask(stft(torch.from_numpy(clean)), stft(torch.from_numpy(noise)))
    gru = {"input": "GRU", **evaluate(clean, enhanced), "train_loss": float(np.mean(losses[-200:]))}
    rows = [{"input": "noisy", **evaluate(clean, noisy)}]
    mlp_file = OUT / "mlp_test.json"
    if mlp_file.exists():
        rows.append(json.load(open(mlp_file)))
    rows.append(gru)
    for r in rows:
        loss = f"   final training loss {r['train_loss']:.4f}" if "train_loss" in r else ""
        print(f"  {r['input']:<6} PESQ {r['PESQ']:.2f}   STOI {r['STOI']:.3f}   SI-SDR {r['SI-SDR']:6.2f} dB{loss}")
    print("  (Week 2 oracle IRM at 0 dB: PESQ 2.90, STOI 0.94, SI-SDR 11.54 dB)")
    check("the GRU improves STOI and gains over 8 dB of SI-SDR", lambda: gru["STOI"] > rows[0]["STOI"] and gru["SI-SDR"] > rows[0]["SI-SDR"] + 8,
          "a full 2000-step run gains about 8.5 dB - is this a full run?")
    if mlp_file.exists():
        mlp = rows[1]
        check("the GRU's training loss is at least 10 % lower than the MLP's", lambda: gru["train_loss"] < 0.9 * mlp["train_loss"],
              "both must be trained with the same number of steps")
        check("the GRU beats the MLP on STOI", lambda: gru["STOI"] > mlp["STOI"])
    else:
        print("  out/mlp_test.json not found - run lab2_mlp.py to compare with the MLP")
    json.dump(gru, open(OUT / "gru_test.json", "w"))
    plot_training(losses, mask, oracle, OUT / "lab3_gru.png", "GRU")
    print("  saved out/gru.pt, out/gru_test.json and out/lab3_gru.png (questions Q4, Q5)")
    summary()
