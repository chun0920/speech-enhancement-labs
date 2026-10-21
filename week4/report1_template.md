# Report 1 — GTCRN: block diagram and tensor shapes (master's students)

Presented to Prof. Hung. About 10 minutes. Build it from `results/layer_table.md` (Lab 1), the paper, and the
code (`~/baselines/gtcrn/gtcrn.py`). Every shape below must come from your own run, not from the paper.

## 1. One sentence

What GTCRN does, for whom, and at what cost (parameters, MACs per second).

>

## 2. Block diagram

Draw it (slides or a clean hand drawing). One box per top-level block, in the order they run, and on every
arrow the tensor shape for 4 s of audio: `(batch, channels, frames, frequency)` — write which axis is which.

>

## 3. Block by block

| Block | Input shape | Output shape | Weights | What it does | Why it is there (paper section) |
|---|---|---|---|---|---|
| STFT (in `infer.py`) | | | 0 | | |
| erb.bm | | | | | |
| sfe | | | | | |
| encoder | | | | | |
| dpgrnn1, dpgrnn2 | | | | | |
| decoder | | | | | |
| erb.bs | | | | | |
| mask | | | | | |
| iSTFT | | | 0 | | |

## 4. Where the weights are

Which blocks hold most of the 48 K weights? Which part of the model costs computation without any weights?

>

## 5. Paper vs code

Anything that differs between the paper and the official code (the README mentions two), and anything in the
code the paper does not describe.

>

## 6. Questions for the discussion

>
