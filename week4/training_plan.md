# Training plan — GTCRN on VCTK-DEMAND

Fill in the measured part from your 3-epoch run (Step 5). The plan part is what you propose for Weeks 5–7;
discuss it with your mentor before Week 5. Week 5 trains on the lab server lab112, so measure one epoch
there too when you get access.

## Measured (Week 4)

| | your computer | lab112 |
|---|---|---|
| GPU | | |
| minutes per epoch (training) | | |
| minutes per validation (1,572 pairs) | | |
| GPU memory used (nvidia-smi) | | |

| epoch | training loss | validation loss | validation PESQ (train.py) |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

se_eval of `model_003.tar` on `vctk-demand-val`: PESQ ___ STOI ___ SI-SDR ___

## Plan (Weeks 5–7)

| Decision | Your proposal | Why |
|---|---|---|
| number of epochs | | (the scheduler's decay ends at step 250,000) |
| STFT window | | (Q2) |
| which checkpoint you will report | | (picked on validation, never on test) |
| when you score checkpoints on validation | | |
| total GPU hours | | epochs × (training + validation) minutes |
| fits in Weeks 5–7 on lab112, shared with the other group? | | |
