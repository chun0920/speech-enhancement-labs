# Week 4 — Read GTCRN's Code; Run Its Training for a Few Epochs

Last week you ran the authors' checkpoint. From Week 5 on you will train GTCRN yourselves and, in Week 7,
compare your result with the paper. Before that, you need to know the training code well enough to trust
it. This week you read it — with the shape of every tensor next to it — write the one piece VCTK-DEMAND
needs, and run the whole training loop for three epochs: data → model → loss → validation → checkpoint →
scored with se_eval.

GTCRN's authors train with **SEtrain**, their training template: https://github.com/Xiaobin-Rong/SEtrain
(MIT licence; we use the `plus` branch at commit `3ec437d`). It was written for DNS3; you will adapt it to
VCTK-DEMAND.

When you finish you can:

1. find your way around a research training repository: which file does what, where the settings come from
2. print the shape of every layer of a network with forward hooks
3. adapt a training pipeline to a new dataset without changing anything else
4. run a short training, score its checkpoint on the validation set, and estimate how long the real run takes

Your mentor walks through the code with you this week; bring `results/layer_table.md`.
Master's students: Report 1 — the model's block diagram with the tensor shape between every two blocks.

## What's in the folder

| File | Purpose |
|---|---|
| `README.md` | this guide |
| `check_env.py` | checks this week's setup |
| `lab1_shapes.py` | **your work**: forward hooks — every layer's input and output shape |
| `dataloader_vctk.py` | **your work**: VCTK-DEMAND for SEtrain (whole utterances, padded per batch) |
| `cfg_train_vctk.yaml` | SEtrain's configuration for VCTK-DEMAND, 3 epochs |
| `setrain_model.py` | runs a checkpoint trained with SEtrain through `infer_dir.py` (given) |
| `training_plan.md` | **your work**: what the short run measured and what the full run will cost |
| `report1_template.md` | master's students: Report 1 |
| `ANSWERS.md` | code-reading questions |

You also need `my_model.py` and `infer_dir.py` from your Week 3 folder (Step 0).

## Step 0 — Copy last week's wrapper

```bash
cp -r ~/speech-enhancement-labs/week4 ~/week4_<yourname> && cd ~/week4_<yourname>
cp ~/week3_<yourname>/my_model.py ~/week3_<yourname>/infer_dir.py .
conda activate se
pip install omegaconf tensorboard joblib tqdm librosa pesq      # what SEtrain's train.py imports (pesq compiles: needs gcc)
```

## Step 1 — The training data (start it first: ~5 GB download)

```bash
uv run --project $SE $SE/prepare_vctk_demand.py --root ~/data/VCTK-DEMAND --train
```

It adds `train/` (10,000 pairs) and `val/` (1,572 pairs, the list in `tools/se_eval/splits/vctk_demand_val.txt`)
next to last week's `test/`. The test set is for Week 7's final numbers only: **never train on it, never pick a
checkpoint with it.** Everything you look at while developing is scored on `val/`.

## Step 2 — Get SEtrain and read it (≈ 60 min, with your mentor)

```bash
git clone https://github.com/Xiaobin-Rong/SEtrain.git ~/baselines/SEtrain
cd ~/baselines/SEtrain && git checkout 3ec437d && cd -
```

Read in this order; the questions in `ANSWERS.md` follow it:

| File | What it does | Look for |
|---|---|---|
| `README.md` | the intended workflow | which files you are expected to change |
| `train.py` | builds dataset, model, optimizer, scheduler, loss; the `Trainer` runs epochs | `_train_epoch`, `_validation_epoch`, `_save_checkpoint`; which data decides the "best" checkpoint |
| `models/gtcrn_end2end.py` | GTCRN with the STFT inside: waveform in, waveform out | the STFT window — compare with `~/baselines/gtcrn/infer.py` |
| `loss_factory.py` | `HybridLoss` | each term, and its weight |
| `scheduler.py` | warm-up then cosine decay, per step | how many steps the warm-up lasts |
| `dataloader.py` | the DNS3 dataset | why it cannot be used for VCTK-DEMAND |
| `configs/cfg_train.yaml` | all the settings | which ones `cfg_train_vctk.yaml` changes |

## Step 3 — Every layer's shape (Lab 1, ≈ 30 min)

```bash
python lab1_shapes.py
```

Finish the two TODOs. The output, `results/layer_table.md`, lists every layer that ran with its input and output
shape and its weights; the first table — the top-level blocks — is the block diagram's skeleton.

## Step 4 — VCTK-DEMAND for SEtrain (Lab 2, ≈ 30 min)

Finish the TODOs of `dataloader_vctk.py` and run it until all checks pass. Then put it into SEtrain, together with
the configuration, and point `train.py` at it — the **only** line of SEtrain you change:

```bash
python dataloader_vctk.py
cp dataloader_vctk.py ~/baselines/SEtrain/ && cp cfg_train_vctk.yaml ~/baselines/SEtrain/configs/
```

In `~/baselines/SEtrain/train.py`, line 21, replace
`from dataloader_dns3 import DNS3Dataset as Dataset` with `from dataloader_vctk import VCTKDataset as Dataset`.
(Why does the original line not work at all? Question Q1.)

## Step 5 — Three epochs (≈ 10–15 min of training)

```bash
cd ~/baselines/SEtrain
python train.py -C configs/cfg_train_vctk.yaml -D 0
tensorboard --logdir ~/exp --port 6006            # in a second terminal: the loss curves
cd -
```

Write down, per epoch: training loss, validation loss, validation PESQ, and the minutes it took.
Each epoch saves `~/exp/gtcrn_vctk_week4_<date>/checkpoints/model_00N.tar`.

## Step 6 — Score the checkpoint on the validation set

```bash
RUN=$(ls -d ~/exp/gtcrn_vctk_week4_* | tail -1)
python infer_dir.py --adapter setrain_model --checkpoint $RUN/checkpoints/model_003.tar \
                    --noisy ~/data/VCTK-DEMAND/val/noisy --out out/val_epoch003
se_eval --enhanced out/val_epoch003 --clean ~/data/VCTK-DEMAND/val/clean \
        --name gtcrn-week4-epoch003 --dataset vctk-demand-val --note "SEtrain, 3 epochs"
```

Three epochs are far from a trained model — that is fine. What matters this week: every step ran, the numbers
moved in the right direction, and you know how long a full run takes. Fill in `training_plan.md`.

## Hand-in

Your folder `week4_<yourname>/` with: the two labs finished (both end with **All done!**), `results/layer_table.md`,
`results/eval_summary.csv`, `training_plan.md`, `ANSWERS.md`; master's students also `report1_template.md` filled in.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'dataloader_dns3'` | Step 4: the import line of `train.py` |
| `ModuleNotFoundError: omegaconf` (or `tensorboard`, `joblib`) | Step 0: `pip install ...` in the `se` environment |
| `TypeError: __init__() got an unexpected keyword argument` | the names in `cfg_train_vctk.yaml` and in `VCTKDataset.__init__` must be the same |
| `RuntimeError: stack expects each tensor to be equal size` | `collate_fn` is not used or does not pad |
| a folder named `~` or `${oc.env:HOME}` appears | the paths in the configuration must keep `${oc.env:HOME}` exactly |
| `CUDA out of memory` | another job is on the GPU (`nvidia-smi`); or lower `batch_size` and write that down |
| the training loss does not go down at all in 3 epochs | look at the learning rate in TensorBoard: the warm-up starts very low (question Q4) |
| anything else | screenshot the whole error and send it to your mentor |
