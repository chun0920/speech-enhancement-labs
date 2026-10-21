# Speech Enhancement Labs

Hands-on labs that follow the lab's ten-week plan: from audio as numbers to running, training and
evaluating a published speech-enhancement model (GTCRN), with one evaluation that both groups share.
The labs run in a terminal on the lab's Linux computers; full training (Weeks 5–7) runs on lab112.

| Week | Topic | Folder | Solutions |
|---|---|---|---|
| 1 | Audio as data: tensors, a training loop, WAV files, SNR | [`week1/`](week1/) | [`solutions/week1/`](solutions/week1/) |
| 2 | Seeing and scoring noise: STFT, spectrograms, SI-SDR / PESQ / STOI | [`week2/`](week2/) | [`solutions/week2/`](solutions/week2/) |
| 3 ◆ | Your baseline: GTCRN's official checkpoint on the test set | [`week3/`](week3/) | |
| 4 | Read GTCRN's code; run its training for a few epochs | [`week4/`](week4/) | |
| 5–6 | Full training; loss and metric curves; checkpoints scored on the validation set | coming | |
| 7 ◆ | Your training against the paper's numbers | coming | |
| 8 | Parameters, MACs, real-time factor | coming | |
| 9 | Mismatched conditions; the comparison table; a demo page | coming | |
| 10 ◆ | Report; a clean repository | coming | |

◆ acceptance weeks, checked by Prof. Hung. Do the weeks in order: each folder has its own `README.md` —
start there. Solutions for a week are added to [`solutions/`](solutions/) once that week is over.

| Also here | What it is |
|---|---|
| [`tools/se_eval/`](tools/se_eval/) | the evaluation both groups share (from Week 3): one script, one locked environment, the data preparation, the listening lists |
| [`extra/1_mask_networks/`](extra/1_mask_networks/) | optional: train your own mask networks (MLP vs GRU) — the former Week 3 |
| [`extra/2_research_method/`](extra/2_research_method/) | optional: reproduce, listen, predict, change one thing — the former Week 4; needs Optional 1 |

## Getting the labs onto a computer

This repository is private, so the computer needs your permission to download it. You do this once.

**1. Accept the invitation.** Your mentor adds you as a collaborator; accept the e-mail or the
notification on github.com.

**2. Make a token** (GitHub no longer accepts your password on the command line):
github.com → your picture → **Settings** → **Developer settings** → **Personal access tokens** →
**Tokens (classic)** → **Generate new token (classic)**. Tick the **`repo`** box, choose an expiry
date, generate, and copy the token (it starts with `ghp_`). You will not see it again.

**3. Clone:**

```bash
git config --global credential.helper 'cache --timeout=86400'   # remember the token for a day
git clone https://github.com/chun0920/speech-enhancement-labs.git ~/speech-enhancement-labs
```

Username: your GitHub username. Password: paste the token. A new computer needs more than git — Week 3's
README, Step 0, lists everything.

## Work on a copy, not in the clone

Copy a week out of the clone and do the lab in your copy:

```bash
cp -r ~/speech-enhancement-labs/week3 ~/week3_<yourname>
cd ~/week3_<yourname>
```

That folder is also what you hand in. When a new week is published, update the clone — it never
conflicts with your work, because you never edit the clone:

```bash
cd ~/speech-enhancement-labs
git pull
```

**Please do not push to this repository.** It is shared by everyone; your answers stay in your own copy.
`tools/se_eval` is used straight from the clone — do not edit it either.

## Data and licences

- Week 2 and the optional labs include speech from the LibriSpeech corpus (Panayotov et al., ICASSP 2015,
  https://www.openslr.org/12), licensed CC BY 4.0. Details are in each folder's `data/README.txt`.
- From Week 3 on, the data are downloaded by `tools/se_eval`'s scripts, not stored here: VCTK-DEMAND
  (Valentini-Botinhao, University of Edinburgh, https://doi.org/10.7488/ds/2117) and the DNS Challenge 3
  blind test set (Microsoft, CC BY 4.0).
- `tools/se_eval/dnsmos/` contains Microsoft's DNSMOS models (CC BY 4.0, see its `NOTICE.txt`).
- GTCRN (https://github.com/Xiaobin-Rong/gtcrn) and SEtrain (https://github.com/Xiaobin-Rong/SEtrain) are
  cloned from their authors' repositories (MIT licence); they are not part of this repository.
