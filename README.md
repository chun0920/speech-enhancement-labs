# Speech Enhancement Labs

Weekly hands-on labs: from audio as numbers to training your own speech-enhancement network.
Everything runs in a terminal on the lab's Linux server.

| Week | Topic | Folder |
|---|---|---|
| 1 | Audio as data: tensors, a training loop, WAV files, SNR | [`week1/`](week1/) |
| 2 | Seeing and scoring noise: STFT, spectrograms, SI-SDR / PESQ / STOI | [`week2/`](week2/) |
| 3 | Your first enhancement network: learning to guess the mask | [`week3/`](week3/) |

Do the weeks in order: Week 1 sets up the environment that Weeks 2 and 3 reuse, and each week builds
on the one before. Each folder has its own `README.md` — start there.

## Getting the labs onto the lab server

This repository is private, so the server needs your permission to download it. You do this once.

**1. Accept the invitation.** Your mentor adds you as a collaborator; accept the e-mail or the
notification on github.com.

**2. Make a token** (GitHub no longer accepts your password on the command line):
github.com → your picture → **Settings** → **Developer settings** → **Personal access tokens** →
**Tokens (classic)** → **Generate new token (classic)**. Tick the **`repo`** box, choose an expiry
date, generate, and copy the token (it starts with `ghp_`). You will not see it again.

**3. Clone on the server:**

```bash
git config --global credential.helper 'cache --timeout=86400'   # remember the token for a day
git clone https://github.com/chun0920/speech-enhancement-labs.git
```

Username: your GitHub username. Password: paste the token.

## Work on a copy, not in the clone

Copy a week out of the clone and do the lab in your copy:

```bash
cp -r speech-enhancement-labs/week1 ~/week1_<yourname>
cd ~/week1_<yourname>
```

That folder is also what you hand in. When a new week is published, update the clone — it never
conflicts with your work, because you never edit the clone:

```bash
cd ~/speech-enhancement-labs
git pull
```

**Please do not push to this repository.** It is shared by everyone; your answers stay in your own copy.

## Audio

Weeks 2 and 3 include speech from the LibriSpeech corpus (Panayotov et al., ICASSP 2015,
https://www.openslr.org/12), licensed CC BY 4.0. Details are in each week's `data/README.txt`.
