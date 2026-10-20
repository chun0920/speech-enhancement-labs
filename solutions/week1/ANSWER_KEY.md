# Week 1 — Answer key

The finished lab files in this folder are the reference solutions. Try the lab yourself first; compare afterwards.

## Reference numbers (so you can tell a broken run from a normal one)

Identical on the lab server lab112 (RTX 5060 Ti, Python 3.11, torch 2.14 cu130), on an RTX 3060 (Python 3.9,
torch 2.6) and on a CPU-only setup (Python 3.11, torch 2.14, numpy 2.4).

| Run | Result |
|---|---|
| `lab2_train_loop.py` (default) | loss 0.270 → 0.0028; SNR −3.0 → 16.6 dB (**+19.6 dB**) |
| `--kernel 15` | +12.3 dB |
| `--kernel 5` | +7.8 dB |
| `--kernel 3` | +6.5 dB |
| `--forget-zero-grad` | loss 0.27 → 20.9 → 128 → 272 → 0.14 → 546 (swings wildly); SNR ends at −36 dB |
| Lab 3, 16-bit round trip | 71–73 dB |
| Lab 3, 8-bit round trip | 23–24 dB |
| Lab 3, energy above 4.5 kHz kept by 16k→8k→16k | ≈ 0.001 % |

Check counts when everything passes: Lab 1 14/14, Lab 2 2/2, Lab 3 15/15, Lab 4 14/14.

## Answers

**Q1 (clicks).** The hard beeps click at every start and stop; the faded ones don't. A sudden jump —
or a sudden kink — in a waveform contains energy spread over a very wide range of frequencies, not
just 880 Hz, and that broadband burst is what we hear as a click. A 10 ms ramp makes the edge smooth,
which keeps the energy close to 880 Hz. *Link to next week:* this is exactly why the STFT multiplies
every frame by a smooth window (Hann) before the FFT.

**Q2 (what the filter learned).** It passes the four tones (500 Hz–4 kHz, the red lines) at about
0 dB and blocks everything below ~200 Hz by 20–60 dB, i.e. it has become a high-pass filter. The
rumble and the tones live in *different* frequency bands, so a filter that blocks one band and passes
the other removes the noise without hurting the target. Good answers mention both halves: noise
energy is low-frequency, and the tones sit above it. Between the red lines the curve also dips
(down to about −33 dB near 800 Hz): the rumble has a little energy there and no tone needs those
frequencies, so turning them down only lowers the loss. A student who says "it keeps exactly the tone
frequencies" has read the plot correctly too.

**Q3 (forgetting `zero_grad`).** `loss.backward()` *adds* new gradients to whatever is stored in
`.grad`. Without `zero_grad()`, the gradient used at step t is the sum of every gradient since step 0.
The update keeps pushing in directions that were right many steps ago, so the weights overshoot the
minimum and keep swinging further (loss 0.27 → 21 → 128 → 0.14 → 546). Like momentum with no friction.
Accept any answer that says "gradients accumulate, so the steps use stale/ever-growing gradients".

**Q4 (5 weights).** Worse: about +8 dB instead of +20 dB. A filter with N weights can only change its
gain gradually across frequency — the finest detail it can draw is roughly sr / N = 16000 / 5 = 3.2 kHz
wide. To block 200 Hz and pass 500 Hz it needs a steep edge inside a 300 Hz gap; 63 weights give
~250 Hz detail, 5 weights only give a gentle slope, so it either lets rumble through or also dims the
500 Hz tone. (Short filter = coarse frequency resolution; this returns as frame length vs. frequency
resolution in the STFT.)

**Q5 (a second talker).** No. A fixed filter does the same thing at all times and can only separate
sounds that occupy *different frequencies*. A second voice covers the same band (~100 Hz–4 kHz+) as
the target voice, so anything that blocks it blocks the target too. You need processing that changes
over time *and* frequency (spectrogram masks, next weeks) or a model that knows what the target
speaker sounds like.

**Q6 (telephone).** It sounds muffled, like a phone call. At 8 kHz the highest frequency that can be
represented is 4 kHz (Nyquist = half the sample rate), so everything above 4 kHz is removed. The
sounds that suffer most are fricatives — s, sh, f, th — which carry much of their energy above 4 kHz;
"s" and "f" become hard to tell apart. Resampling back to 16 kHz only adds samples *between* the
existing ones; the information above 4 kHz was thrown away and cannot be recovered (guessing it back
is a research topic of its own: bandwidth extension).

**Q7 (8 bits).** A grainy hiss that follows the speech, most obvious in quiet parts. 16-bit gave
≈ 72 dB and 8-bit ≈ 24 dB: 48 dB for 8 bits, so ≈ 6 dB per bit (theory: 6.02 dB/bit). Rounding to a
grid of levels is like adding a small noise; every extra bit halves the grid spacing, i.e. −6 dB of
that noise. At 16 bits it sits far below anything audible, which is why 16-bit WAV is fine for
storing model outputs.

**Q8 (scaling the pair).** In `make_pair` the noise is scaled relative to the clean speech, so at −10 dB
the noise is 10 dB *louder* than the speech and sets the mixture's peak. Normalising that peak to 0.9
needs a small g, and the target gets the same small g. If you saved the target unscaled, then
`noisy − target` would no longer be pure noise — it would contain a piece of the speech, (g − 1)·clean —
and the SNR written in the manifest would be wrong. A model trained on such pairs would be asked to
change the volume by an amount that depends on the noise level, which it cannot know, and a
loss like MSE would punish it for that instead of for leftover noise.

**Q9 (listening).** Open answer. Typical: 20 and 10 dB are easy, 0 dB takes effort but is intelligible,
−10 dB is hard or impossible. Good answers notice that it depends on the *type* of noise: pink or
babble noise overlapping the 1–4 kHz region hurts much more than low-frequency rumble at the same SNR.
That is the motivation for STOI next week: SNR alone does not predict intelligibility.

## Common mistakes to look for

| Lab | Symptom | Cause |
|---|---|---|
| 1.1 | `RuntimeError` / shape mismatch only when `fade=0` | `env[-0:]` is the whole array; needs `if F > 0` |
| 1.3 | output shape `[4, 4, 16000]` | `amax(dim=-1)` without `keepdim=True` — broadcasting silently pairs every clip with every peak |
| 1.3 | peaks 0.07 / 0.2 / … instead of 0.9 | used `x.abs().max()` over the whole batch |
| 2 | loss never moves | `zero_grad()` placed between `backward()` and `step()` (the gradients are erased before the step uses them) |
| 2 | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | the forward pass `model(noisy)` is inside `torch.no_grad()` |
| 3.2 | first check fails, others pass | divided by 32767 (−32768 → −1.00003) |
| 3.3 | length check fails | `up` and `down` swapped |
| 3.4 | clipping check fails on some files | used `x.max()` instead of `np.abs(x).max()` — misses a loud negative peak |
| 4.1 | sine level −1.5 dB instead of −3.0 dB | `10 * log10(RMS)` instead of `20 * log10(RMS)` |
| 4.2 | measured SNR misses the target by several dB | used `10 ** (gain_db / 10)` — amplitudes need `/ 20` |
| 4.3 / 4.4 | "pair still measures 5 dB" fails | target not multiplied by g |
