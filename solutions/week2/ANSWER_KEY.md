# Week 2 — Answer key

The finished lab files in this folder are the reference solutions. Try the lab yourself first; compare afterwards.

Every student uses the same clip (`data/speech_sample.wav`) and the same noise seed, so their printed
numbers should match the ones below. Verified identical on the lab server lab112 (RTX 5060 Ti, Python 3.11,
torch 2.14 cu130), on an RTX 3060 (Python 3.9, torch 2.6) and on a CPU-only setup (Python 3.11,
torch 2.14, numpy 2.4). A student whose numbers differ has a bug, even if all checks pass.

## Reference numbers (default runs: pink noise)

Check counts when everything passes: Lab 1 11/11, Lab 2 8/8, Lab 3 10/10, Lab 4 8/8.

**Lab 1**

| | |
|---|---|
| frames in 10 s, 25 ms / 10 ms | **998** = 1 + (160000 − 400) // 160 |
| leakage above 3 kHz, 1030 Hz tone | rectangular −43.4 dB, Hann −111.3 dB |
| 1000 + 1040 Hz, level halfway between | 25 ms: 0.0 dB (one bump); 100 ms: −57.4 dB (two peaks) |

**Lab 2** — STFT shape (257, 507); STFT → iSTFT error 1.2e-7; at +5 dB the noise is louder than the
speech in **82 %** of time-frequency bins.

**Lab 3**

| SNR (dB) | PESQ | STOI | SI-SDR (dB) |
|---|---|---|---|
| 20 | 2.17 | 0.93 | 20.00 |
| 10 | 1.23 | 0.81 | 10.01 |
| 5 | 1.10 | 0.71 | 5.01 |
| 0 | 1.05 | 0.59 | 0.02 |
| −5 | 1.04 | 0.48 | −4.97 |
| −10 | 1.03 | 0.39 | −9.94 |

| Break it (5 dB) | plain SNR | PESQ | STOI | SI-SDR |
|---|---|---|---|---|
| as is | 5.00 | 1.10 | 0.71 | 5.01 |
| volume × 2 | −3.56 | 1.10 | 0.71 | 5.01 |
| 1 ms late | −3.46 | 1.10 | 0.71 | −27.27 |
| args swapped | 6.20 | 1.09 | 0.61 | 5.01 |

**Lab 4** — noisy phase (default) and clean phase (`--clean-phase`):

| SNR | input | PESQ | STOI | SI-SDR | PESQ, clean phase | SI-SDR, clean phase |
|---|---|---|---|---|---|---|
| 10 | noisy | 1.23 | 0.81 | 10.01 | | |
| 10 | IBM | 2.20 | 0.94 | 19.12 | 2.34 | 22.55 |
| 10 | IRM | 3.73 | 0.96 | 18.63 | 4.46 | 24.73 |
| 5 | noisy | 1.10 | 0.71 | 5.01 | | |
| 5 | IBM | 1.62 | 0.91 | 15.67 | 1.71 | 19.09 |
| 5 | IRM | 3.36 | 0.95 | 15.05 | 4.38 | 21.31 |
| 0 | noisy | 1.05 | 0.59 | 0.02 | | |
| 0 | IBM | 1.33 | 0.86 | 12.30 | 1.39 | 15.57 |
| 0 | IRM | 2.90 | 0.94 | 11.54 | 4.24 | 18.13 |
| −5 | noisy | 1.04 | 0.48 | −4.97 | | |
| −5 | IBM | 1.19 | 0.80 | 9.07 | 1.21 | 11.95 |
| −5 | IRM | 2.46 | 0.92 | 8.15 | 4.08 | 15.35 |

## Answers

**Q1 (windows).** Cutting out a frame multiplies the signal by a box. A 1030 Hz tone does not complete
a whole number of cycles in 25 ms, so the frame starts and ends in mid-swing — the hard edges are
jumps, and a jump contains energy at every frequency. That energy shows up everywhere in the spectrum
(about −43 dB here). Hann fades the frame to zero at both ends, so there are no jumps and leakage
falls to about −111 dB. Speech spans 50–60 dB between its loud low harmonics and its weak high
frequencies; leakage from the loud parts would fill in the quiet regions and the gaps between
harmonics, hiding exactly the detail a model needs (and making every bin look noisier than it is).

**Q2 (resolution).** 100 ms separates them (−57 dB dip); 25 ms merges them into one bump. The finest
frequency detail of a frame of length L is about 1/L (25 ms → 40 Hz bins, 100 ms → 10 Hz); with a Hann
window two tones need roughly 2/L between them (80 Hz for 25 ms, 20 Hz for 100 ms). Longer frames are
not better for speech because speech is not steady for 100 ms: phonemes change, pitch glides, a "t"
lasts a few ms. A long frame blurs those changes together (and adds delay in a real-time system).
Time detail and frequency detail trade off against each other.

**Q3 (where the noise is).** The SNR averages over the whole file, and speech energy is concentrated in
few bins: the low harmonics (roughly 100–1000 Hz) during vowels, plus fricative bursts. Pink noise is
spread evenly over all bins — every frequency, all the time, including pauses and the gaps between
harmonics. So the few speech bins carry most of the energy (speech wins on average), while most bins
are noise-dominated. Speech wins on the harmonics of voiced sounds and in strong fricatives. This
sparsity is the reason masks work: keep the speech-dominated bins, turn the rest down.

**Q4 (short vs long frames).** n_fft = 2048 (128 ms) shows the harmonics as sharp horizontal lines,
but onsets are smeared sideways. n_fft = 128 (8 ms) shows sharp vertical edges at every onset (and
individual glottal pulses as vertical striations), but the harmonics blur into broad bands. 512 (32 ms)
has 31 Hz bins — fine enough to resolve harmonics spaced by a typical F0 of 100–250 Hz — while staying
shorter than most phonemes.

**Q5 (phase).** 0: identical to the original. 1 (zero phase): buzzy and robotic — every frame becomes
a pulse centred in the frame, repeating every hop (8 ms → a 125 Hz buzz). 2 (random phase): rough,
noisy, whispery and smeared, like heavy reverberation. 3 (noisy phase): close to clean but rough,
with a noisy edge. Phase sets where each frequency component sits inside the frame; with the wrong
phase the overlapping frames no longer add up coherently. File 3 is the best that any method which
only cleans the magnitude can achieve — a perfect magnitude still keeps the noisy phase (see Q10).

**Q6 (PESQ vs STOI).** PESQ predicts *quality*: how a listener would rate the sound, and any audible
noise is penalized hard, so it is already near its floor (~1.0) by 10 dB and can no longer separate
the lower SNRs. STOI predicts *intelligibility*: it correlates the short-time envelopes of clean and
noisy speech in bands up to ~4 kHz over 384 ms segments, and speech can be fully understandable while
clearly noisy. At 5 dB you can follow every word (STOI 0.71) but it sounds poor (PESQ 1.10). A good
answer says the two measure different things, so a paper needs both.

**Q7 (break it).**
- *Volume × 2*: only plain SNR moved (5.0 → −3.6 dB: the "error" now contains a second copy of the
  speech). PESQ, STOI and SI-SDR are all level-independent: PESQ aligns levels, STOI normalizes each
  segment, SI-SDR rescales the reference with alpha.
- *1 ms late*: SI-SDR collapsed (5.0 → −27 dB) and plain SNR dropped; PESQ and STOI did not move.
  SI-SDR compares sample by sample, and a 16-sample shift is a large phase shift for everything above
  a few hundred Hz. PESQ searches for the delay and aligns before scoring; STOI's 384 ms envelopes
  don't notice 1 ms.
- *Args swapped*: SI-SDR is unchanged — it depends only on the correlation between the two signals,
  so it is symmetric. STOI dropped (0.71 → 0.61) because it is not symmetric: it removes silent frames
  based on the reference and normalizes/clips the estimate. PESQ moved only slightly here, but it is
  not symmetric either. Always call metric(reference, estimate).
- A model whose output is a few samples late (STFT padding, a convolution with latency) gets a
  terrible SI-SDR even if it sounds perfect. Check the alignment before you report SI-SDR.

**Q8 (oracle).** They need the clean speech S and the noise N separately, which you never have for a
real noisy recording — if you had them, there would be nothing left to enhance. They are good for
(1) an upper bound: the best that masking this STFT, with the noisy phase, can possibly do;
(2) a training target: next week a network learns to predict the IRM from the noisy spectrogram,
and the IRM is its label; (3) testing a pipeline, since if the oracle doesn't improve the metrics,
the STFT/iSTFT or the metric code is broken.

**Q9 (IBM vs IRM).** At 0 dB the IBM wins SI-SDR (12.3 vs 11.5 dB) but loses badly on PESQ (1.33 vs
2.90) and STOI (0.86 vs 0.94). The binary mask switches bins fully on or off, frame by frame; isolated
bins that survive in noise-dominated regions sound like random short tones ("musical noise"), and
speech below the threshold is cut away completely, so the result sounds choppy and harsh. The soft
IRM turns noisy bins down gradually. SI-SDR rewards removing energy aggressively, PESQ punishes the
artifacts — one more reason never to report a single metric.

**Q10 (clean phase).** IRM PESQ at +10 dB: 3.73 → 4.46 (+0.73). At −5 dB: 2.46 → 4.08 (+1.62). SI-SDR
gains +6.1 dB at +10 dB and +7.2 dB at −5 dB. The lower the SNR, the more the noisy phase costs: in
bins where the noise dominates, the phase of the mixture is mostly the noise's phase. So a
magnitude-only mask has a ceiling that drops at low SNR, which is why modern models (complex ratio
masks, models that predict real and imaginary parts) also try to fix the phase.

## Common mistakes to look for

| Lab | Symptom | Cause |
|---|---|---|
| 1.1 | paper answer 1000 or 997 | 1000 = T / hop (ignores the length of the last frame); 997 drops the first frame |
| 1.2 | shape (400, 160) instead of (998, 400) | `x.unfold(0, hop, frame_len)` — size and step swapped |
| 1.3 | peak check fails, shows ~17 dB | `10 * log10` of a magnitude — magnitudes need `20 * log10` |
| 1.3 | peak shows 40 dB and the leakage check fails | forgot to multiply the frame by `window` (so both spectra are rectangular) |
| 1.3 | bin 32 at 1.02 Hz | `k * n_fft / sr` instead of `k * sr / n_fft` |
| 2.1 | `RuntimeError: stft requires the return_complex parameter` | newer PyTorch requires `return_complex=True` |
| 2.3 | reconstruction error 0.22 instead of 1e-7 (output at half volume) | `torch.istft` called without the Hann window |
| 3.1 | "turning the volume down" fails | `target = ref` instead of `alpha * ref` |
| 3.2 | `ValueError: The truth value of an array ...` inside pesq | `pesq(ref, est, SR)` — the sample rate goes first |
| 3.2 | PESQ check fails, no error | `"nb"` mode, or ref and est swapped |
| 4.1 | `RuntimeError: ... not implemented for 'ComplexFloat'` | compared complex values; use `.abs()` |
| 4.3 | length check fails by a few samples | `istft` without `length=` |
