# Week 2 — Answers

Name:

Answer in your own words; two or three sentences each. Quote the numbers your labs printed where
a question asks about them.

## Lab 1

**Q1.** In `out/lab1_windows.png` the rectangular frame spreads the 1030 Hz tone over the whole
spectrum; the Hann window does not. Why does cutting a frame out with hard edges do that? Why would
it be a problem for speech, where loud and quiet frequencies sit side by side?

>

**Q2.** In `out/lab1_resolution.png`, which frame length separates the tones at 1000 Hz and 1040 Hz?
Roughly how far apart must two tones be for a frame of length L to separate them? Why don't we simply
use 100 ms frames for speech?

>

## Lab 2

**Q3.** Lab 2 printed that the noise is louder than the speech in most time-frequency bins, although
over the whole file the speech is 5 dB louder. Using `out/lab2_spectrograms.png`, explain how both can
be true. Where does the speech win?

>

**Q4.** In `out/lab2_resolution.png`, which picture shows the harmonics of the voice (horizontal
stripes), and which shows exactly when each sound starts? What is n_fft = 512 a compromise between?

>

**Q5.** Listen to the four files in `out/phase/`. All four have exactly the same magnitude. Describe
what each sounds like and explain why phase matters. What does file 3 (noisy phase) tell you about a
method that only cleans the magnitude?

>

## Lab 3

**Q6.** In `out/lab3_metrics.csv`, PESQ is already low at 10 dB while STOI is still fairly high. What
does each metric try to predict that explains the difference? Does it match what you hear in
`out/lab2_noisy_5dB.wav`?

>

**Q7.** In the "break it on purpose" table, which numbers changed for *volume × 2*, for *1 ms late*,
and for *args swapped*? Explain each. What would happen if you scored a model whose output is a few
samples late with SI-SDR?

>

## Lab 4

**Q8.** Why are these masks called *oracle* masks? Could you compute them for a real noisy recording?
If not, what are they good for?

>

**Q9.** At 0 dB, which mask wins on SI-SDR, and which on PESQ and STOI? Listen to
`out/oracle/ibm_0dB.wav` and `out/oracle/irm_0dB.wav`: what does the binary mask do that you can hear?

>

**Q10.** Compare `out/week2_oracle.csv` with `out/week2_oracle_cleanphase.csv`. How much does the IRM's
PESQ improve with the clean phase at +10 dB, and at −5 dB? What does that say about re-using the noisy
phase when the SNR is low?

>
