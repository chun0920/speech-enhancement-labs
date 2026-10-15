# Week 1 — Answers

Name:

Answer in your own words. Two or three sentences each is enough, but every answer needs a *because*.

## Lab 1

**Q1.** Listen to `out/lab1_beeps_hard.wav` and `out/lab1_beeps_fade.wav`. What do you hear at the start
and end of each beep in the "hard" version? Why does an abrupt start or stop make that sound?
(Think: which frequencies does a sudden jump contain?)

>

## Lab 2

**Q2.** Look at the right panel of `out/lab2_training.png`. Which frequencies does the learned filter
let through, and which does it block? Why does that remove the rumble but keep the tones?

>

**Q3.** Run `python lab2_train_loop.py --forget-zero-grad`. What happens to the loss? Explain why
skipping `optimizer.zero_grad()` causes it.

>

**Q4.** Run `python lab2_train_loop.py --kernel 5`. Is the SNR improvement better or worse than with 63
weights? Look at the filter panel: what can't a 5-weight filter do?

>

**Q5.** Would a fixed filter like this one remove a *second person talking* in the background?
Why or why not?

>

## Lab 3

**Q6.** Listen to `out/phone_quality.wav` and compare it with `out/speech_16k.wav`, then look at
`out/lab3_spectrum.png`. What is missing? Which speech sounds suffer most? Why can't resampling
back up to 16 kHz bring it back?

>

**Q7.** Listen to `out/lab3_8bit.wav`. What do you hear? From the two round-trip SNRs that Lab 3
printed, roughly how many dB is each extra bit worth?

>

## Lab 4

**Q8.** In `out/week1_pairs.png` the clean target in the −10 dB pair is much smaller than in the
20 dB pair. Why? And why must the target be scaled by the *same* gain as the noisy input — what would
go wrong if you saved the target unscaled?

>

**Q9.** Listen to the four `out/pairs/noisy_*dB.wav` files. At which SNR does the speech stop being
easy to understand? Was it the SNR you expected?

>
