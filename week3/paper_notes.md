# Paper notes — GTCRN (master's students)

Rong et al., "GTCRN: A speech enhancement model requiring ultralow computational resources," ICASSP 2024,
pp. 971–975. Read the paper itself (not a summary of it), and the official repository's README.
Write in your own words; mark anything you did not understand with **?** — those are the questions for your
mentor and for Report 1 next week.

## 1. The problem

Who needs this model, and under which constraint (computation, memory, latency)? Which earlier models does
the paper position itself against, and why are they not good enough?

>

## 2. The model, block by block

Draw (by hand, photo is fine) the block diagram: from the noisy waveform to the enhanced waveform. For each
block write one line: what goes in, what comes out, and why the block is there. Use the paper's names
(ERB, SFE, encoder, DPGRNN / grouped RNN, decoder, mask).

>

## 3. What makes it small

List the design choices the paper credits for the low parameter count and MACs. For each: what it saves,
and what it might cost.

>

## 4. Training and testing

Training data, test data, sample rate, STFT settings, loss, optimizer, number of epochs or steps — whatever
the paper and the repository say. Mark what is **not** stated: Week 5 will have to decide those things.

>

## 5. The numbers you will compare with

Copy Tables 1 and 2 (the rows you need) with table number and page. Which metric implementation does the
paper use (PESQ wide-band or narrow-band? SI-SNR with or without mean removal? which DNSMOS)? Write
"not stated" where it does not say.

>

## 6. What I did not understand

>
