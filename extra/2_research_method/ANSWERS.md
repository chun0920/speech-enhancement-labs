# Week 4 — Answers

Name:

Answer in your own words; two or three sentences each. Quote the numbers your labs printed where a
question asks about them.

## Lab 1

**Q1.** Copy the reproduction lines of Lab 1 (seed 0 against the published Week 3 numbers). Did it
reproduce, by the 5 % / 10 % rule? Your code and seed are the same as in Week 3 — name two reasons
why the numbers are still not exactly the published ones.

>

**Q2.** How far apart are your three seeds on the Week 3 condition (largest − smallest PESQ and STOI)?
In Week 3 you concluded from ONE run of each that the GRU beats the MLP at −5 dB, where the gap was
about 0.04 PESQ and 0.01 STOI. What can you honestly say about that conclusion now?

>

**Q3.** Why is this week's exam (20 sentences, 20 speakers, two noises, two SNRs) a better exam than the
single Week 3 sentence? Name one thing it still cannot tell you.

>

## Lab 2

**Q4.** Paste the tally from `lab2_listen.py --tally`. What is the largest category, how large is its
share, and in which noise does it occur? Of the failures, how many could a human still follow? What
does that point to — and what does it NOT tell you?

>

## Lab 3

**Q5.** Write your hypothesis as "If …, then …, because …", and the cost you expect. Listen to
`out/lab3_examples/`: is the target still the loudest voice? Why does that matter for whether the
network can learn this at all?

>

**Q6.** In your Table 1 (design.md part 4): which code did you give the Week 3 GRU row, and why not C,
even though Week 3's answer key has its numbers? Why is the GTCRN row "—" this week? What would it
take to make it an R, and why would even that not be a fair comparison?

>

**Q7.** Why must `p_babble = 0` give exactly the Week 3 examples? What would it mean for Lab 4 if
it did not?

>

## Lab 4

**Q8.** Copy Table 1 from `results/table1.md` and the verdict. By the rule you wrote before running, does
the hypothesis hold? Is the gain larger than the spread between seeds? What did it cost on pink noise,
and why does it cost anything at all, with the same network?

>

**Q9.** Your prediction (design.md part 5) against the result: how far off were you? If you missed,
what did you not see when you made the prediction? (One sentence is enough — it is the most useful
sentence of the week.)

>

**Q10.** Listen to `out/compare/` and look at the analysis table (STOI by SNR). Where did the change help
and where not? What is still wrong at babble 0 dB? Propose the next ONE change: which cell of Table 1
should it move, and what would you write as its cost?

>
