# Week 3 — Answers

Name:

Answer in your own words; two or three sentences each. Quote the numbers your labs printed where a
question asks about them.

## Lab 1

**Q1.** Listen to a few files in `out/lab1_examples/`. Instead of saving a fixed set of noisy files
like in Week 1, the Dataset mixes a new example every time. What would go wrong if we trained on just
117 fixed noisy files, one per clean clip?

>

**Q2.** Lab 1 checks that a 10× louder input gives the same features. Why do we want that? What would
the network have to learn if the features changed with the volume?

>

## Lab 2

**Q3.** In `out/lab2_mlp.png`, compare the MLP's predicted mask with the ideal ratio mask. Where does
it go wrong? Why is that hard for a network that sees only one 32 ms frame at a time?

>

## Lab 3

**Q4.** The MLP and the GRU have the same number of weights and were trained on the same data, with
the same loss, for the same number of steps. Which one did better — in training loss and in the test
scores? What information can the GRU use that the MLP cannot, and why does it help against the
noise we trained on?

>

**Q5.** The GRU only looks backward in time. Why would a hearing aid or a phone call need that? What
could a model gain by also looking ahead, and what would it cost?

>

## Lab 4

**Q6.** On pink noise, both networks reach a large fraction of the SI-SDR ceiling but a much smaller
fraction of the PESQ ceiling (see the table in Lab 4). Listen to `out/exam/pink_0dB_GRU.wav` and
`out/exam/pink_0dB_oracle.wav`. What is still wrong with the network's output that PESQ notices?

>

**Q7.** On babble, both networks do almost nothing — look at the GRU mask in `out/week3_masks.png`.
Why? What did the networks actually learn about "what noise is" from the training data?

>

**Q8.** If you could change ONE thing to make the GRU handle babble, what would you change? Why do you
expect it to help, and how would you test that it was this change - and nothing else - that helped?

>
