# Week 4 — Answers

Name:

Answer in your own words; quote the line of code (file and line number) or the number your run printed.

## Reading the code

**Q1 (it does not run as published).** What is wrong with the import of the dataset in SEtrain's `train.py`
at commit `3ec437d`, and how did you find out? What does this tell you about running a paper's official code?

>

**Q2 (the window).** Which STFT window does `models/gtcrn_end2end.py` use, and which one does the official
`~/baselines/gtcrn/infer.py` use? Your mentor measured the official VCTK checkpoint both ways: PESQ 2.867 with
the window of `infer.py`, 2.837 with SEtrain's. What does that suggest about how the official checkpoint was
trained, and what will it mean for Week 7 if you train with SEtrain unchanged?

>

**Q3 (the loss).** `HybridLoss` adds three terms. For each: what does it compare, and what kind of mistake
does it punish? What do `compress_factor` and the weights 30 and 70 do? (Master's students: Report 2 next week
goes deeper.)

>

**Q4 (the learning rate).** The scheduler warms up for 25,000 steps and decays until step 250,000. With
10,000 training pairs and batch size 8, how many steps is one epoch? After how many epochs does the
warm-up end, and when does the decay end? What learning rate does your 3-epoch run reach?

>

**Q5 (the best checkpoint).** Which data and which number decide the "best" checkpoint in `train.py`?
Why must that never be the test set — and what would go wrong in Week 7 if it were?

>

**Q6 (shapes and weights).** From `results/layer_table.md`: the shape after `erb.erb_fc`, after `encoder`,
after the two `dpgrnn` blocks, and the final output. Why does the frequency axis go 257 → 129 → 33? The block
`erb` is missing from the top-level table although its layers ran — why? The model has 48,245 parameters, the
paper says 23.7 K: where does the difference come from (lab1_shapes.py printed the answer)?

>

## The run

**Q7 (three epochs).** Your training loss, validation loss and validation PESQ for each epoch, and the
validation scores of `model_003.tar` from se_eval. Did everything move in the right direction?

>

**Q8 (the full run).** From `training_plan.md`: minutes per epoch on your computer, the number of epochs
you would train, and the total hours. What do you need to decide before Week 5 (number of epochs, window,
which checkpoint you will report) — and who decides it?

>
