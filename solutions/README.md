# Solutions

Reference solutions for the weeks that are finished. Do the lab yourself first, then compare.

| Week | Folder | What is in it |
|---|---|---|
| 1 | [`week1/`](week1/) | finished `lab*.py` and [`ANSWER_KEY.md`](week1/ANSWER_KEY.md) |
| 2 | [`week2/`](week2/) | finished `lab*.py` and [`ANSWER_KEY.md`](week2/ANSWER_KEY.md) |

Each `ANSWER_KEY.md` has three parts:

- **Reference numbers**: what a correct run prints. In Week 2 everyone uses the same clip and noise
  seed, so your numbers should match these exactly; if they differ, there is a bug even when every
  check passes.
- **Answers** to the questions in `ANSWERS.md`. Your wording can be different; what matters is the
  *because*.
- **Common mistakes**: the symptom you see and what usually causes it.

Each folder runs on its own, the same way as the week it belongs to (same environment, same commands):

```bash
cp -r ~/speech-enhancement-labs/solutions/week2 ~/week2_solution
cd ~/week2_solution
python lab4_oracle_mask.py
```

To see exactly where your code differs from the reference:

```bash
diff ~/week2_<yourname>/lab3_metrics.py ~/speech-enhancement-labs/solutions/week2/lab3_metrics.py
```
