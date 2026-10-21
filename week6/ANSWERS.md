# Week 6 — Questions

Answer these questions as you complete Lab 1, Lab 2, and evaluate your training checkpoints.

---

### Q1. Checkpoint Selection Strategy
When selecting the final model checkpoint to report in Week 7, you could pick:
  (a) The checkpoint with lowest validation loss
  (b) The checkpoint with highest validation PESQ (`best_model_*.tar`)
  (c) The last epoch (`model_200.tar`)
Which one should you pick for a paper reproduction comparison, and why must you NEVER use the test set to choose the checkpoint?

> 

---

### Q2. Metric Progression Across Milestones
From your evaluations in `results/eval_summary.csv`:
How did PESQ, STOI, and SI-SDR improve as training progressed from Epoch 3 (Week 4) $\to$ Epoch 50 $\to$ Epoch 100 $\to$ Epoch 200? Did all metrics improve at the same rate, or did some plateau earlier?

> 

---

### Q3. Algorithmic Latency vs Processing Latency
1. What is the difference between "algorithmic latency" and "computational/processing latency"?
2. In GTCRN with $f_s = 16000\text{ Hz}$, $N_{fft} = 512$, and hop length $R = 256$, what is the algorithmic latency in milliseconds?

> 

---

### Q4. Causal Convolution Padding
In standard PyTorch `nn.Conv2d(..., kernel_size=(3, 3), padding=(1, 1))`, why does this violate temporal causality? How must the time-axis padding be modified to guarantee causality?

> 

---

### Q5. Dual-Path GRNN Causality
In GTCRN's DPGRNN block, there are two RNN submodules:
1. `intra_rnn`: Which axis does it process? Can it be bidirectional?
2. `inter_rnn`: Which axis does it process? Why must it be unidirectional?

> 

---

### Q6. Future-Perturbation Test
In `lab2_causality.py`, explain the logic of the future-perturbation test. If a network had a 2-frame lookahead ($32\text{ ms}$), how would the test result change?

> 

---

### Q7. Causal vs Non-Causal Performance Gap
Non-causal speech enhancement models (which look into the entire future of the utterance) typically score +0.3 to +0.5 higher in PESQ than causal models. Why does having access to future speech frames provide such a large advantage?

> 

---

### Q8. Status Check on lab112 Training Run
Record the status of your 200-epoch training run on lab112:
- Current epoch reached:
- Latest validation loss & PESQ:
- Estimated completion time:

> 
