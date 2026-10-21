# Week 8 — Questions

Answer these questions as you complete Lab 1 and Lab 2.

---

### Q1. Trainable vs Fixed ERB Parameters
In `lab1_profile.py`, you observed that GTCRN has 23,669 trainable parameters and 24,576 fixed (non-trainable) parameters, totaling 48,245 parameters.
1. What do the 24,576 fixed parameters represent?
2. Why did the authors choose to make the ERB filterbank weights non-trainable instead of learning them with standard backpropagation?

> 

---

### Q2. Architectural Sources of Low Complexity
GTCRN requires only 33.0 MMACs per second of audio.
Identify the three key architectural designs that allow GTCRN to keep its computational complexity so low while maintaining high speech enhancement performance (PESQ 2.87).

> 

---

### Q3. MACs vs FLOPs
1. What is the fundamental difference between a MAC (Multiply-Accumulate) operation and a FLOP (Floating-Point Operation)?
2. If GTCRN takes 33.0 MMACs per second, how many MFLOPs (or GFLOPs) per second does that correspond to?

> 

---

### Q4. Real-Time Factor (RTF) & Latency Budget
1. Define Real-Time Factor (RTF). What does an RTF of 0.04 mean in terms of processing speed?
2. In streaming mode, the STFT hop size is 256 samples at 16 kHz (16 ms). If the model's chunk execution latency is 2.5 ms, what fraction of the 16 ms budget is consumed, and why is this headroom critical for real-world devices?

> 

---

### Q5. CUDA Synchronization in Profiling
In `lab2_rtf.py`, why is `torch.cuda.synchronize()` called before `t0` and after `t1`? What would happen to your timing measurements if you omitted `torch.cuda.synchronize()` on a GPU?

> 

---

### Q6. Single-Thread vs Multi-Thread CPU RTF
Many embedded devices (earbuds, hearing aids, IoT smart speakers) have limited CPU cores and run audio enhancement on a single dedicated DSP/CPU thread.
Run `lab2_rtf.py` with `torch.set_num_threads(1)` versus default multi-threading. Does single-core GTCRN still achieve RTF < 1.0 on your machine?

> 

---

### Q7. End-to-End Streaming Latency Breakdown
List the four main latency components in an end-to-end real-time speech enhancement system from microphone capture to speaker playback, and work out GTCRN's algorithmic latency (window 512, hop 256, 16 kHz) and how much of the budget it uses.

> 
