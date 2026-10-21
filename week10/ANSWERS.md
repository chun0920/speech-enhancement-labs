# Week 10 — Questions

Final comprehensive reflection questions for your 10-week Speech Enhancement journey.

---

### Q1. Technical Obstacle & Debugging Reflection
Looking back across the entire 10-week curriculum (from Week 1 STFT fundamentals to Week 10 final acceptance):
What was the most challenging technical obstacle or unexpected bug you encountered (e.g. CUDA memory limits, data pipeline bottlenecks, metric discrepancies, or causality violations), and how did you diagnose and resolve it?

> 

---

### Q2. The Value of Exact Table 1 Reproduction
In deep learning research, why is rigorously reproducing Table 1 within a $\le 5\%$ error margin on an official test set an indispensable prerequisite before proposing architectural modifications or writing a thesis?

> 

---

### Q3. Causality as an Inviolable Engineering Constraint
Explain why temporal causality cannot be relaxed in interactive communication systems (e.g. cellular voice, VoIP, teleconferencing). If a model uses symmetric centered padding or future frames, what real-world failure occurs?

> 

---

### Q4. Model Size (Parameters) vs Computational Complexity (MACs / RTF)
Why is parameter count alone an unreliable proxy for inference latency? Give an example where a model with fewer parameters requires higher compute or runs slower than a model with more parameters.

> 

---

### Q5. Git Hygiene & Large Binary Bloat
In `lab1_audit_repo.py`, you enforced that no single file in the repository exceeds 10 MB and that checkpoints (`*.tar`) are excluded via `.gitignore`.
1. What permanent damage occurs to a Git repository if a student accidentally commits and pushes a 500 MB model checkpoint?
2. What are the proper industry and academic practices for storing and distributing large neural network weights?

> 

---

### Q6. Master's Thesis Research Proposal
Based on your findings over the past 10 weeks, formulate a concrete research hypothesis for your Master's thesis building upon GTCRN:
1. **Research Question / Hypothesis**: What specific limitation of GTCRN do you propose to overcome?
2. **Proposed Methodology**: What architectural or loss function modification will you implement?
3. **Evaluation Strategy**: Which datasets and metrics will prove your hypothesis?

> 
