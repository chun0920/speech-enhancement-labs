# Report 3 — Causal Architecture Design in GTCRN (master's students)

Presented to Prof. Hung. About 10 minutes. Build it from your implementation in `lab2_causality.py`,
the GTCRN architecture (`gtcrn.py`), and real-time audio processing principles.

---

## 1. Executive Summary

In one sentence, explain why strict causality is required for real-time speech enhancement and what price in performance (e.g. PESQ) is typically paid compared to non-causal systems.

> 

---

## 2. Latency Budget in Real-Time Telecommunications

ITU-T recommendations (e.g. G.114) specify maximum acceptable mouth-to-ear delay for teleconferencing.

Break down the complete latency budget:
$$\text{Total Latency} = \text{Algorithmic Latency} + \text{Processing Time (RTF)} + \text{Audio Hardware Buffer} + \text{Network Delay}$$

| Component | Definition | GTCRN Value / Bound | Practical Constraint |
|---|---|---|---|
| Algorithmic Latency | Minimum buffering required before a frame can be processed | | One STFT window (512 samples); the hop (16 ms) is the per-frame compute deadline |
| Processing Latency | Time taken by CPU/GPU to compute forward pass for one frame | | Must finish within one hop (16 ms); measured in Week 8 |
| Frame Lookahead | Number of future frames needed by the network | 0 frames | 0 ms |

---

## 3. Causal Design Mechanisms in GTCRN

Detail how each block in GTCRN enforces temporal causality:

1. **Causal Convolution (SFE, Encoder, Decoder)**:
   - What padding is applied along the time axis ($T$)?
   - Why would standard symmetric padding `padding=(k-1)//2` violate causality?
2. **Dual-Path GRNN (DPGRNN)**:
   - Intra-RNN: Over which dimension does it run? Why is Intra-RNN allowed to be bidirectional?
   - Inter-RNN: Over which dimension does it run? Why must Inter-RNN be strictly unidirectional?
3. **STFT and iSTFT Synthesis**:
   - How does causal synthesis work without requiring future frames?

> 

---

## 4. The Future-Perturbation Verification Protocol

Describe the perturbation experiment implemented in `lab2_causality.py`:
1. What was the mathematical hypothesis?
2. How did the causal block and non-causal block compare when a future perturbation was injected?
3. What is the maximum difference observed on past output samples?

> 

---

## 5. Performance Trade-Off: Causal vs Non-Causal Baselines

Compare causal speech enhancement (such as GTCRN, DTLN, DPRNN) against non-causal offline models (such as BSRNN, MP-SENet, SepFormer):
1. Why do non-causal models consistently achieve higher PESQ (+0.3 to +0.6 higher)?
2. In which real-world products is causal processing non-negotiable?

> 

---

## 6. Questions for Discussion with Prof. Hung

List 2–3 questions regarding streaming neural architectures (e.g. state caching in ONNX/C++, chunked lookahead vs zero lookahead, or streaming conformer).

> 
