# DECISIONS — items requiring Ali's sign-off before finalising

Each entry has a status: **OPEN** (needs decision), **DECIDED** (resolved), or
**FLAGGED** (technical risk noted but proceeding with stated default).

---

## D-01 — SV kernel AXI master count (routing risk)  **FLAGGED**

**Issue**: The new `quantum_fpga_kernel_c128.cpp` uses 16 HBM data banks + 1
gate-sequence port = **17 AXI masters** on one U55C. The previous v06 kernel
caused a routing failure at 17 masters and was fixed by reducing to 8 banks.

**Default action**: Proceed with 16 banks as specified in the brief; add
`Congestion_SpreadLogic_high` strategy in `connectivity_c128.cfg`. If routing
still fails, drop to 8 banks (halves HBM bandwidth but removes routing risk).

**Decision needed from Ali**: Accept 16-bank routing risk, or pre-emptively
drop to 8 banks?

---

## D-02 — MPS qubit ceiling  **OPEN**

**Issue**: The brief says "500+ qubits/card" as the Phase B target. The v06
kernel `CHI_MAX_SVD=64` and `MAX_GATES_BRAM=512` may limit performance past
~600 qubits for chi_max=64 brickwall circuits.

**Default action**: Target 500 clean qubits for GHZ, 1Q-Rots, and Brickwall.
Report the practical ceiling found at chi_max=64 in the paper.

**Decision needed from Ali**: Should we push to 1024q? This requires a longer
benchmark run (~4h) and the paper claim changes from "500+" to "1024+".

---

## D-03 — Paper byline / author list  **OPEN**

**Default (reused from attached statevector paper)**:
- Nasir Ali, C-DAC, STPI Campus, Noida
- Abhishek Tiwari, C-DAC, STPI Campus, Noida
- Nadeem Khan, C-DAC, STPI Campus, Noida
- Imam Mazhar, C-DAC, STPI Campus, Noida
- Karan Islur, C-DAC, STPI Campus, Noida
- Himanshu Gupta, C-DAC, STPI Campus, Noida
- Sachin Kumar, Department of Computer Science, University of Delhi
- Pankaj Tyagi, Cluster Innovation Centre, University of Delhi
- Rahul Kumar Neiwal, MeitY, New Delhi

**Decision needed from Ali**: Confirm or modify author list and order before
submission. Corresponding author email?

---

## D-04 — Target venue  **OPEN**

**Brief says**: Springer-Nature class paper (`sn-jnl`).

**Options**:
- *Quantum Information Processing* (Springer, Q1, 2–3 month review)
- *Journal of Supercomputing* (Springer, Q2, fast track)
- *Nature Electronics* (high bar, requires broader impact framing)

**Decision needed from Ali**: Which journal? This affects word-count limit,
figure count, and abstract framing.

---

## D-05 — QFT fidelity reference implementation  **FLAGGED**

**Issue**: The acceptance gate requires "QFT fidelity ≥ 1−1e-9 for n≤22".
The reference is Qiskit AerSimulator (statevector). For n>24 the Aer reference
requires ≥ 256 GB RAM. For 28q we compare norm‖SV‖ rather than fidelity.

**Default action**: n≤22: fidelity vs Aer. n=23..28: norm residual only.

**Decision needed from Ali**: Is norm residual sufficient for the 28q result, or
should we run on a high-memory server for the full fidelity check?

---

## D-06 — 4-card shots-parallel throughput metric  **OPEN**

**Issue**: 4× throughput claim requires showing wall-clock time vs 1-card
baseline. The brief asks for "~4× throughput" in shots mode.

**Default action**: Measure 4-card vs 1-card at n=20, shots=1024, chi_max=32.
If the speedup is 3.6–4.0×, report as "~4×". If lower, report actual value.

**Decision needed from Ali**: Minimum speedup ratio acceptable for the paper
claim? (Suggest: 3.5×.)
