# SCHEDULE — phase gates and target dates

| Milestone | Gate condition | Target |
|-----------|----------------|--------|
| **Phase A scaffold** | CLAUDE.md + docs + scripts written | 2026-06-17 ✅ |
| **Phase A kernel written** | `quantum_fpga_kernel_c128.cpp` compiles (v++ sw_emu) | 2026-06-17 ✅ |
| **Phase A host written** | `distributed_statevector.py` c128 + QFT path complete | 2026-06-17 ✅ |
| **Phase A gate** | `tests/final_test22.py` PASS; QFT fidelity ≥ 1−1e-9 n≤22; 28q norm < 1e-9; CSV emitted | TBD — needs hardware |
| **Phase B gate** | MPS ≥ 500q/card GHZ + 1Q-Rots + Brickwall; 4× shots throughput; CSV emitted | TBD — after Phase A gate |
| **Phase C gate** | `pdflatex paper/main.tex` succeeds; figures present; all tables filled | TBD — after Phase B gate |

## Commit protocol
- Commit after each gate with message matching pattern in PROGRESS.md
- Branch per phase: `phase-a-qft`, `phase-b-mps`, `phase-c-paper`
- Do NOT merge to `main` until Phase C gate passes
