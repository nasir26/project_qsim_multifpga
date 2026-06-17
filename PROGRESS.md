# PROGRESS — append-only log

## 2026-06-17

### Scaffold
- [x] Directory tree created: `host/`, `kernels/statevector_c128/`, `kernels/mps/`,
      `src/`, `tests/`, `bench/`, `paper/figs/`, `results/`, `docs/legacy/`
- [x] Source files copied:
  - `host/distributed_statevector.py` ← `fpga_distributed_statevector12.py`
  - `host/mps_simulator.py` ← `fpga_mps_simulator.py`
  - `kernels/statevector_c128/quantum_gates_kernel_c64_ref.cpp` ← `quantum_gates_kernel.cpp`
  - `kernels/statevector_c128/quantum_fpga_kernel_complex128_v06.cpp` (reference — 8-bank)
  - `kernels/mps/fpga_mps_simulator.cpp`
  - `src/cpu_reference.py`
  - `tests/test_sv_distributed.py, final_test.py, final_test22.py, btest.py`
  - `bench/test_scaling_mps.py`
  - `paper/figs/` ← 5 PNG figures
  - `docs/legacy/` ← v10, v11 distributed SV
- [x] Git init on branch `phase-a-qft`
- [x] CLAUDE.md written

### Phase A bootstrap
- [x] PROGRESS.md, SCHEDULE.md, DECISIONS.md written
- [x] `scripts/setup_env.sh` and `scripts/check_env.py` written
- [x] `kernels/statevector_c128/quantum_fpga_kernel_c128.cpp` written
      (16-bank HBM, double-precision, 16+1=17 AXI masters — routing risk flagged)
- [x] `host/distributed_statevector.py` promoted to complex128 end-to-end
      + QFT diagonal-gate local path + swap-as-relabel
- [x] `bench/bench_qft_distributed.py` written (QFT n=4..28, 4 cards, CSV output)
- [x] `paper/main.tex` skeleton (sn-jnl, 11 sections, placeholder tables)

### PENDING — Phase A gate
- [ ] Run `tests/final_test22.py` on hardware
- [ ] Run `bench/bench_qft_distributed.py` → `results/qft_distributed.csv`
- [ ] Verify QFT fidelity ≥ 1−1e-9 for n≤22, norm residual < 1e-9 at n=28
- [ ] `git commit -m "phase-a: acceptance gate passed"` on branch `phase-a-qft`

### PENDING — Phase B
- [ ] Harden `host/mps_simulator.py` FPGA path for ≥500 qubits/card
- [ ] Run `bench/test_scaling_mps.py` → `results/mps_scaling.csv`
- [ ] Verify ≥500-qubit clean point for GHZ, 1Q-Rots, Brickwall
- [ ] Confirm ~4× throughput in 4-card shots mode
- [ ] `git commit -m "phase-b: mps 500q gate passed"` on branch `phase-b-mps`

### PENDING — Phase C
- [ ] Fill in `paper/main.tex` with real numbers from Phase A + B CSVs
- [ ] Generate combined exact-vs-MPS figure
- [ ] Clean pdflatex build
- [ ] `git commit -m "phase-c: paper skeleton complete"` on branch `phase-c-paper`
