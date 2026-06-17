# qsim_multifpga — Session Continuity

**Fresh session? Read this file, PROGRESS.md, and SCHEDULE.md before anything else.**

## Project summary

Two complementary FPGA quantum-circuit simulators on 4× Xilinx Alveo U55C,
developed into one paper:

1. **Distributed exact statevector** (`host/distributed_statevector.py` +
   `kernels/statevector_c128/`) — complex128, 16-bank HBM, top-k partition,
   QFT showcase at 28 qubits across 4 cards.
2. **MPS simulator** (`host/mps_simulator.py` +
   `kernels/mps/fpga_mps_simulator.cpp`) — 9 AXI masters, host SVD, 500+
   qubits/card for low-entanglement circuits.

## Three-phase plan

| Phase | Goal | Branch | Gate |
|-------|------|--------|------|
| A | Distributed SV c128, 28q, QFT showcase | `phase-a-qft` | `tests/final_test22.py` passes |
| B | MPS ≥ 500q/card, harden, 4× shots throughput | `phase-b-mps` | GHZ/1Q-Rots/Brickwall ≥ 500q |
| C | One unified paper (`paper/main.tex`) | `phase-c-paper` | clean pdflatex build |

Do NOT start Phase C until A and B both pass their gates.

## Hard constraints — never violate

1. **≤ 9 AXI masters on MPS kernel** (`bank0..bank7` + `gate_seq`). 17
   masters caused routing failure in v06 — non-negotiable for the MPS kernel.
   The SV kernel uses 16+1=17; note routing risk in DECISIONS.md.

2. **`mp.set_start_method('spawn', force=False)`** must be the first
   multiprocessing call in every host file — prevents fork()+XRT conflict.

3. **`ctypes.memmove()`** for ALL cross-process shared-memory I/O. NumPy
   buffer views on `mp.Array` return stale cached data across processes.

4. **engine='fpga' → RuntimeError, never silent CPU fallback** in the
   statevector path. CPU is only allowed as an explicit oracle.

5. **complex128 everywhere** in all new work. Shared-memory Array type:
   `ctypes.c_double` (float64). Bytes per amplitude: 16 (2 × float64).

6. **Partition convention**: top `k = log2(K)` qubits select the card.
   `card_id j` owns amplitudes where bits `[n-k .. n-1]` of the index equal `j`.

7. **Diagonal gate insight (QFT)**: any diagonal gate (Z, S, T, RZ, P, CZ, CP,
   RZZ, …) on a partition qubit is applied *locally* — no cross-card exchange.
   Non-diagonal partition-qubit gates use the 3-barrier double-buffered exchange.

8. **Swap-as-relabel**: QFT terminal bit-reversal swaps touching partition
   qubits are implemented as an index permutation on read-out, not physical
   amplitude movement.

9. **NEVER use "Qniverse" or "NQM" anywhere** — see memory constraint file.
   Author: Nasir Ali, Org: C-DAC Noida ONLY in code headers.

## Key file locations

```
host/distributed_statevector.py   — canonical c128 distributed SV host
host/mps_simulator.py             — canonical MPS host
kernels/statevector_c128/         — c128 HBM SV HLS kernel
kernels/mps/fpga_mps_simulator.cpp — MPS HLS kernel (9 AXI masters)
src/cpu_reference.py              — complex128 MPS oracle
tests/final_test22.py             — FPGA-only audit (SV)
tests/final_test.py               — 10q distributed GHZ
bench/bench_qft_distributed.py    — QFT 4→28q distributed benchmark
bench/test_scaling_mps.py         — MPS scaling benchmark
paper/main.tex                    — LaTeX paper
results/                          — CSVs for all benchmarks
docs/DECISIONS.md                 — items needing Ali's sign-off
```

## xclbin build commands

```bash
# SV kernel (c128, 16-bank):
cd kernels/statevector_c128
v++ -t hw --platform $PLATFORM -c quantum_fpga_kernel_c128.cpp \
    -o quantum_fpga_kernel_c128.xo
v++ -t hw --platform $PLATFORM -l quantum_fpga_kernel_c128.xo \
    --config connectivity_c128.cfg \
    -o quantum_fpga_kernel_c128.xclbin

# MPS kernel (already built at /home/abhishek/fpga_mps_simulator/fpga_mps_simulator.xclbin):
# cp /home/abhishek/fpga_mps_simulator/fpga_mps_simulator.xclbin kernels/mps/
```

## Environment activation

```bash
source ~/qsim_multifpga/.venv/bin/activate
source /opt/xilinx/xrt/setup.sh
export PYTHONPATH=/opt/xilinx/xrt/python:$PYTHONPATH
source /tools/Xilinx/Vitis/2023.2/settings64.sh
export PLATFORM=xilinx_u55c_gen3x16_xdma_3_202210_1
```
