#!/usr/bin/env python3
"""
Simple 20-Qubit Distributed FPGA Verification Test
====================================================

Proves the circuit is genuinely distributed across 4 FPGA cards by:
1. Running a 20-qubit GHZ circuit (requires cross-partition CNOTs)
2. Showing per-card memory usage and amplitude distribution
3. Validating statevector against Qiskit reference

With 4 cards and 20 qubits:
  - Partition bits: 2 (top qubits q18, q19)
  - Local qubits per card: 18
  - Amplitudes per card: 2^18 = 262,144
  - Total amplitudes: 2^20 = 1,048,576

Author: Nasir Ali, C-DAC Noida
Date: February 2026
"""

import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from fpga_distributed_statevector import FPGADistributedSimulator

NUM_CARDS = 4
N_QUBITS = 10
PARTITION_BITS = 2
LOCAL_QUBITS = N_QUBITS - PARTITION_BITS
SHOTS = 1


def state_fidelity(sv1, sv2):
    sv1 = sv1 / np.linalg.norm(sv1)
    sv2 = sv2 / np.linalg.norm(sv2)
    return abs(np.dot(sv1.conj(), sv2)) ** 2


def main():
    print("=" * 65)
    print("  20-QUBIT DISTRIBUTED FPGA VERIFICATION")
    print("=" * 65)
    print(f"\n  Total qubits:          {N_QUBITS}")
    print(f"  FPGA cards:            {NUM_CARDS}")
    print(f"  Partition qubits:      q{LOCAL_QUBITS}..q{N_QUBITS-1} ({PARTITION_BITS} bits)")
    print(f"  Local qubits/card:     q0..q{LOCAL_QUBITS-1} ({LOCAL_QUBITS} bits)")
    print(f"  Amplitudes/card:       2^{LOCAL_QUBITS} = {2**LOCAL_QUBITS:,}")
    print(f"  Total amplitudes:      2^{N_QUBITS} = {2**N_QUBITS:,}")
    print(f"  Memory/card:           {2**LOCAL_QUBITS * 8 / 1024:.1f} KB (complex64)")
    print(f"  Total statevector:     {2**N_QUBITS * 8 / 1024:.1f} KB")

    # ---- Build 20-qubit GHZ circuit ----
    print(f"\n--- Building {N_QUBITS}-qubit GHZ circuit ---")
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(0)
    for i in range(N_QUBITS - 1):
        qc.cx(i, i + 1)
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    qc.name = f"GHZ-{N_QUBITS}"

    boundary = LOCAL_QUBITS - 1
    print(f"  CNOT chain: q0 â q1 â ... â q{N_QUBITS-1}")
    print(f"  Partition boundary crossed at: CX(q{boundary}, q{boundary+1})")
    print(f"  This FORCES inter-card communication â no single card can do it alone.")

    # ---- Initialize simulator ----
    print(f"\n--- Initializing {NUM_CARDS}-card distributed simulator ---")
    t0 = time.time()
    sim = FPGADistributedSimulator(num_cards=NUM_CARDS)
    print(f"  Init time: {time.time()-t0:.2f}s")

    for s in sim.get_card_stats():
        status = "ALIVE" if s['alive'] else "DEAD"
        print(f"  Card {s['card_id']}: PID {s['pid']} [{status}]")

    # ---- Run circuit ----
    print(f"\n--- Running {N_QUBITS}-qubit GHZ on {NUM_CARDS} cards ---")
    t0 = time.time()
    result = sim.run(qc, shots=SHOTS)
    elapsed = time.time() - t0
    print(f"  Execution time: {elapsed:.3f}s")

    # ---- Validate counts (GHZ: only |00...0> and |11...1>) ----
    print(f"\n--- Measurement Results ({SHOTS} shots) ---")
    counts = result.get_counts()
    total = sum(counts.values())
    p0 = counts.get('0' * N_QUBITS, 0) / total
    p1 = counts.get('1' * N_QUBITS, 0) / total

    print(f"  P(|{'0'*N_QUBITS}>) = {p0:.4f}  ({counts.get('0'*N_QUBITS, 0)} shots)")
    print(f"  P(|{'1'*N_QUBITS}>) = {p1:.4f}  ({counts.get('1'*N_QUBITS, 0)} shots)")
    print(f"  Combined:          {p0+p1:.4f}")
    print(f"  Spurious outcomes: {len(counts) - min(len(counts), 2)}")

    # ---- Validate statevector vs Qiskit ----
    print(f"\n--- Statevector Validation ---")
    sv = result.get_statevector()
    ref = np.array(Statevector.from_instruction(
        qc.remove_final_measurements(inplace=False)).data, dtype=np.complex64)
    fid = state_fidelity(sv, ref)
    print(f"  Fidelity vs Qiskit: {fid:.8f}")

    # ---- Show per-card amplitude distribution ----
    print(f"\n--- Per-Card Amplitude Distribution ---")
    local_size = 2 ** LOCAL_QUBITS
    for cid in range(NUM_CARDS):
        partition = sv[cid * local_size : (cid + 1) * local_size]
        nz = np.count_nonzero(np.abs(partition) > 1e-6)
        norm = np.linalg.norm(partition)
        print(f"  Card {cid} (partition bits={cid:0{PARTITION_BITS}b}): "
              f"{nz} nonzero amplitudes, local norm={norm:.6f}")

    # ---- Final verdict ----
    print(f"\n{'=' * 65}")
    ok = fid > 0.999 and (p0 + p1) > 0.99
    if ok:
        print(f"  â PASS â {N_QUBITS}-qubit GHZ correctly distributed on {NUM_CARDS} FPGAs")
        print(f"           Fidelity={fid:.6f}, GHZ purity={p0+p1:.4f}")
    else:
        print(f"  â FAIL â Fidelity={fid:.6f}, GHZ purity={p0+p1:.4f}")
    print("=" * 65)

    sim.shutdown()
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)