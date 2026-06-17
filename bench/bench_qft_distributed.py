#!/usr/bin/env python3
"""
bench_qft_distributed.py — Phase A acceptance benchmark.

Runs n-qubit QFT on 4 Alveo U55C cards for n = 4, 6, 8, …, 28.
Emits results/qft_distributed.csv with columns:
    n, fidelity, norm_residual, time_s, cards, diag_gates_local

Acceptance gate:
    fidelity >= 1 - 1e-9 for n <= 22
    norm_residual < 1e-9 for n = 23..28 (Aer reference too large)

Author: Nasir Ali, C-DAC Noida
"""

import sys
import os
import csv
import time
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QISKIT_OK = True
except ImportError:
    QISKIT_OK = False

from host.distributed_statevector import (
    FPGADistributedSimulator, XRT_AVAILABLE
)


def build_qft_circuit(n: int) -> QuantumCircuit:
    """Standard QFT circuit on n qubits (no final swap reversal)."""
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, n):
            qc.cp(2 * np.pi / (2 ** (j - i + 1)), i, j)
    # Bit-reversal swaps (handled by swap-as-relabel in simulator,
    # but included here for reference; the simulator skips SWAP exchange
    # when both qubits are partition qubits via swap-as-relabel).
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    qc.measure_all()
    return qc


def aer_statevector(qc: QuantumCircuit) -> np.ndarray:
    """Run circuit on AerSimulator statevector method, return full statevector."""
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    sim = AerSimulator(method='statevector')
    from qiskit import transpile
    tc = transpile(qc_no_meas, sim)
    job = sim.run(tc)
    sv = job.result().get_statevector()
    return np.array(sv, dtype=np.complex128)


def fidelity(sv_fpga: np.ndarray, sv_ref: np.ndarray) -> float:
    """State fidelity = |<ψ_ref|ψ_fpga>|^2."""
    inner = np.vdot(sv_ref, sv_fpga)
    return float(np.abs(inner) ** 2)


def run_benchmark(
    xclbin_path: str,
    num_cards: int = 4,
    n_range=None,
    output_csv: str = None,
):
    if n_range is None:
        n_range = list(range(4, 29, 2))

    if not XRT_AVAILABLE:
        raise RuntimeError(
            "pyxrt not available — FPGA path requires XRT runtime. "
            "Run on a system with Xilinx XRT installed."
        )

    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    if output_csv is None:
        output_csv = os.path.join(ROOT, 'results', 'qft_distributed.csv')

    print(f"QFT distributed benchmark — {num_cards} cards, xclbin={xclbin_path}")
    print(f"n range: {n_range}")
    print(f"Output: {output_csv}")
    print()

    rows = []
    header = ['n', 'fidelity', 'norm_residual', 'time_s', 'cards',
              'diag_gates_local_pct']

    sim = FPGADistributedSimulator(xclbin_path=xclbin_path, num_cards=num_cards)

    try:
        for n in n_range:
            print(f"  n={n:2d}  building circuit...", end=' ', flush=True)
            qc = build_qft_circuit(n)

            t0 = time.time()
            result = sim.run(qc, shots=1)
            elapsed = time.time() - t0

            sv_fpga = result.get_statevector().astype(np.complex128)
            norm = float(np.linalg.norm(sv_fpga))
            norm_residual = abs(norm - 1.0)

            if n <= 22 and QISKIT_OK:
                sv_ref = aer_statevector(qc)
                # Align global phase
                f = fidelity(sv_fpga, sv_ref)
                print(f"fidelity={f:.6e}  norm_res={norm_residual:.2e}  t={elapsed:.1f}s")
            else:
                f = float('nan')
                print(f"norm_res={norm_residual:.2e}  t={elapsed:.1f}s  (no Aer ref for n>{22})")

            # Estimate diagonal-gate fraction for QFT
            # QFT has n*H + n*(n-1)/2 CP gates + n//2 SWAP gates
            total_gates = n + n * (n - 1) // 2 + n // 2
            diag_gates = n * (n - 1) // 2  # All CP gates are diagonal
            diag_pct = 100.0 * diag_gates / total_gates if total_gates > 0 else 0.0

            rows.append({
                'n': n,
                'fidelity': f if not np.isnan(f) else 'nan',
                'norm_residual': norm_residual,
                'time_s': elapsed,
                'cards': num_cards,
                'diag_gates_local_pct': f'{diag_pct:.1f}',
            })

            # Check acceptance gate
            if n <= 22 and not np.isnan(f) and f < 1 - 1e-9:
                print(f"  [FAIL] n={n}: fidelity {f:.6e} < 1-1e-9 — GATE NOT MET")
            elif n > 22 and norm_residual >= 1e-9:
                print(f"  [FAIL] n={n}: norm_residual {norm_residual:.2e} >= 1e-9 — GATE NOT MET")
    finally:
        sim.shutdown()

    with open(output_csv, 'w', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written: {output_csv}")
    print("\nAcceptance summary:")
    passed = 0
    for row in rows:
        n = row['n']
        if n <= 22:
            fid = row['fidelity']
            if fid != 'nan':
                ok = float(fid) >= 1 - 1e-9
                status = "PASS" if ok else "FAIL"
                if ok:
                    passed += 1
            else:
                status = "SKIP (no Aer)"
            print(f"  n={n:2d}: {status}  fidelity={fid}")
        else:
            nr = float(row['norm_residual'])
            ok = nr < 1e-9
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            print(f"  n={n:2d}: {status}  norm_residual={nr:.2e}")

    print(f"\n{passed}/{len(rows)} points passed acceptance gate.")
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QFT distributed benchmark')
    parser.add_argument('xclbin', help='Path to quantum_fpga_kernel_c128.xclbin')
    parser.add_argument('--cards', type=int, default=4, help='Number of FPGA cards')
    parser.add_argument('--n-min', type=int, default=4, help='Minimum qubit count')
    parser.add_argument('--n-max', type=int, default=28, help='Maximum qubit count')
    parser.add_argument('--n-step', type=int, default=2, help='Qubit count step')
    parser.add_argument('--output', default=None, help='Output CSV path')
    args = parser.parse_args()

    n_range = list(range(args.n_min, args.n_max + 1, args.n_step))
    run_benchmark(
        xclbin_path=args.xclbin,
        num_cards=args.cards,
        n_range=n_range,
        output_csv=args.output,
    )
