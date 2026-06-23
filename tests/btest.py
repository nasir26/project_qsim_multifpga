#!/usr/bin/env python3
"""
FPGA 4-Card Maximum Qubit Limit Test (FIXED)
==============================================

Tests the theoretical maximum qubit capacity when distributing
across 4 Xilinx Alveo U55C FPGA cards.

  Noiseless (statevector):
    Single card: 30 qubits (2^30 amplitudes Ã 8B = 8 GB HBM)
    4 cards:     32 qubits (2^30 amplitudes/card, +2 qubits from log2(4))

FIXES:
  - Uses the fixed FPGADistributedSimulator with dynamic memory allocation
  - Progressively scales up (doesn't jump straight to 32 qubits)
  - Validates at each scale point before attempting next

Author: Nasir Ali, C-DAC Noida
Date: February 2026
"""

import time
import sys
import numpy as np
from qiskit import QuantumCircuit


NUM_CARDS = 4
PARTITION_BITS = 2
NOISELESS_SINGLE_CARD_MAX = 30
NOISELESS_4CARD_MAX = NOISELESS_SINGLE_CARD_MAX + PARTITION_BITS  # 32


def format_bytes(num_bytes):
    if num_bytes >= 1 << 30:
        return f"{num_bytes / (1 << 30):.2f} GB"
    elif num_bytes >= 1 << 20:
        return f"{num_bytes / (1 << 20):.2f} MB"
    elif num_bytes >= 1 << 10:
        return f"{num_bytes / (1 << 10):.2f} KB"
    return f"{num_bytes} B"


def print_banner(title):
    w = 70
    print("\n" + "â" + "â" * w + "â")
    print("â" + title.center(w) + "â")
    print("â" + "â" * w + "â")


def print_section(title):
    print(f"\n{'â' * 70}")
    print(f"  {title}")
    print(f"{'â' * 70}")


def print_scaling_summary():
    print_banner("SCALING SUMMARY: ALVEO U55C (8 GB HBM2 PER CARD)")
    print(f"""
  âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
  â                NOISELESS (STATEVECTOR) MODE                     â
  â                                                                 â
  â  Statevector size: 2^N amplitudes Ã 8 bytes (complex64)         â
  â  Each card holds: 2^(N-k) amplitudes, k = log2(K)              â
  â                                                                 â
  â  Cards   k    Max N    Amps/Card         Memory/Card            â
  â  âââââ   ââ   âââââ    âââââââââ         âââââââââââ            â
  â    1      0     30     2^30 = 1.07B      8.00 GB                â
  â    2      1     31     2^30 = 1.07B      8.00 GB                â
  â    4      2     32     2^30 = 1.07B      8.00 GB  â TARGET     â
  â    8      3     33     2^30 = 1.07B      8.00 GB                â
  â   16      4     34     2^30 = 1.07B      8.00 GB                â
  â                                                                 â
  â  Formula: Max qubits = 30 + log2(K)                             â
  âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
""")


def test_ghz_at_scale(sim, n_qubits, shots=1024):
    """
    Run a GHZ circuit at the given qubit count and validate.
    Returns (passed, execution_time, details_dict).
    """
    local_qubits = n_qubits - PARTITION_BITS
    local_size = 2 ** local_qubits
    mem_per_card = local_size * 8
    total_mem = mem_per_card * NUM_CARDS

    print(f"\n  N={n_qubits}  |  Local={local_qubits}  |  "
          f"Amps/card={local_size:,}  |  Mem/card={format_bytes(mem_per_card)}  |  "
          f"Total={format_bytes(total_mem)}")

    # Build GHZ circuit
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n_qubits), range(n_qubits))
    qc.name = f"GHZ-{n_qubits}"

    # Execute
    t0 = time.time()
    result = sim.run(qc, shots=shots)
    elapsed = time.time() - t0

    # Validate
    counts = result.get_counts()
    total = sum(counts.values())
    p0 = counts.get('0' * n_qubits, 0) / total
    p1 = counts.get('1' * n_qubits, 0) / total
    ghz_purity = p0 + p1

    sv = result.get_statevector()
    amp0 = abs(sv[0])
    amp1 = abs(sv[2 ** n_qubits - 1])
    nz = np.count_nonzero(np.abs(sv) > 1e-6)
    structural_fid = amp0 ** 2 + amp1 ** 2

    print(f"  Time={elapsed:.3f}s  |  Purity={ghz_purity:.4f}  |  "
          f"Nonzero={nz}  |  Fidelity={structural_fid:.8f}")

    passed = ghz_purity > 0.98 and nz <= 4 and structural_fid > 0.99
    status = "â PASS" if passed else "â FAIL"
    print(f"  {status}")

    return passed, elapsed, {
        'n_qubits': n_qubits,
        'mem_per_card': mem_per_card,
        'total_mem': total_mem,
        'ghz_purity': ghz_purity,
        'structural_fid': structural_fid,
        'nonzero': nz,
        'execution_time': elapsed,
    }


def main():
    print("â" + "â" * 70 + "â")
    print("â" + "FPGA 4-CARD SCALING TEST (FIXED SIMULATOR)".center(70) + "â")
    print("â" + "Xilinx Alveo U55C Ã 4 â Progressive Scale-Up".center(70) + "â")
    print("â" + "â" * 70 + "â")

    print_scaling_summary()

    # Import the FIXED simulator
    from fpga_distributed_statevector import FPGADistributedSimulator

    print_section("Initializing Simulator")
    t0 = time.time()
    sim = FPGADistributedSimulator(num_cards=NUM_CARDS)
    print(f"  Constructor: {time.time()-t0:.3f}s")

    # Progressive scale-up: start small, increase to max
    scale_points = [5, 8, 10, 12, 16, 20, 24, 28, 30, 32]
    results = []

    print_banner("PROGRESSIVE SCALE-UP TEST")

    for n in scale_points:
        local_qubits = n - PARTITION_BITS
        mem_per_card = (2 ** local_qubits) * 8

        # Safety check: skip if memory would exceed system RAM
        total_mem = mem_per_card * NUM_CARDS * 3  # 3x for read+write+local
        available_gb = 64  # Conservative estimate; adjust for your system
        if total_mem > available_gb * (1024 ** 3):
            print(f"\n  Skipping {n} qubits: would need {format_bytes(total_mem)} "
                  f"(> {available_gb} GB available)")
            results.append((n, False, None, "SKIPPED (memory)"))
            continue

        print_section(f"SCALE POINT: {n} QUBITS")
        try:
            passed, elapsed, details = test_ghz_at_scale(sim, n, shots=1024)
            results.append((n, passed, details, "PASS" if passed else "FAIL"))
        except MemoryError:
            print(f"  â MemoryError at {n} qubits")
            results.append((n, False, None, "MemoryError"))
            break
        except Exception as e:
            print(f"  â Exception at {n} qubits: {e}")
            import traceback; traceback.print_exc()
            results.append((n, False, None, str(e)[:40]))
            # Don't break â try to continue

    # ---- Final Summary ----
    print("\n" + "â" + "â" * 70 + "â")
    print("â" + "SCALING RESULTS SUMMARY".center(70) + "â")
    print("â " + "â" * 70 + "â£")

    max_passed = 0
    for n, passed, details, status_str in results:
        sym = "â" if passed else "â"
        if details:
            mem_str = format_bytes(details['mem_per_card'])
            time_str = f"{details['execution_time']:.2f}s"
            fid_str = f"fid={details['structural_fid']:.4f}"
        else:
            mem_str = "â"
            time_str = "â"
            fid_str = status_str

        line = f"  {sym} {n:2d}q  |  mem/card={mem_str:>10s}  |  time={time_str:>8s}  |  {fid_str}"
        padding = 70 - 2 - len(line)
        print(f"â{line}{' ' * max(padding, 1)}â")

        if passed:
            max_passed = n

    print("â " + "â" * 70 + "â£")
    msg = f"Maximum verified: {max_passed} qubits on {NUM_CARDS} FPGA cards"
    print(f"â  {msg}{' ' * (70 - 4 - len(msg))}â")
    print("â" + "â" * 70 + "â")

    sim.shutdown()
    return max_passed >= 20  # Pass if we can do at least 20 qubits


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)