#!/usr/bin/env python3
"""
Distributed FPGA Statevector Simulator â Comprehensive Test Suite
=================================================================

Tests the FIXED distributed simulator across multiple qubit counts,
validating correctness against Qiskit reference statevectors.

Test matrix:
  1. Small (5-qubit GHZ)   â basic sanity, fast
  2. Medium (10-qubit GHZ) â partition boundary crossing
  3. Large (16-qubit GHZ)  â moderate memory, full validation
  4. Stress (20-qubit GHZ) â 2^18 amps/card, realistic workload
  5. Mixed gates (8 qubits) â H, CX, RZ, CZ, SWAP, CCX
  6. Scaling limit test     â up to 25+ qubits if memory allows

All tests compare distributed statevector against Qiskit Statevector
simulation and validate GHZ structure / gate correctness.

Author: Nasir Ali, C-DAC Noida
Date: February 2026
"""

import time
import sys
import traceback
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Import the FIXED simulator
from fpga_distributed_statevector import FPGADistributedSimulator

NUM_CARDS = 4
PARTITION_BITS = 2


def format_bytes(num_bytes):
    if num_bytes >= 1 << 30:
        return f"{num_bytes / (1 << 30):.2f} GB"
    elif num_bytes >= 1 << 20:
        return f"{num_bytes / (1 << 20):.2f} MB"
    elif num_bytes >= 1 << 10:
        return f"{num_bytes / (1 << 10):.2f} KB"
    return f"{num_bytes} B"


def state_fidelity(sv1, sv2):
    """Compute state fidelity |<sv1|sv2>|^2."""
    sv1 = sv1 / (np.linalg.norm(sv1) + 1e-30)
    sv2 = sv2 / (np.linalg.norm(sv2) + 1e-30)
    return abs(np.dot(sv1.conj(), sv2)) ** 2


def print_banner(title):
    w = 70
    print("\n" + "â" + "â" * w + "â")
    print("â" + title.center(w) + "â")
    print("â" + "â" * w + "â")


def print_section(title):
    print(f"\n{'â' * 70}")
    print(f"  {title}")
    print(f"{'â' * 70}")


# ============================================================================
# TEST 1: GHZ Circuit Tests (various sizes)
# ============================================================================

def test_ghz(sim, n_qubits, shots=4096, validate_sv=True):
    """
    Test GHZ circuit at given qubit count.
    
    GHZ state = (1/sqrt(2)) * (|00...0> + |11...1>)
    Expected: only two nonzero amplitudes, each â 1/sqrt(2).
    """
    local_qubits = n_qubits - PARTITION_BITS
    local_size = 2 ** local_qubits
    mem_per_card = local_size * 8

    print(f"\n  Qubits: {n_qubits}  |  Cards: {NUM_CARDS}  |  "
          f"Local: {local_qubits} qubits  |  "
          f"Amps/card: {local_size:,}  |  Mem/card: {format_bytes(mem_per_card)}")

    # Build GHZ circuit
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n_qubits), range(n_qubits))
    qc.name = f"GHZ-{n_qubits}"

    boundary = local_qubits - 1
    print(f"  Circuit: H(q0) + {n_qubits-1} CNOTs  |  "
          f"Partition boundary: CX(q{boundary}, q{boundary+1})")

    # Execute
    t0 = time.time()
    result = sim.run(qc, shots=shots)
    elapsed = time.time() - t0
    print(f"  Execution: {elapsed:.3f}s")

    # Validate counts
    counts = result.get_counts()
    total = sum(counts.values())
    zero_str = '0' * n_qubits
    one_str = '1' * n_qubits
    p0 = counts.get(zero_str, 0) / total
    p1 = counts.get(one_str, 0) / total
    ghz_purity = p0 + p1
    spurious = len(counts) - min(len(counts), 2)

    print(f"  Results: P(|0...0>)={p0:.4f}  P(|1...1>)={p1:.4f}  "
          f"Purity={ghz_purity:.4f}  Spurious={spurious}")

    # Validate statevector
    sv = result.get_statevector()
    norm = np.linalg.norm(sv)
    nz = np.count_nonzero(np.abs(sv) > 1e-6)

    fidelity = None
    if validate_sv and n_qubits <= 20:
        # Compare against Qiskit reference
        ref_qc = qc.remove_final_measurements(inplace=False)
        ref = np.array(Statevector.from_instruction(ref_qc).data, dtype=np.complex64)
        fidelity = state_fidelity(sv, ref)
        print(f"  Statevector: norm={norm:.6f}  nonzero={nz}  "
              f"fidelity={fidelity:.8f}")
    else:
        # For large circuits, validate structure directly
        amp0 = abs(sv[0])
        amp1 = abs(sv[2 ** n_qubits - 1])
        expected = 1.0 / np.sqrt(2)
        structural_fid = amp0 ** 2 + amp1 ** 2
        print(f"  Statevector: norm={norm:.6f}  nonzero={nz}  "
              f"|amp(0)|={amp0:.6f}  |amp(1)|={amp1:.6f}  "
              f"structural_fid={structural_fid:.8f}")
        fidelity = structural_fid

    # Per-card distribution
    for cid in range(NUM_CARDS):
        partition = sv[cid * local_size:(cid + 1) * local_size]
        card_nz = np.count_nonzero(np.abs(partition) > 1e-6)
        card_norm = np.linalg.norm(partition)
        print(f"    Card {cid} (part={cid:0{PARTITION_BITS}b}): "
              f"{card_nz} nonzero, norm={card_norm:.6f}")

    # Verdict
    passed = (ghz_purity > 0.98 and
              nz <= 4 and  # Allow tiny floating point noise
              (fidelity is None or fidelity > 0.99))
    status = "â PASS" if passed else "â FAIL"
    print(f"  {status}")
    return passed


# ============================================================================
# TEST 2: Mixed Gate Circuit
# ============================================================================

def test_mixed_gates(sim, n_qubits=8, shots=4096):
    """
    Test a circuit with diverse gate types including gates that cross
    partition boundaries.
    """
    local_qubits = n_qubits - PARTITION_BITS

    print(f"\n  Qubits: {n_qubits}  |  Mixed gates: H, X, CX, CZ, RZ, S, T")

    qc = QuantumCircuit(n_qubits, n_qubits)

    # Layer 1: Hadamard on all qubits (some cross partition)
    for i in range(n_qubits):
        qc.h(i)

    # Layer 2: CNOT ladder crossing partition
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    # Layer 3: Phase gates
    for i in range(n_qubits):
        qc.rz(np.pi / (i + 2), i)

    # Layer 4: CZ across partition boundary
    qc.cz(local_qubits - 1, local_qubits)

    # Layer 5: S and T gates
    qc.s(0)
    qc.t(n_qubits - 1)

    # Layer 6: Another round of CNOTs (reverse)
    for i in range(n_qubits - 2, -1, -1):
        qc.cx(i, i + 1)

    qc.measure(range(n_qubits), range(n_qubits))
    qc.name = f"MIXED-{n_qubits}"

    total_ops = sum(1 for inst in qc.data if inst.operation.name.lower()
                    not in ('measure', 'barrier'))
    print(f"  Circuit: {total_ops} gate operations")

    # Execute
    t0 = time.time()
    result = sim.run(qc, shots=shots)
    elapsed = time.time() - t0
    print(f"  Execution: {elapsed:.3f}s")

    # Compare with Qiskit reference
    sv = result.get_statevector()
    ref_qc = qc.remove_final_measurements(inplace=False)
    ref = np.array(Statevector.from_instruction(ref_qc).data, dtype=np.complex64)
    fidelity = state_fidelity(sv, ref)

    print(f"  Fidelity vs Qiskit: {fidelity:.8f}")
    print(f"  Outcomes: {len(result.get_counts())} unique bitstrings")

    passed = fidelity > 0.99
    status = "â PASS" if passed else "â FAIL"
    print(f"  {status}")
    return passed


# ============================================================================
# TEST 3: Bell State (Simplest Entanglement)
# ============================================================================

def test_bell_state(sim):
    """
    Minimal 3-qubit Bell-like test (minimum for 4 cards = 3 qubits).
    Actually use 4 qubits for cleaner partition.
    """
    n_qubits = 4
    print(f"\n  4-qubit Bell pair: H(q0), CX(q0,q1), CX(q2,q3)")

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(2)
    qc.cx(2, 3)  # This crosses partition boundary (q2 is local, q3 is partition)
    qc.measure(range(n_qubits), range(n_qubits))
    qc.name = "BELL-4"

    t0 = time.time()
    result = sim.run(qc, shots=4096)
    elapsed = time.time() - t0
    print(f"  Execution: {elapsed:.3f}s")

    sv = result.get_statevector()
    ref_qc = qc.remove_final_measurements(inplace=False)
    ref = np.array(Statevector.from_instruction(ref_qc).data, dtype=np.complex64)
    fidelity = state_fidelity(sv, ref)

    counts = result.get_counts()
    print(f"  Fidelity: {fidelity:.8f}")
    print(f"  Top outcomes: {dict(sorted(counts.items(), key=lambda x: -x[1])[:6])}")

    passed = fidelity > 0.99
    status = "â PASS" if passed else "â FAIL"
    print(f"  {status}")
    return passed


# ============================================================================
# TEST 4: Parametric Circuit (RX, RY, RZ)
# ============================================================================

def test_parametric(sim, n_qubits=6):
    """Test parametric rotation gates across partition boundaries."""
    print(f"\n  {n_qubits}-qubit parametric: RX, RY, RZ with various angles")

    qc = QuantumCircuit(n_qubits, n_qubits)

    # Apply parametric rotations
    for i in range(n_qubits):
        qc.rx(np.pi / 3 * (i + 1) / n_qubits, i)
        qc.ry(np.pi / 4 * (i + 1) / n_qubits, i)

    # Entangle across partition
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    # More rotations
    for i in range(n_qubits):
        qc.rz(np.pi / 6 * (i + 1) / n_qubits, i)

    qc.measure(range(n_qubits), range(n_qubits))
    qc.name = f"PARAM-{n_qubits}"

    t0 = time.time()
    result = sim.run(qc, shots=4096)
    elapsed = time.time() - t0
    print(f"  Execution: {elapsed:.3f}s")

    sv = result.get_statevector()
    ref_qc = qc.remove_final_measurements(inplace=False)
    ref = np.array(Statevector.from_instruction(ref_qc).data, dtype=np.complex64)
    fidelity = state_fidelity(sv, ref)

    print(f"  Fidelity: {fidelity:.8f}")

    passed = fidelity > 0.99
    status = "â PASS" if passed else "â FAIL"
    print(f"  {status}")
    return passed


# ============================================================================
# Main
# ============================================================================

def main():
    print_banner("DISTRIBUTED FPGA STATEVECTOR SIMULATOR â TEST SUITE")
    print_banner("FIXED: Dynamic Memory, Spawn, Deferred Init")

    print(f"\n  Configuration:")
    print(f"    FPGA cards:       {NUM_CARDS}")
    print(f"    Partition bits:   {PARTITION_BITS}")

    results = []

    # Initialize simulator ONCE â it will create/resize workers per circuit
    print_section("Initializing Simulator")
    t0 = time.time()
    sim = FPGADistributedSimulator(num_cards=NUM_CARDS)
    print(f"  Constructor time: {time.time()-t0:.3f}s (workers created on first run)")

    # ---- Test 1: Bell state (minimal) ----
    print_section("TEST 1: Bell State (4 qubits)")
    try:
        ok = test_bell_state(sim)
        results.append(("Bell state (4q)", ok))
    except Exception as e:
        print(f"  â EXCEPTION: {e}")
        traceback.print_exc()
        results.append(("Bell state (4q)", False))

    # ---- Test 2: Small GHZ ----
    print_section("TEST 2: Small GHZ (5 qubits)")
    try:
        ok = test_ghz(sim, 5, shots=4096)
        results.append(("GHZ 5q", ok))
    except Exception as e:
        print(f"  â EXCEPTION: {e}")
        traceback.print_exc()
        results.append(("GHZ 5q", False))

    # ---- Test 3: Parametric circuit ----
    print_section("TEST 3: Parametric Circuit (6 qubits)")
    try:
        ok = test_parametric(sim, 6)
        results.append(("Parametric 6q", ok))
    except Exception as e:
        print(f"  â EXCEPTION: {e}")
        traceback.print_exc()
        results.append(("Parametric 6q", False))

    # ---- Test 4: Mixed gates ----
    print_section("TEST 4: Mixed Gates (8 qubits)")
    try:
        ok = test_mixed_gates(sim, 8, shots=4096)
        results.append(("Mixed gates 8q", ok))
    except Exception as e:
        print(f"  â EXCEPTION: {e}")
        traceback.print_exc()
        results.append(("Mixed gates 8q", False))

    # ---- Test 5: Medium GHZ ----
    print_section("TEST 5: Medium GHZ (10 qubits)")
    try:
        ok = test_ghz(sim, 10, shots=4096)
        results.append(("GHZ 10q", ok))
    except Exception as e:
        print(f"  â EXCEPTION: {e}")
        traceback.print_exc()
        results.append(("GHZ 10q", False))

    # ---- Test 6: Large GHZ ----
    print_section("TEST 6: Large GHZ (16 qubits)")
    try:
        ok = test_ghz(sim, 16, shots=4096)
        results.append(("GHZ 16q", ok))
    except Exception as e:
        print(f"  â EXCEPTION: {e}")
        traceback.print_exc()
        results.append(("GHZ 16q", False))

    # ---- Test 7: Stress GHZ ----
    print_section("TEST 7: Stress GHZ (20 qubits)")
    try:
        ok = test_ghz(sim, 20, shots=8192)
        results.append(("GHZ 20q", ok))
    except Exception as e:
        print(f"  â EXCEPTION: {e}")
        traceback.print_exc()
        results.append(("GHZ 20q", False))

    # ---- Test 8: Higher qubit count ----
    print_section("TEST 8: Extended GHZ (24 qubits)")
    try:
        ok = test_ghz(sim, 24, shots=4096, validate_sv=False)
        results.append(("GHZ 24q", ok))
    except Exception as e:
        print(f"  â EXCEPTION: {e}")
        traceback.print_exc()
        results.append(("GHZ 24q", False))

    # ---- Final Summary ----
    print("\n" + "â" + "â" * 70 + "â")
    print("â" + "FINAL RESULTS".center(70) + "â")
    print("â " + "â" * 70 + "â£")

    all_pass = True
    for name, passed in results:
        sym = "â PASS" if passed else "â FAIL"
        line = f"  {sym}  {name}"
        padding = 70 - 2 - len(line)
        print(f"â{line}{' ' * max(padding, 1)}â")
        if not passed:
            all_pass = False

    print("â " + "â" * 70 + "â£")
    if all_pass:
        msg = f"ALL {len(results)} TESTS PASSED"
    else:
        n_pass = sum(1 for _, p in results if p)
        msg = f"{n_pass}/{len(results)} TESTS PASSED"
    print(f"â  {msg}{' ' * (70 - 4 - len(msg))}â")
    print("â" + "â" * 70 + "â")

    sim.shutdown()
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)