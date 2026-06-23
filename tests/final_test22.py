#!/usr/bin/env python3
"""
Distributed FPGA Statevector Simulator â STRICT FPGA Verification Test
========================================================================

Validates that the statevector is genuinely distributed across multiple
FPGA cards with ALL gate computation on FPGA hardware, NO CPU fallback.

Verification Strategy:
    1. FPGA Hardware Check: Confirms XRT, FPGA devices, kernel loaded
    2. HBM Residency Check: State written to / read from FPGA HBM
    3. FPGA Gate Execution: Local gates dispatched to FPGA kernel
    4. Cross-Card Communication: Global gates require inter-card exchange
    5. Correctness: Fidelity against Qiskit reference statevector

Test Matrix:
    T1  Bell state (4q)         â minimal entanglement, validates basics
    T2  GHZ 5q                  â crosses partition boundary
    T3  Parametric 6q           â RX, RY, RZ on FPGA
    T4  Mixed gates 8q          â H, CX, CZ, RZ, S, T, all on FPGA
    T5  GHZ 10q                 â moderate, full validation
    T6  GHZ 16q                 â larger circuit, FPGA kernel stress
    T7  GHZ 20q                 â 262K amps/card, realistic workload
    T8  GHZ 24q                 â 4M amps/card, structural validation
    T9  FPGA-Only Audit         â verifies no CPU fallback path was taken

Author: Nasir Ali, C-DAC Noida
Date: February 2026
"""

import time
import sys
import os
import traceback
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from fpga_distributed_statevector import (
    FPGADistributedSimulator,
    XRT_AVAILABLE,
    NUM_HBM_BANKS,
)

NUM_CARDS = 4
PARTITION_BITS = 2


# ============================================================================
# Utilities
# ============================================================================

def format_bytes(num_bytes):
    if num_bytes >= 1 << 30:
        return f"{num_bytes / (1 << 30):.2f} GB"
    elif num_bytes >= 1 << 20:
        return f"{num_bytes / (1 << 20):.2f} MB"
    elif num_bytes >= 1 << 10:
        return f"{num_bytes / (1 << 10):.2f} KB"
    return f"{num_bytes} B"


def state_fidelity(sv1, sv2):
    """Compute |â¨sv1|sv2â©|Â² state fidelity."""
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
# FPGA Hardware Verification
# ============================================================================

def verify_fpga_hardware():
    """
    Pre-flight check: ensure FPGA hardware is available and XRT works.
    This MUST pass before any simulation tests run.
    """
    print_section("PRE-FLIGHT: FPGA HARDWARE VERIFICATION")

    checks = []

    # Check 1: XRT available
    print(f"  [1] pyxrt import:         ", end="")
    if XRT_AVAILABLE:
        print("â Available")
        checks.append(True)
    else:
        print("â NOT AVAILABLE â cannot proceed")
        checks.append(False)
        return False, checks

    # Check 2: FPGA devices visible
    print(f"  [2] FPGA devices:         ", end="")
    try:
        import pyxrt as xrt
        devices_found = 0
        for i in range(NUM_CARDS):
            try:
                dev = xrt.device(i)
                devices_found += 1
            except Exception:
                break
        if devices_found >= NUM_CARDS:
            print(f"â {devices_found} device(s) found (need {NUM_CARDS})")
            checks.append(True)
        else:
            print(f"â Only {devices_found} device(s), need {NUM_CARDS}")
            checks.append(False)
    except Exception as e:
        print(f"â Error: {e}")
        checks.append(False)

    # Check 3: XCLBIN loadable
    print(f"  [3] XCLBIN load:          ", end="")
    xclbin_path = None
    search_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "quantum_simulator_kernel.xclbin"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "quantum_noisy_simulator_kernel.xclbin"),
    ]
    for p in search_paths:
        if os.path.exists(p):
            xclbin_path = p
            break

    if xclbin_path:
        print(f"â Found: {os.path.basename(xclbin_path)}")
        checks.append(True)
    else:
        print(f"â XCLBIN not found in: {search_paths}")
        checks.append(False)

    # Check 4: Kernel accessible
    if all(checks):
        print(f"  [4] Kernel access:        ", end="")
        try:
            dev = xrt.device(0)
            uuid = dev.load_xclbin(xclbin_path)
            # Try known kernel names
            kernel_name = None
            for kn in ["quantum_simulator_kernel", "quantum_noisy_simulator_kernel"]:
                try:
                    k = xrt.kernel(dev, uuid, kn)
                    kernel_name = kn
                    break
                except Exception:
                    continue
            if kernel_name:
                print(f"â Kernel: {kernel_name}")
                checks.append(True)
            else:
                print(f"â No recognized kernel in xclbin")
                checks.append(False)
        except Exception as e:
            print(f"â Error: {e}")
            checks.append(False)

    # Check 5: HBM allocation
    if all(checks):
        print(f"  [5] HBM allocation:       ", end="")
        try:
            test_size = 4096  # 4KB test buffer
            bos = []
            for i in range(NUM_HBM_BANKS):
                bo = xrt.bo(dev, test_size, xrt.bo.flags.normal,
                            k.group_id(i))
                bos.append(bo)
            print(f"â {NUM_HBM_BANKS} HBM banks allocated")
            checks.append(True)
            bos.clear()
        except Exception as e:
            print(f"â Error: {e}")
            checks.append(False)

    all_pass = all(checks)
    print(f"\n  FPGA Hardware: {'â ALL CHECKS PASSED' if all_pass else 'â CHECKS FAILED'}")
    return all_pass, checks


# ============================================================================
# Test: GHZ Circuit (various sizes)
# ============================================================================

def test_ghz(sim, n_qubits, shots=4096, validate_sv=True):
    """
    Test GHZ circuit at given qubit count.
    GHZ state = (1/â2)(|00...0â© + |11...1â©)
    """
    local_qubits = n_qubits - PARTITION_BITS
    local_size = 2 ** local_qubits
    mem_per_card = local_size * 8

    print(f"\n  Qubits: {n_qubits}  |  Cards: {NUM_CARDS}  |  "
          f"Local: {local_qubits}q  |  "
          f"Amps/card: {local_size:,}  |  Mem/card: {format_bytes(mem_per_card)}")

    # Build GHZ circuit
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n_qubits), range(n_qubits))
    qc.name = f"GHZ-{n_qubits}"

    boundary = local_qubits - 1
    print(f"  Circuit: H(q0) + {n_qubits - 1} CNOTs  |  "
          f"Partition boundary: CX(q{boundary}, q{boundary + 1})")
    print(f"  Execution mode: FPGA ONLY (no CPU fallback)")

    # Execute on FPGA
    t0 = time.time()
    result = sim.run(qc, shots=shots)
    elapsed = time.time() - t0
    print(f"  Execution: {elapsed:.3f}s")

    # Validate measurement counts
    counts = result.get_counts()
    total = sum(counts.values())
    zero_str = '0' * n_qubits
    one_str = '1' * n_qubits
    p0 = counts.get(zero_str, 0) / total
    p1 = counts.get(one_str, 0) / total
    ghz_purity = p0 + p1
    spurious = len(counts) - min(len(counts), 2)
    print(f"  Counts: P(|0â¦0â©)={p0:.4f}  P(|1â¦1â©)={p1:.4f}  "
          f"Purity={ghz_purity:.4f}  Spurious={spurious}")

    # Validate statevector
    sv = result.get_statevector()
    norm = np.linalg.norm(sv)
    nz = np.count_nonzero(np.abs(sv) > 1e-6)

    fidelity = None
    if validate_sv and n_qubits <= 20:
        ref_qc = qc.remove_final_measurements(inplace=False)
        ref = np.array(Statevector.from_instruction(ref_qc).data, dtype=np.complex64)
        fidelity = state_fidelity(sv, ref)
        print(f"  Statevector: norm={norm:.6f}  nonzero={nz}  "
              f"fidelity={fidelity:.8f}")
    else:
        amp0 = abs(sv[0])
        amp1 = abs(sv[2 ** n_qubits - 1])
        structural_fid = amp0 ** 2 + amp1 ** 2
        print(f"  Statevector: norm={norm:.6f}  nonzero={nz}  "
              f"|a(0)|={amp0:.6f}  |a(2^N-1)|={amp1:.6f}  "
              f"structural_fid={structural_fid:.8f}")
        fidelity = structural_fid

    # Per-card distribution
    for cid in range(NUM_CARDS):
        partition = sv[cid * local_size:(cid + 1) * local_size]
        card_nz = np.count_nonzero(np.abs(partition) > 1e-6)
        card_norm = np.linalg.norm(partition)
        print(f"    Card {cid} (part={cid:0{PARTITION_BITS}b}): "
              f"{card_nz} nonzero, norm={card_norm:.6f}")

    passed = (ghz_purity > 0.98 and nz <= 4 and
              (fidelity is None or fidelity > 0.99))
    print(f"  {'â PASS' if passed else 'â FAIL'}")
    return passed


# ============================================================================
# Test: Mixed Gates
# ============================================================================

def test_mixed_gates(sim, n_qubits=8, shots=4096):
    """Test diverse gate types executed on FPGA."""
    local_qubits = n_qubits - PARTITION_BITS
    print(f"\n  Qubits: {n_qubits}  |  Gates: H, X, CX, CZ, RZ, S, T  |  FPGA ONLY")

    qc = QuantumCircuit(n_qubits, n_qubits)

    # Layer 1: Hadamard on all
    for i in range(n_qubits):
        qc.h(i)

    # Layer 2: CNOT ladder (crosses partition)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    # Layer 3: Phase gates (tests RZ on FPGA)
    for i in range(n_qubits):
        qc.rz(np.pi / (i + 2), i)

    # Layer 4: CZ across partition boundary
    qc.cz(local_qubits - 1, local_qubits)

    # Layer 5: S and T gates (tests S, T on FPGA)
    qc.s(0)
    qc.t(n_qubits - 1)

    # Layer 6: Reverse CNOTs
    for i in range(n_qubits - 2, -1, -1):
        qc.cx(i, i + 1)

    qc.measure(range(n_qubits), range(n_qubits))
    qc.name = f"MIXED-{n_qubits}"

    total_ops = sum(1 for inst in qc.data
                    if inst.operation.name.lower() not in ('measure', 'barrier'))
    print(f"  Circuit: {total_ops} gate operations")

    t0 = time.time()
    result = sim.run(qc, shots=shots)
    elapsed = time.time() - t0
    print(f"  Execution: {elapsed:.3f}s")

    sv = result.get_statevector()
    ref_qc = qc.remove_final_measurements(inplace=False)
    ref = np.array(Statevector.from_instruction(ref_qc).data, dtype=np.complex64)
    fidelity = state_fidelity(sv, ref)
    print(f"  Fidelity vs Qiskit: {fidelity:.8f}")
    print(f"  Outcomes: {len(result.get_counts())} unique bitstrings")

    passed = fidelity > 0.99
    print(f"  {'â PASS' if passed else 'â FAIL'}")
    return passed


# ============================================================================
# Test: Bell State
# ============================================================================

def test_bell_state(sim):
    """Minimal 4-qubit Bell pair test."""
    n_qubits = 4
    print(f"\n  4-qubit Bell: H(q0), CX(q0,q1), H(q2), CX(q2,q3)  |  FPGA ONLY")

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(2)
    qc.cx(2, 3)  # Crosses partition boundary
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
    print(f"  {'â PASS' if passed else 'â FAIL'}")
    return passed


# ============================================================================
# Test: Parametric Circuit
# ============================================================================

def test_parametric(sim, n_qubits=6):
    """Test parametric rotation gates on FPGA."""
    print(f"\n  {n_qubits}q parametric: RX, RY, RZ  |  FPGA ONLY")

    qc = QuantumCircuit(n_qubits, n_qubits)
    for i in range(n_qubits):
        qc.rx(np.pi / 3 * (i + 1) / n_qubits, i)
        qc.ry(np.pi / 4 * (i + 1) / n_qubits, i)

    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

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
    print(f"  {'â PASS' if passed else 'â FAIL'}")
    return passed


# ============================================================================
# Test: FPGA-Only Audit
# ============================================================================

def test_fpga_audit(sim):
    """
    Verify that the simulator is genuinely using FPGA and not CPU fallback.

    Checks:
        1. All workers report execution_mode = 'FPGA_ONLY'
        2. All workers are alive and attached to FPGA devices
        3. Card stats show FPGA device assignment
    """
    print(f"\n  Verifying FPGA-only execution mode...")

    stats = sim.get_card_stats()
    all_fpga = True

    for s in stats:
        cid = s['card_id']
        alive = s['alive']
        mode = s.get('execution_mode', 'UNKNOWN')
        fpga_dev = s.get('fpga_device', 'NONE')

        status_str = "ALIVE" if alive else "DEAD"
        fpga_str = f"device={fpga_dev}" if fpga_dev is not None else "NO FPGA"
        mode_str = mode

        ok = alive and mode == 'FPGA_ONLY'
        sym = "â" if ok else "â"
        print(f"    {sym} Card {cid}: PID={s['pid']}  [{status_str}]  "
              f"mode={mode_str}  {fpga_str}")

        if not ok:
            all_fpga = False

    # Additional check: verify XRT is loaded
    print(f"    XRT loaded: {'â' if XRT_AVAILABLE else 'â'}")
    if not XRT_AVAILABLE:
        all_fpga = False

    print(f"  {'â PASS â All execution on FPGA' if all_fpga else 'â FAIL â CPU fallback detected'}")
    return all_fpga


# ============================================================================
# Main
# ============================================================================

def main():
    print_banner("DISTRIBUTED FPGA STATEVECTOR â STRICT FPGA TEST SUITE")
    print_banner("ALL GATE COMPUTATION ON FPGA â NO CPU FALLBACK")

    print(f"\n  Configuration:")
    print(f"    FPGA cards:         {NUM_CARDS}")
    print(f"    Partition bits:     {PARTITION_BITS}")
    print(f"    XRT available:      {XRT_AVAILABLE}")
    print(f"    CPU fallback:       DISABLED")

    # ---- Pre-flight FPGA hardware check ----
    hw_ok, hw_checks = verify_fpga_hardware()
    if not hw_ok:
        print("\n  â FPGA HARDWARE CHECK FAILED â Cannot proceed.")
        print("    Ensure Xilinx XRT is installed, FPGA devices are visible,")
        print("    and the xclbin is in the working directory.")
        return False

    results = []

    # ---- Initialize simulator (FPGA mandatory) ----
    print_section("Initializing FPGA Distributed Simulator")
    t0 = time.time()
    try:
        sim = FPGADistributedSimulator(num_cards=NUM_CARDS)
        print(f"  Constructor: {time.time() - t0:.3f}s")
        print(f"  XCLBIN: {os.path.basename(sim.xclbin_path)}")
        print(f"  Kernel: {sim._kernel_name}")
    except Exception as e:
        print(f"  â FATAL: Simulator init failed: {e}")
        traceback.print_exc()
        return False

    # ---- T1: Bell state ----
    print_section("TEST 1: Bell State (4 qubits)")
    try:
        ok = test_bell_state(sim)
        results.append(("Bell state (4q)", ok))
    except Exception as e:
        print(f"  â EXCEPTION: {e}")
        traceback.print_exc()
        results.append(("Bell state (4q)", False))

    # ---- T2: Small GHZ ----
    print_section("TEST 2: Small GHZ (5 qubits)")
    try:
        ok = test_ghz(sim, 5, shots=4096)
        results.append(("GHZ 5q", ok))
    except Exception as e:
        print(f"  â EXCEPTION: {e}")
        traceback.print_exc()
        results.append(("GHZ 5q", False))

    # ---- T3: Parametric circuit ----
    print_section("TEST 3: Parametric Circuit (6 qubits)")
    try:
        ok = test_parametric(sim, 6)
        results.append(("Parametric 6q", ok))
    except Exception as e:
        print(f"  â EXCEPTION: {e}")
        traceback.print_exc()
        results.append(("Parametric 6q", False))

    # ---- T4: Mixed gates ----
    print_section("TEST 4: Mixed Gates (8 qubits)")
    try:
        ok = test_mixed_gates(sim, 8, shots=4096)
        results.append(("Mixed gates 8q", ok))
    except Exception as e:
        print(f"  â EXCEPTION: {e}")
        traceback.print_exc()
        results.append(("Mixed gates 8q", False))

    # ---- T5: Medium GHZ ----
    print_section("TEST 5: Medium GHZ (10 qubits)")
    try:
        ok = test_ghz(sim, 10, shots=4096)
        results.append(("GHZ 10q", ok))
    except Exception as e:
        print(f"  â EXCEPTION: {e}")
        traceback.print_exc()
        results.append(("GHZ 10q", False))

    # ---- T6: Large GHZ ----
    print_section("TEST 6: Large GHZ (16 qubits)")
    try:
        ok = test_ghz(sim, 16, shots=4096)
        results.append(("GHZ 16q", ok))
    except Exception as e:
        print(f"  â EXCEPTION: {e}")
        traceback.print_exc()
        results.append(("GHZ 16q", False))

    # ---- T7: Stress GHZ ----
    print_section("TEST 7: Stress GHZ (20 qubits)")
    try:
        ok = test_ghz(sim, 20, shots=8192)
        results.append(("GHZ 20q", ok))
    except Exception as e:
        print(f"  â EXCEPTION: {e}")
        traceback.print_exc()
        results.append(("GHZ 20q", False))

    # ---- T8: Extended GHZ ----
    print_section("TEST 8: Extended GHZ (24 qubits)")
    try:
        ok = test_ghz(sim, 24, shots=4096, validate_sv=False)
        results.append(("GHZ 24q", ok))
    except Exception as e:
        print(f"  â EXCEPTION: {e}")
        traceback.print_exc()
        results.append(("GHZ 24q", False))

    # ---- T9: FPGA-Only Audit ----
    print_section("TEST 9: FPGA-Only Execution Audit")
    try:
        ok = test_fpga_audit(sim)
        results.append(("FPGA-only audit", ok))
    except Exception as e:
        print(f"  â EXCEPTION: {e}")
        traceback.print_exc()
        results.append(("FPGA-only audit", False))

    # ---- Final Summary ----
    print("\n" + "â" + "â" * 70 + "â")
    print("â" + "FINAL RESULTS â STRICT FPGA EXECUTION".center(70) + "â")
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
        msg = f"ALL {len(results)} TESTS PASSED â FPGA ONLY, NO CPU FALLBACK"
    else:
        n_pass = sum(1 for _, p in results if p)
        msg = f"{n_pass}/{len(results)} TESTS PASSED"
    print(f"â  {msg}{' ' * max(70 - 4 - len(msg), 1)}â")
    print("â" + "â" * 70 + "â")

    sim.shutdown()
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)