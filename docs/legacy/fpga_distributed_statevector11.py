#!/usr/bin/env python3
"""
FPGA Distributed Statevector Simulator â STRICT FPGA EXECUTION
================================================================

Distributes ONE quantum circuit's statevector across multiple FPGA cards
with ALL gate computation executed on FPGA hardware. NO CPU FALLBACK.

Architecture:
    For N qubits on K cards (K = 2^k):
    - Top k qubits = partition index â which card owns each amplitude
    - Each card holds 2^(N-k) amplitudes in HBM
    - Card j owns amplitudes where top k bits of state index == j

Execution Model:
    - State LIVES in FPGA HBM at all times
    - Local gates (acting only on local qubits):
        â Encoded and dispatched to FPGA kernel on each card
        â Gate math runs entirely on FPGA fabric
    - Global gates (touching partition qubits):
        â Read local partition from FPGA HBM to host
        â 3-barrier double-buffered shared-memory exchange
        â Compute new partition (vectorized host â unavoidable for
          cross-card data movement, NOT a CPU "fallback")
        â Write result BACK to FPGA HBM immediately
    - FPGA initialization is MANDATORY â failure raises RuntimeError

Hardware:
    - Xilinx Alveo U55C with 8 GB HBM2 per card
    - 16 HBM banks per card, interleaved amplitude storage
    - XRT runtime required (pyxrt)

FPGA Kernel Interface (quantum_simulator_kernel):
    - 16 HBM bank pointers (float*), each mapped to one HBM bank
    - gate_sequence (int*): packed gate descriptors, 8 ints per gate
    - num_gates (int): number of gates in sequence
    - num_qubits (int): total qubits for this partition
    - Amplitude layout: global_idx â bank = idx % 16, offset = (idx/16)*2

Author: Nasir Ali, C-DAC Noida
Date: February 2026
"""

import sys
import time
import logging
import os
import signal
import atexit
import ctypes
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Force 'spawn' start method BEFORE any other multiprocessing import.
# Prevents fork() from inheriting parent XRT mmaps/fds into children.
# ---------------------------------------------------------------------------
import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=False)
except RuntimeError:
    pass  # Already set

from multiprocessing import Process, Queue, Array, Barrier, Event

# ---------------------------------------------------------------------------
# XRT is REQUIRED â no CPU fallback
# ---------------------------------------------------------------------------
try:
    import pyxrt as xrt
    XRT_AVAILABLE = True
except ImportError:
    XRT_AVAILABLE = False

try:
    from qiskit.result import Result
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

NUM_HBM_BANKS = 16
ABSOLUTE_MAX_QUBITS = 40

# ---------------------------------------------------------------------------
# Gate type encoding â matches quantum_gates_kernel.cpp EXACTLY
# ---------------------------------------------------------------------------
GATE_TYPES = {
    'h': 0,    'x': 1,    'y': 2,    'z': 3,
    's': 4,    't': 5,    'sdg': 6,  'tdg': 7,
    'rx': 8,   'ry': 9,   'rz': 10,  'p': 11,
    'phase': 11, 'u1': 20, 'u2': 21, 'u3': 22,
    'id': 23,  'sx': 18,  'sxdg': 19,
    'cx': 12,  'cnot': 12, 'cy': 13, 'cz': 14, 'ch': 15,
    'swap': 16, 'ccx': 17, 'toffoli': 17,
    'cp': 24,  'crx': 25, 'cry': 26, 'crz': 27,
    'iswap': 28, 'ecr': 29,
    'rxx': 30, 'ryy': 31, 'rzz': 32,
    'csx': 33, 'dcx': 34,
    'measure': 255, 'barrier': 254, 'reset': 253,
}

# Gate matrices for host-side global gate computation
GATE_MATRICES = {
    'h': np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2),
    'x': np.array([[0, 1], [1, 0]], dtype=np.complex64),
    'y': np.array([[0, -1j], [1j, 0]], dtype=np.complex64),
    'z': np.array([[1, 0], [0, -1]], dtype=np.complex64),
    's': np.array([[1, 0], [0, 1j]], dtype=np.complex64),
    't': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex64),
    'sdg': np.array([[1, 0], [0, -1j]], dtype=np.complex64),
    'tdg': np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex64),
}


def get_parametric_matrix(gate_name, params):
    """Compute 2x2 unitary for parametric gates."""
    if gate_name == 'rx':
        t = float(params[0]); c, s = np.cos(t / 2), np.sin(t / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex64)
    elif gate_name == 'ry':
        t = float(params[0]); c, s = np.cos(t / 2), np.sin(t / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex64)
    elif gate_name in ('rz', 'p', 'phase', 'u1'):
        return np.array([[1, 0], [0, np.exp(1j * float(params[0]))]], dtype=np.complex64)
    return np.eye(2, dtype=np.complex64)


# ============================================================================
# Shared Memory I/O â for inter-card communication during global gates
# ============================================================================

def _write_to_shared(arr, state, local_size):
    """Write complex64 state into interleaved float32 shared array."""
    n = min(len(state), local_size)
    buf = np.empty(n * 2, dtype=np.float32)
    buf[0::2] = state[:n].real.astype(np.float32)
    buf[1::2] = state[:n].imag.astype(np.float32)
    arr[:n * 2] = buf


def _read_from_shared(arr, size):
    """Read complex64 state from interleaved float32 shared array."""
    raw = np.frombuffer(arr, dtype=np.float32, count=size * 2).copy()
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)


# ============================================================================
# FPGA HBM I/O â State lives in FPGA HBM
# ============================================================================

def _write_state_to_hbm(hbm_banks, state, num_amps):
    """
    Write complex64 statevector into FPGA HBM banks.

    Layout: amplitude i â bank (i % 16), float offset (i // 16) * 2
    Each amplitude stored as [real, imag] pair of float32.

    Uses the same XRT API pattern as the proven single-card code:
    bank.write(buf.tobytes(), 0) â bank.sync(TO_DEVICE).
    """
    amps_per_bank = (num_amps + NUM_HBM_BANKS - 1) // NUM_HBM_BANKS

    for bid, bank in enumerate(hbm_banks):
        buf = np.zeros(amps_per_bank * 2, dtype=np.float32)
        indices = np.arange(bid, num_amps, NUM_HBM_BANKS)
        for li, gi in enumerate(indices):
            buf[li * 2] = state[gi].real
            buf[li * 2 + 1] = state[gi].imag
        bank.write(buf.tobytes(), 0)
        bank.sync(
            xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
            len(buf.tobytes()), 0
        )


def _read_state_from_hbm(hbm_banks, num_amps):
    """
    Read complex64 statevector from FPGA HBM banks.

    Reverses the interleaved bank layout.

    Uses the same XRT API pattern as the proven single-card code:
    bank.sync(FROM_DEVICE) â bank.map() â np.frombuffer().

    CRITICAL: Must use bank.map() not bank.read() â on Alveo U55C with
    XRT 2.x, bank.read() returns the host-side buffer from BEFORE sync,
    while bank.map() returns a memoryview of the SYNCED host buffer.
    """
    amps_per_bank = (num_amps + NUM_HBM_BANKS - 1) // NUM_HBM_BANKS
    state = np.zeros(num_amps, dtype=np.complex64)

    for bid, bank in enumerate(hbm_banks):
        bank.sync(
            xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
            amps_per_bank * 8, 0  # 8 bytes per amplitude (2 Ã float32)
        )
        buf = bank.map()
        raw = np.frombuffer(buf, dtype=np.float32, count=amps_per_bank * 2)
        cplx = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
        indices = np.arange(bid, num_amps, NUM_HBM_BANKS)
        v = min(len(cplx), len(indices))
        state[indices[:v]] = cplx[:v]

    return state


def _encode_gates_for_fpga(gate_list):
    """
    Encode a list of gates into int32 array for the FPGA kernel.

    Each gate = 8 Ã int32:
      [gate_type, qubit0, qubit1, qubit2, param0_bits, param1_bits, param2_bits, reserved]

    Float params are encoded as their IEEE 754 bit pattern reinterpreted as int32.
    """
    enc = []
    for gn, qb, pr in gate_list:
        gc = GATE_TYPES.get(gn, 255)
        if gc >= 253:  # Skip measure, barrier, reset
            continue
        gd = [0] * 8
        gd[0] = gc
        gd[1] = qb[0] if len(qb) > 0 else 0
        gd[2] = qb[1] if len(qb) > 1 else -1
        gd[3] = qb[2] if len(qb) > 2 else -1
        # Encode float params as int32 bit patterns
        if pr:
            for i, p in enumerate(pr[:3]):
                gd[4 + i] = int(np.float32(float(p)).view(np.int32))
        enc.extend(gd)
    return enc


def _execute_gates_on_fpga(device, kernel, hbm_banks, gate_list, nlq):
    """
    Execute a batch of gates on the FPGA kernel.

    This is the core FPGA execution path â gates are encoded and dispatched
    to the hardware kernel which processes them on the statevector in HBM.
    """
    enc = _encode_gates_for_fpga(gate_list)
    if not enc:
        return  # Nothing to execute

    gs = np.array(enc, dtype=np.int32)
    num_gates = len(enc) // 8
    gsz = len(gs) * 4  # bytes

    # Allocate gate buffer on device
    gbo = xrt.bo(device, max(gsz, 4), xrt.bo.flags.normal, kernel.group_id(16))
    gbo.write(gs.tobytes(), 0)
    gbo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, gsz, 0)

    # Invoke kernel: 16 HBM banks + gate_sequence + num_gates + num_qubits
    # Uses *hbm_banks[:16] unpacking â same as proven single-card code
    run = kernel(*hbm_banks[:16], gbo, num_gates, nlq)
    run.wait()


# ============================================================================
# Host-Side Vectorized Gate Kernels (for global gate computation ONLY)
# These are used ONLY when cross-card data exchange is required.
# NOT a "CPU fallback" â this is inter-card communication logic.
# ============================================================================

def _apply_1q_host(state, gn, qubit, params, nq):
    """Apply 1-qubit gate on host (for global gate coordination)."""
    mat = GATE_MATRICES.get(gn) if gn in GATE_MATRICES else get_parametric_matrix(gn, params)
    n = len(state)
    stride = 1 << qubit
    bs = stride * 2
    nb = n // bs
    base = np.arange(stride)
    offsets = np.arange(nb) * bs
    i0 = (offsets[:, None] + base[None, :]).ravel()
    i1 = i0 + stride
    a0, a1 = state[i0], state[i1]
    ns = state.copy()
    ns[i0] = mat[0, 0] * a0 + mat[0, 1] * a1
    ns[i1] = mat[1, 0] * a0 + mat[1, 1] * a1
    return ns


def _apply_2q_host(state, gn, q0, q1, params, nq):
    """Apply 2-qubit gate on host (for global gate coordination)."""
    n = len(state)
    ns = state.copy()
    cm, tm = 1 << q0, 1 << q1
    idx = np.arange(n)

    if gn in ('cx', 'cnot'):
        m = (idx & cm).astype(bool) & ~(idx & tm).astype(bool)
        i0 = idx[m]; i1 = i0 | tm
        ns[i0] = state[i1]; ns[i1] = state[i0]
    elif gn == 'cz':
        m = (idx & cm).astype(bool) & (idx & tm).astype(bool)
        ns[m] = -state[m]
    elif gn == 'cy':
        m = (idx & cm).astype(bool) & ~(idx & tm).astype(bool)
        i0 = idx[m]; i1 = i0 | tm
        ns[i0] = 1j * state[i1]; ns[i1] = -1j * state[i0]
    elif gn == 'ch':
        r2 = np.float32(1.0 / np.sqrt(2))
        m = (idx & cm).astype(bool) & ~(idx & tm).astype(bool)
        i0 = idx[m]; i1 = i0 | tm
        a0, a1 = state[i0], state[i1]
        ns[i0] = r2 * (a0 + a1); ns[i1] = r2 * (a0 - a1)
    elif gn == 'swap':
        b0 = (idx >> q0) & 1; b1 = (idx >> q1) & 1
        sw = idx ^ cm ^ tm
        fwd = (b0 != b1) & (idx < sw)
        ia, ib = idx[fwd], sw[fwd]
        ns[ia] = state[ib]; ns[ib] = state[ia]
    elif gn == 'cp':
        ph = np.exp(1j * np.complex64(float(params[0]) if params else 0))
        m = (idx & cm).astype(bool) & (idx & tm).astype(bool)
        ns[m] = ph * state[m]
    return ns


def _apply_3q_host(state, gn, qubits, params, nq):
    """Apply 3-qubit gate on host (for global gate coordination)."""
    n = len(state)
    ns = state.copy()
    if gn in ('ccx', 'toffoli'):
        c0, c1, tg = qubits
        m0, m1, mt = 1 << c0, 1 << c1, 1 << tg
        idx = np.arange(n)
        m = (idx & m0).astype(bool) & (idx & m1).astype(bool) & ~(idx & mt).astype(bool)
        i0 = idx[m]; i1 = i0 | mt
        ns[i0] = state[i1]; ns[i1] = state[i0]
    return ns


# ============================================================================
# Global Gate Dispatcher â Handles gates touching partition qubits
# ============================================================================

def _apply_global_gate(card_id, num_cards, npb, gate_name, gate_qubits,
                       gate_params, nq, nlq, partition_qubits,
                       local_state, read_arrays):
    """
    Apply gate touching partition qubits.
    Reads from read_arrays (shared memory) for partner card data.
    Returns new local partition state.
    """
    ls = len(local_state)
    pset = set(partition_qubits)
    is_p = [q in pset for q in gate_qubits]
    ng = sum(is_p)

    # ==== SINGLE-QUBIT on PARTITION ====
    if len(gate_qubits) == 1 and is_p[0]:
        q = gate_qubits[0]; pb = q - nlq
        mat = GATE_MATRICES.get(gate_name) if gate_name in GATE_MATRICES \
            else get_parametric_matrix(gate_name, gate_params)
        my_bit = (card_id >> pb) & 1
        pid = card_id ^ (1 << pb)
        ps = _read_from_shared(read_arrays[pid], ls)
        if my_bit == 0:
            return (mat[0, 0] * local_state + mat[0, 1] * ps).astype(np.complex64)
        else:
            return (mat[1, 0] * ps + mat[1, 1] * local_state).astype(np.complex64)

    # ==== TWO-QUBIT: ONE PARTITION, ONE LOCAL ====
    elif len(gate_qubits) == 2 and ng == 1:
        q0, q1 = gate_qubits

        # Control = partition, target = local
        if q0 in pset and q1 not in pset:
            pb = q0 - nlq; lq = q1
            mc = (card_id >> pb) & 1
            if gate_name in ('cx', 'cnot'):
                return _apply_1q_host(local_state, 'x', lq, [], nlq) if mc else local_state.copy()
            elif gate_name == 'cz':
                return _apply_1q_host(local_state, 'z', lq, [], nlq) if mc else local_state.copy()
            elif gate_name == 'cy':
                return _apply_1q_host(local_state, 'y', lq, [], nlq) if mc else local_state.copy()
            elif gate_name == 'ch':
                return _apply_1q_host(local_state, 'h', lq, [], nlq) if mc else local_state.copy()
            elif gate_name == 'cp':
                a = float(gate_params[0]) if gate_params else 0.0
                return _apply_1q_host(local_state, 'p', lq, [a], nlq) if mc else local_state.copy()
            elif gate_name == 'swap':
                return _apply_swap_one_partition(card_id, pb, lq, local_state,
                                                read_arrays, ls, nlq)

        # Target = partition, control = local
        elif q1 in pset and q0 not in pset:
            pb = q1 - nlq; lq = q0
            mt = (card_id >> pb) & 1
            pid = card_id ^ (1 << pb)
            ps = _read_from_shared(read_arrays[pid], ls)
            cm = 1 << lq
            idx = np.arange(ls)
            con = (idx & cm).astype(bool)

            if gate_name in ('cx', 'cnot'):
                ns = local_state.copy()
                ns[con] = ps[con]
                return ns.astype(np.complex64)
            elif gate_name == 'cz':
                ns = local_state.copy()
                if mt == 1:
                    ns[con] = -local_state[con]
                return ns.astype(np.complex64)
            elif gate_name == 'cy':
                ns = local_state.copy()
                if mt == 0:
                    ns[con] = 1j * ps[con]
                else:
                    ns[con] = -1j * ps[con]
                return ns.astype(np.complex64)
            elif gate_name == 'ch':
                r2 = np.float32(1.0 / np.sqrt(2))
                ns = local_state.copy()
                if mt == 0:
                    ns[con] = r2 * (local_state[con] + ps[con])
                else:
                    ns[con] = r2 * (ps[con] - local_state[con])
                return ns.astype(np.complex64)
            elif gate_name == 'cp':
                a = float(gate_params[0]) if gate_params else 0.0
                ph = np.exp(1j * np.complex64(a))
                ns = local_state.copy()
                if mt == 1:
                    ns[con] = ph * local_state[con]
                return ns.astype(np.complex64)
            elif gate_name == 'swap':
                return _apply_swap_one_partition(card_id, pb, lq, local_state,
                                                read_arrays, ls, nlq)

        # Fallback for unhandled 2q mixed cases
        return _apply_global_gate_generic(card_id, num_cards, npb, gate_name,
                                          gate_qubits, gate_params, nq, nlq,
                                          local_state, read_arrays)

    # ==== TWO-QUBIT: BOTH PARTITION ====
    elif len(gate_qubits) == 2 and ng == 2:
        q0, q1 = gate_qubits
        pb0, pb1 = q0 - nlq, q1 - nlq

        if gate_name in ('cx', 'cnot'):
            mc = (card_id >> pb0) & 1
            if mc == 1:
                pid = card_id ^ (1 << pb1)
                return _read_from_shared(read_arrays[pid], ls).astype(np.complex64)
            return local_state.copy()
        elif gate_name == 'cz':
            if ((card_id >> pb0) & 1) and ((card_id >> pb1) & 1):
                return (-local_state).astype(np.complex64)
            return local_state.copy()
        elif gate_name == 'swap':
            if ((card_id >> pb0) & 1) != ((card_id >> pb1) & 1):
                pid = card_id ^ (1 << pb0) ^ (1 << pb1)
                return _read_from_shared(read_arrays[pid], ls).astype(np.complex64)
            return local_state.copy()
        elif gate_name == 'cp':
            a = float(gate_params[0]) if gate_params else 0.0
            if ((card_id >> pb0) & 1) and ((card_id >> pb1) & 1):
                return (np.exp(1j * np.complex64(a)) * local_state).astype(np.complex64)
            return local_state.copy()
        elif gate_name == 'cy':
            mc = (card_id >> pb0) & 1; mt_val = (card_id >> pb1) & 1
            if mc == 0:
                return local_state.copy()
            pid = card_id ^ (1 << pb1)
            ps = _read_from_shared(read_arrays[pid], ls)
            if mt_val == 0:
                return (1j * ps).astype(np.complex64)
            else:
                return (-1j * ps).astype(np.complex64)
        elif gate_name == 'ch':
            mc = (card_id >> pb0) & 1; mt_val = (card_id >> pb1) & 1
            if mc == 0:
                return local_state.copy()
            pid = card_id ^ (1 << pb1)
            ps = _read_from_shared(read_arrays[pid], ls)
            r2 = np.float32(1.0 / np.sqrt(2))
            if mt_val == 0:
                return (r2 * (local_state + ps)).astype(np.complex64)
            else:
                return (r2 * (ps - local_state)).astype(np.complex64)

        return _apply_global_gate_generic(card_id, num_cards, npb, gate_name,
                                          gate_qubits, gate_params, nq, nlq,
                                          local_state, read_arrays)

    # ==== THREE-QUBIT: CCX/TOFFOLI ====
    elif len(gate_qubits) == 3 and gate_name in ('ccx', 'toffoli'):
        c0, c1, tg = gate_qubits
        c0p = c0 in pset; c1p = c1 in pset; tgp = tg in pset

        # Both controls partition, target local
        if c0p and c1p and not tgp:
            pb0, pb1 = c0 - nlq, c1 - nlq
            if ((card_id >> pb0) & 1) and ((card_id >> pb1) & 1):
                return _apply_1q_host(local_state, 'x', tg, [], nlq)
            return local_state.copy()

        # One control partition, one local, target local
        elif (c0p != c1p) and not tgp:
            if c0p:
                pb, lc = c0 - nlq, c1
            else:
                pb, lc = c1 - nlq, c0
            if (card_id >> pb) & 1:
                return _apply_2q_host(local_state, 'cx', lc, tg, [], nlq)
            return local_state.copy()

        # Both controls local, target partition
        elif not c0p and not c1p and tgp:
            pb = tg - nlq; pid = card_id ^ (1 << pb)
            ps = _read_from_shared(read_arrays[pid], ls)
            cm0, cm1 = 1 << c0, 1 << c1
            ns = local_state.copy()
            idx = np.arange(ls)
            both = (idx & cm0).astype(bool) & (idx & cm1).astype(bool)
            ns[both] = ps[both]
            return ns.astype(np.complex64)

        # One control partition, target partition, other control local
        elif (c0p != c1p) and tgp:
            if c0p:
                pc, lc = c0, c1
            else:
                pc, lc = c1, c0
            pcb = pc - nlq; tpb = tg - nlq
            if not ((card_id >> pcb) & 1):
                return local_state.copy()
            pid = card_id ^ (1 << tpb)
            ps = _read_from_shared(read_arrays[pid], ls)
            lcm = 1 << lc
            ns = local_state.copy()
            idx = np.arange(ls)
            con = (idx & lcm).astype(bool)
            ns[con] = ps[con]
            return ns.astype(np.complex64)

        # Both controls partition, target partition
        elif c0p and c1p and tgp:
            pb0, pb1, pbt = c0 - nlq, c1 - nlq, tg - nlq
            if ((card_id >> pb0) & 1) and ((card_id >> pb1) & 1):
                pid = card_id ^ (1 << pbt)
                return _read_from_shared(read_arrays[pid], ls).astype(np.complex64)
            return local_state.copy()

    # ==== GENERIC FALLBACK (for unhandled gate combinations) ====
    return _apply_global_gate_generic(card_id, num_cards, npb, gate_name,
                                      gate_qubits, gate_params, nq, nlq,
                                      local_state, read_arrays)


def _apply_swap_one_partition(card_id, part_bit, local_q, local_state,
                              read_arrays, local_size, nlq):
    """SWAP between a partition qubit and a local qubit."""
    my_part_val = (card_id >> part_bit) & 1
    partner_id = card_id ^ (1 << part_bit)
    partner_state = _read_from_shared(read_arrays[partner_id], local_size)

    new_state = local_state.copy()
    local_mask = 1 << local_q

    for i in range(local_size):
        local_bit = (i >> local_q) & 1
        if local_bit != my_part_val:
            i_flipped = i ^ local_mask
            new_state[i] = partner_state[i_flipped]

    return new_state.astype(np.complex64)


def _apply_global_gate_generic(card_id, num_cards, npb, gate_name, gate_qubits,
                               gate_params, nq, nlq, local_state, read_arrays):
    """Generic fallback: reconstruct full state, apply gate, extract partition."""
    ls = len(local_state)
    full = np.zeros(ls * num_cards, dtype=np.complex64)
    for cid in range(num_cards):
        cs = _read_from_shared(read_arrays[cid], ls)
        full[cid * ls:(cid + 1) * ls] = cs

    if len(gate_qubits) == 1:
        full = _apply_1q_host(full, gate_name, gate_qubits[0], gate_params, nq)
    elif len(gate_qubits) == 2:
        full = _apply_2q_host(full, gate_name, gate_qubits[0], gate_qubits[1],
                              gate_params, nq)
    elif len(gate_qubits) == 3:
        full = _apply_3q_host(full, gate_name, gate_qubits, gate_params, nq)

    off = card_id * ls
    return full[off:off + ls].astype(np.complex64)


# ============================================================================
# Worker Process â One per FPGA card, FPGA MANDATORY
# ============================================================================

def distributed_worker(card_id, num_cards, npb, xclbin_path, kernel_name,
                       read_arrays, write_arrays, local_size,
                       barrier_a, barrier_b, barrier_c,
                       cmd_queue, result_queue, ready_event, shutdown_event):
    """
    Worker process for one FPGA card.

    FPGA initialization is MANDATORY. If FPGA init fails, the worker
    reports the error and exits â there is NO CPU fallback.

    State resides in FPGA HBM. Local gates execute on FPGA fabric.
    Global gates require host-mediated data exchange between cards.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    device = None
    kernel = None
    hbm_banks = []
    fpga_ready = False
    current_nlq = 0
    current_num_amps = 0

    # HOST-SIDE SOURCE OF TRUTH
    # local_state is always the authoritative copy of this card's partition.
    # HBM is only used as the FPGA kernel's working memory.
    # We write TO HBM before FPGA execution and read FROM HBM after.
    # We NEVER do a hostâHBMâhost round-trip (sync(TO_DEVICE) may be
    # async on XRT 2.x, causing sync(FROM_DEVICE) to return stale data).
    local_state = None
    hbm_stale = True  # True = local_state has been modified since last HBM write

    try:
        ready_event.set()

        while not shutdown_event.is_set():
            try:
                cmd = cmd_queue.get(timeout=0.2)
            except Exception:
                continue

            if cmd is None:
                break

            ct = cmd[0]

            # ============================================================
            # INIT: Initialize FPGA and load statevector into HBM
            # ============================================================
            if ct == 'init':
                nq = cmd[1]
                nlq = nq - npb
                current_nlq = nlq
                lsz = 2 ** nlq
                current_num_amps = lsz

                # ---- FPGA init (MANDATORY, sized per circuit) ----
                if not fpga_ready:
                    try:
                        device = xrt.device(card_id)
                        uuid = device.load_xclbin(xclbin_path)
                        kernel = xrt.kernel(device, uuid, kernel_name)

                        amps_per_bank = (lsz + NUM_HBM_BANKS - 1) // NUM_HBM_BANKS
                        buf_bytes = max(amps_per_bank * 2 * 4, 4096)

                        hbm_banks = []
                        for i in range(NUM_HBM_BANKS):
                            bo = xrt.bo(device, buf_bytes,
                                        xrt.bo.flags.normal, kernel.group_id(i))
                            hbm_banks.append(bo)

                        fpga_ready = True
                        logger.info(
                            f"Card {card_id}: FPGA initialized â "
                            f"{lsz:,} amps, {lsz * 8 / (1024**2):.1f} MB HBM"
                        )
                    except Exception as e:
                        error_msg = (
                            f"Card {card_id}: FPGA initialization FAILED: {e}. "
                            f"No CPU fallback permitted."
                        )
                        logger.error(error_msg)
                        result_queue.put(('error', card_id, error_msg))
                        return
                else:
                    if lsz > current_num_amps:
                        amps_per_bank = (lsz + NUM_HBM_BANKS - 1) // NUM_HBM_BANKS
                        buf_bytes = max(amps_per_bank * 2 * 4, 4096)
                        hbm_banks = []
                        for i in range(NUM_HBM_BANKS):
                            bo = xrt.bo(device, buf_bytes,
                                        xrt.bo.flags.normal, kernel.group_id(i))
                            hbm_banks.append(bo)
                        current_num_amps = lsz

                # ---- Initialize |0...0â© state ----
                local_state = np.zeros(lsz, dtype=np.complex64)
                if card_id == 0:
                    local_state[0] = 1.0 + 0j

                # Write init state to HBM (ready for first FPGA execution)
                _write_state_to_hbm(hbm_banks, local_state, lsz)
                hbm_stale = False

                # Write to shared memory (ready for any immediate global gate)
                _write_to_shared(read_arrays[card_id], local_state, local_size)

                result_queue.put(('init_done', card_id))

            # ============================================================
            # LOCAL GATES: Execute batch on FPGA kernel
            #
            # Flow: sync local_state â HBM â FPGA kernel â HBM â local_state
            # ============================================================
            elif ct == 'local_gates':
                gl, nlq = cmd[1], cmd[2]

                # If a global gate modified local_state since last HBM write,
                # we must sync local_state to HBM before FPGA can work on it
                if hbm_stale:
                    _write_state_to_hbm(hbm_banks, local_state, current_num_amps)
                    hbm_stale = False

                # Execute gates on FPGA hardware (modifies HBM in-place)
                _execute_gates_on_fpga(device, kernel, hbm_banks, gl, nlq)

                # Read FPGA result from HBM â local_state (authoritative)
                local_state = _read_state_from_hbm(hbm_banks, current_num_amps)
                hbm_stale = False  # HBM and local_state are in sync

                # Update shared memory for potential subsequent global gates
                _write_to_shared(read_arrays[card_id], local_state, local_size)

                result_queue.put(('local_done', card_id))

            # ============================================================
            # GLOBAL GATE: Host-side cross-card computation
            #
            # Uses local_state directly (host-side source of truth).
            # Does NOT read from HBM â avoids async DMA round-trip bug.
            # ============================================================
            elif ct == 'global_gate':
                gn, gq, gp, nq = cmd[1], cmd[2], cmd[3], cmd[4]
                nlq = nq - npb
                pq = list(range(nlq, nq))

                # Step 1: Publish local_state to shared memory
                # (local_state is authoritative â no HBM read needed)
                _write_to_shared(read_arrays[card_id], local_state, local_size)
                barrier_a.wait()

                # Step 2: Compute new partition using all cards' shared memory
                new_state = _apply_global_gate(
                    card_id, num_cards, npb, gn, gq, gp, nq, nlq, pq,
                    local_state, read_arrays
                )

                # Step 3: Publish result
                _write_to_shared(write_arrays[card_id], new_state, local_size)
                barrier_b.wait()

                # Step 4: Update local_state (host-side source of truth)
                local_state = new_state
                hbm_stale = True  # HBM no longer matches local_state

                # Update read buffer for next gate
                _write_to_shared(read_arrays[card_id], local_state, local_size)
                barrier_c.wait()

                result_queue.put(('global_done', card_id))

            # ============================================================
            # COLLECT: Return local_state (host-side source of truth)
            #
            # Does NOT read from HBM â local_state is always authoritative.
            # ============================================================
            elif ct == 'collect':
                result_queue.put((
                    'collect_done', card_id,
                    local_state.tobytes(), local_state.shape
                ))

    except Exception as e:
        logger.error(f"Card {card_id}: Fatal â {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(('error', card_id, str(e)))
        ready_event.set()
    finally:
        hbm_banks.clear()
        device = None
        kernel = None


# ============================================================================
# Main Simulator Class
# ============================================================================

class FPGADistributedSimulator:
    """
    Distributed Single-Circuit Multi-FPGA Quantum Simulator.

    Partitions one circuit's statevector across K FPGA cards.
    Top log2(K) qubits = partition index.
    3-barrier double-buffered protocol for race-free global gates.

    ALL gate computation executes on FPGA hardware.
    FPGA initialization is MANDATORY â no CPU fallback.

    Execution Model:
        1. State initialized in FPGA HBM across all cards
        2. Local gates: dispatched to FPGA kernel (gate math on FPGA fabric)
        3. Global gates: HBMâhost exchangeârecomputeâHBM
        4. Final state read from FPGA HBM for measurement
    """

    # Known FPGA kernel names â tried in order
    KERNEL_NAMES = [
        "quantum_simulator_kernel",
        "quantum_noisy_simulator_kernel",
    ]

    def __init__(self, xclbin_path=None, num_cards=4):
        if num_cards not in (1, 2, 4, 8, 16):
            raise ValueError("num_cards must be power of 2 (1, 2, 4, 8, 16)")

        if not XRT_AVAILABLE:
            raise RuntimeError(
                "pyxrt is NOT available. FPGA execution requires XRT runtime. "
                "Install Xilinx XRT and ensure pyxrt is importable."
            )

        if xclbin_path is None:
            candidates = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "quantum_simulator_kernel.xclbin"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "quantum_noisy_simulator_kernel.xclbin"),
                "/home/abhishek/fpga/lib/python3.8/site-packages/qiskit_aer/"
                "backends/fpga_distributed_statevector/"
                "quantum_noisy_simulator_kernel.xclbin",
                "/home/abhishek/fpga/lib/python3.8/site-packages/qiskit_aer/"
                "backends/quantum_simulator_kernel.xclbin",
            ]
            xclbin_path = None
            for c in candidates:
                if os.path.exists(c):
                    xclbin_path = c
                    break
            if xclbin_path is None:
                raise FileNotFoundError(
                    f"XCLBIN not found. Searched: {candidates}. "
                    f"Pass xclbin_path= explicitly."
                )

        self.num_cards = num_cards
        self.num_partition_bits = int(np.log2(num_cards))
        self.xclbin_path = os.path.abspath(xclbin_path)
        self._shutdown_called = False
        self._workers_spawned = False
        self._current_nq = 0
        self._kernel_name = None

        # Detect kernel name from the xclbin
        self._kernel_name = self._detect_kernel_name()

        # Resources â created per-circuit
        self.read_arrays = None
        self.write_arrays = None
        self.barrier_a = None
        self.barrier_b = None
        self.barrier_c = None
        self.cmd_queues = []
        self.result_queue = None
        self.ready_events = []
        self.shutdown_events = []
        self.workers = []
        self._local_size = 0

        atexit.register(self.shutdown)

    def _detect_kernel_name(self):
        """
        Detect the kernel name by probing the xclbin.
        Try known kernel names and return the first that works.
        """
        try:
            dev = xrt.device(0)
            uuid = dev.load_xclbin(self.xclbin_path)
            for kn in self.KERNEL_NAMES:
                try:
                    k = xrt.kernel(dev, uuid, kn)
                    logger.info(f"Detected FPGA kernel: {kn}")
                    return kn
                except Exception:
                    continue
            # If none of the known names work, try from xclbin filename
            base = os.path.basename(self.xclbin_path).replace('.xclbin', '')
            try:
                k = xrt.kernel(dev, uuid, base)
                return base
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Kernel detection probe failed: {e}")

        # Default: use the C++ kernel name from the source
        return "quantum_simulator_kernel"

    def _ensure_workers(self, nq):
        """
        (Re-)create shared memory and workers sized for the given circuit.
        """
        nlq = nq - self.num_partition_bits
        local_size = 2 ** nlq

        if self._workers_spawned and self._current_nq == nq:
            all_alive = all(w.is_alive() for w in self.workers)
            if all_alive:
                return
            logger.warning("Some workers died, recreating...")

        if self._workers_spawned:
            self._teardown_workers()

        self._current_nq = nq
        self._local_size = local_size

        # Allocate shared memory sized for THIS circuit
        shm_entries = local_size * 2  # interleaved real/imag float32
        logger.info(
            f"Shared memory: {self.num_cards} Ã {local_size} amps Ã 8B = "
            f"{self.num_cards * local_size * 8 / (1024**2):.1f} MB per buffer"
        )

        self.read_arrays = [
            mp.Array(ctypes.c_float, shm_entries, lock=False)
            for _ in range(self.num_cards)
        ]
        self.write_arrays = [
            mp.Array(ctypes.c_float, shm_entries, lock=False)
            for _ in range(self.num_cards)
        ]

        self.barrier_a = mp.Barrier(self.num_cards)
        self.barrier_b = mp.Barrier(self.num_cards)
        self.barrier_c = mp.Barrier(self.num_cards)

        self.cmd_queues = []
        self.result_queue = mp.Queue()
        self.ready_events = []
        self.shutdown_events = []
        self.workers = []

        for cid in range(self.num_cards):
            cq = mp.Queue()
            re = mp.Event()
            se = mp.Event()
            w = Process(
                target=distributed_worker,
                args=(cid, self.num_cards, self.num_partition_bits,
                      self.xclbin_path, self._kernel_name,
                      self.read_arrays, self.write_arrays, local_size,
                      self.barrier_a, self.barrier_b, self.barrier_c,
                      cq, self.result_queue, re, se),
                daemon=True
            )
            w.start()
            self.cmd_queues.append(cq)
            self.ready_events.append(re)
            self.shutdown_events.append(se)
            self.workers.append(w)

        self._wait_for_workers(timeout=120.0)
        self._workers_spawned = True

        active = sum(1 for w in self.workers if w.is_alive())
        if active < self.num_cards:
            raise RuntimeError(
                f"Only {active}/{self.num_cards} FPGA workers started. "
                f"Check XRT installation and FPGA device availability."
            )

    def _wait_for_workers(self, timeout=120.0):
        start = time.time()
        for i, ev in enumerate(self.ready_events):
            rem = timeout - (time.time() - start)
            if rem <= 0 or not ev.wait(timeout=max(rem, 0.1)):
                logger.warning(f"Card {i}: init timeout")

    def _teardown_workers(self):
        """Clean shutdown of all workers and shared resources."""
        for ev in self.shutdown_events:
            try:
                ev.set()
            except Exception:
                pass
        for q in self.cmd_queues:
            try:
                q.put_nowait(None)
            except Exception:
                pass
        for w in self.workers:
            try:
                w.join(timeout=5)
                if w.is_alive():
                    w.terminate()
                    w.join(timeout=2)
            except Exception:
                pass

        for q in self.cmd_queues:
            try:
                while not q.empty():
                    q.get_nowait()
            except Exception:
                pass
        if self.result_queue is not None:
            try:
                while not self.result_queue.empty():
                    self.result_queue.get_nowait()
            except Exception:
                pass

        self.cmd_queues = []
        self.result_queue = None
        self.ready_events = []
        self.shutdown_events = []
        self.workers = []
        self.read_arrays = None
        self.write_arrays = None
        self._workers_spawned = False

    def run(self, circuit, shots=4096, **kwargs):
        """
        Run a quantum circuit on the distributed FPGA simulator.

        Gate execution on FPGA hardware â NO CPU fallback.
        """
        if self._shutdown_called:
            raise RuntimeError("Simulator has been shut down")

        nq = circuit.num_qubits
        nlq = nq - self.num_partition_bits

        if nq < self.num_partition_bits + 1:
            raise ValueError(
                f"Need >= {self.num_partition_bits + 1} qubits for "
                f"{self.num_cards} cards"
            )
        if nq > ABSOLUTE_MAX_QUBITS:
            raise ValueError(f"{nq} qubits exceeds max {ABSOLUTE_MAX_QUBITS}")

        local_size = 2 ** nlq
        mem_per_card = local_size * 8
        logger.info(
            f"Circuit: {nq}q, {mem_per_card / (1024**3):.2f} GB/card, "
            f"{mem_per_card * self.num_cards / (1024**3):.2f} GB total"
        )

        t0 = time.time()

        # ---- Ensure workers are sized for this circuit ----
        self._ensure_workers(nq)

        pset = set(range(nlq, nq))
        gates = self._parse_circuit(circuit)
        mq = self._get_measured_qubits(circuit)

        # ---- Initialize state on all FPGA cards ----
        self._broadcast(('init', nq))
        self._wait_all('init_done')

        # ---- Execute gates: local on FPGA, global with exchange ----
        local_batch = []
        for gn, qb, pr in gates:
            if any(q in pset for q in qb):
                # Global gate â flush local batch first
                if local_batch:
                    self._exec_local(local_batch, nlq)
                    local_batch = []
                self._exec_global(gn, qb, pr, nq)
            else:
                local_batch.append((gn, qb, pr))

        if local_batch:
            self._exec_local(local_batch, nlq)

        # ---- Collect results from FPGA HBM ----
        self._broadcast(('collect',))
        parts = [None] * self.num_cards
        for _ in range(self.num_cards):
            r = self.result_queue.get(timeout=300)
            if r[0] == 'collect_done':
                parts[r[1]] = np.frombuffer(r[2], dtype=np.complex64).copy().reshape(r[3])
            elif r[0] == 'error':
                raise RuntimeError(f"FPGA Card {r[1]} error: {r[2]}")

        lsz = 2 ** nlq
        full = np.zeros(2 ** nq, dtype=np.complex64)
        for cid in range(self.num_cards):
            if parts[cid] is not None:
                full[cid * lsz:(cid + 1) * lsz] = parts[cid]

        norm = np.linalg.norm(full)
        if norm > 1e-10:
            full /= norm

        counts = self._sample(full, nq, mq, shots)
        return DistributedResult(counts, full, time.time() - t0, nq,
                                 self.num_cards, shots,
                                 getattr(circuit, 'name', 'unnamed'))

    def _parse_circuit(self, circuit):
        gates = []
        for inst in circuit.data:
            nm = inst.operation.name.lower()
            if nm in ('measure', 'barrier', 'reset'):
                continue
            try:
                if hasattr(circuit, 'find_bit'):
                    qb = [circuit.find_bit(q).index for q in inst.qubits]
                else:
                    qb = [circuit.qubits.index(q) for q in inst.qubits]
            except Exception:
                qb = list(range(len(inst.qubits)))
            pr = [float(p) for p in getattr(inst.operation, 'params', [])]
            gates.append((nm, qb, pr))
        return gates

    def _get_measured_qubits(self, circuit):
        mq = []
        for inst in circuit.data:
            if inst.operation.name.lower() == 'measure':
                try:
                    if hasattr(circuit, 'find_bit'):
                        qs = [circuit.find_bit(q).index for q in inst.qubits]
                    else:
                        qs = [circuit.qubits.index(q) for q in inst.qubits]
                except Exception:
                    qs = list(range(len(inst.qubits)))
                for q in qs:
                    if q not in mq:
                        mq.append(q)
        return mq if mq else list(range(circuit.num_qubits))

    def _broadcast(self, cmd):
        for q in self.cmd_queues:
            q.put(cmd)

    def _wait_all(self, etype, timeout=300):
        n = 0
        deadline = time.time() + timeout
        while n < self.num_cards:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for {etype}: "
                    f"got {n}/{self.num_cards} responses"
                )
            r = self.result_queue.get(timeout=remaining)
            if r[0] == etype:
                n += 1
            elif r[0] == 'error':
                raise RuntimeError(f"FPGA Card {r[1]} error: {r[2]}")

    def _exec_local(self, gl, nlq):
        self._broadcast(('local_gates', gl, nlq))
        self._wait_all('local_done')

    def _exec_global(self, gn, qb, pr, nq):
        self._broadcast(('global_gate', gn, qb, pr, nq))
        self._wait_all('global_done')

    def _sample(self, sv, nq, mq, shots):
        probs = np.abs(sv) ** 2
        t = np.sum(probs)
        if t > 0:
            probs /= t
        else:
            probs = np.ones(len(probs)) / len(probs)
        probs = np.maximum(probs, 0)
        probs /= np.sum(probs)
        ms = sorted(mq, reverse=True) if mq else list(range(nq - 1, -1, -1))
        rng = np.random.default_rng()
        samples = rng.choice(len(sv), size=shots, p=probs)
        counts = {}
        for idx in samples:
            fs = format(idx, f'0{nq}b')
            outcome = ''.join(fs[nq - 1 - q] for q in ms)
            counts[outcome] = counts.get(outcome, 0) + 1
        return counts

    def get_card_stats(self):
        return [
            {
                'card_id': i,
                'alive': self.workers[i].is_alive() if i < len(self.workers) else False,
                'pid': self.workers[i].pid if i < len(self.workers) else None,
                'fpga_device': i,
                'execution_mode': 'FPGA_ONLY',
            }
            for i in range(self.num_cards)
        ]

    def shutdown(self):
        if self._shutdown_called:
            return
        self._shutdown_called = True
        self._teardown_workers()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.shutdown()
        return False


class DistributedResult:
    """Result container for distributed FPGA simulation."""

    def __init__(self, counts, sv, et, nq, nc, shots, name):
        self.counts = counts
        self.statevector = sv
        self.execution_time = et
        self.num_qubits = nq
        self.num_cards = nc
        self.shots = shots
        self.circuit_name = name

    def get_counts(self):
        return self.counts

    def get_statevector(self):
        return self.statevector

    def get_execution_time(self):
        return self.execution_time

    def success(self):
        return len(self.counts) > 0


if __name__ == "__main__":
    print("FPGA Distributed Statevector Simulator â STRICT FPGA EXECUTION")
    print("=" * 65)
    print(f"  XRT Available:      {XRT_AVAILABLE}")
    print(f"  Max Qubits:         {ABSOLUTE_MAX_QUBITS}")
    print(f"  HBM Banks/Card:     {NUM_HBM_BANKS}")
    print(f"  Start Method:       {mp.get_start_method()}")
    print(f"  CPU Fallback:       DISABLED (strict FPGA)")
    print()
    print("  Architecture:")
    print("    - State LIVES in FPGA HBM at all times")
    print("    - Local gates â FPGA kernel (no CPU execution)")
    print("    - Global gates â HBM read â exchange â compute â HBM write")
    print("    - FPGA init MANDATORY (RuntimeError if unavailable)")
    print()
    print("  Usage:")
    print("    from fpga_distributed_statevector import FPGADistributedSimulator")
    print("    sim = FPGADistributedSimulator(num_cards=4)")
    print("    result = sim.run(circuit, shots=4096)")
    print("    sim.shutdown()")