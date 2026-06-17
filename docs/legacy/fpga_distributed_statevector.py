#!/usr/bin/env python3
"""
FPGA Distributed Statevector Simulator - Single Circuit on Multiple FPGAs
=========================================================================

Distributes ONE quantum circuit's statevector across multiple FPGA cards.

Architecture:
    For N qubits on K cards (K = 2^k):
    - Top k qubits = partition index â which card owns each amplitude
    - Each card holds 2^(N-k) amplitudes
    - Card j owns amplitudes where top k bits of state index == j

Communication:
    Uses DOUBLE-BUFFERED shared memory with 3-barrier synchronization:
    1. Write pre-gate state â read_buffer, Barrier A
    2. Compute (read from read_buffer only), write â write_buffer, Barrier B
    3. Copy write â read, Barrier C (ready for next gate)

Gate Execution:
    - Local gates (acting only on local qubits): executed via vectorized
      NumPy on each card's partition (2^(N-k) amplitudes).
    - Global gates (touching partition qubits): handled by shared-memory
      exchange + vectorized NumPy across cards with 3-barrier synchronization.
    - FPGA kernel reserved for future single-card full-statevector mode.

FIXES over previous version:
    - Dynamic shared memory sizing per circuit (was MAX_QUBITS=40 â ~16 TB SIGBUS)
    - 'spawn' multiprocessing start method (avoids fork+XRT conflicts)
    - Deferred FPGA initialization (sized per circuit, not at constructor)
    - Graceful XRT fallback with proper error handling
    - Safe shutdown with resource cleanup

Author: Nasir Ali, C-DAC Noida
Date: February 2026
"""

import sys, time, logging, os, signal, atexit, ctypes, struct
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Force 'spawn' start method BEFORE any other multiprocessing import.
# This prevents fork() from inheriting parent XRT mmaps/fds into children.
# ---------------------------------------------------------------------------
import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=False)
except RuntimeError:
    pass  # Already set â fine

from multiprocessing import Process, Queue, Array, Barrier, Event

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

# ---------------------------------------------------------------------------
# Safety cap: absolute max qubits we will ever attempt.
# Shared memory is allocated PER-CIRCUIT, not at this cap.
# This is only used for validation.
# ---------------------------------------------------------------------------
ABSOLUTE_MAX_QUBITS = 40

GATE_TYPES = {
    'h': 0, 'x': 1, 'y': 2, 'z': 3, 's': 4, 't': 5, 'sdg': 6, 'tdg': 7,
    'rx': 8, 'ry': 9, 'rz': 10, 'p': 11, 'phase': 11, 'u1': 11,
    'cx': 12, 'cnot': 12, 'cy': 13, 'cz': 14, 'ch': 15,
    'swap': 16, 'ccx': 17, 'toffoli': 17,
    'measure': 255, 'barrier': 254, 'reset': 253,
}

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
# Shared Memory I/O
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
# Vectorized Gate Kernels
# ============================================================================

def _execute_gates_vectorized(state, gate_list, nlq):
    """Execute a batch of gates on a local partition using vectorized NumPy."""
    for gn, qb, pr in gate_list:
        if len(qb) == 1:
            state = _apply_1q(state, gn, qb[0], pr, nlq)
        elif len(qb) == 2:
            state = _apply_2q(state, gn, qb[0], qb[1], pr, nlq)
        elif len(qb) == 3:
            state = _apply_3q(state, gn, qb, pr, nlq)
    return state


def _apply_1q(state, gn, qubit, params, nq):
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


def _apply_2q(state, gn, q0, q1, params, nq):
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


def _apply_3q(state, gn, qubits, params, nq):
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
# SWAP Helper (one partition qubit, one local qubit)
# ============================================================================

def _apply_swap_one_partition(card_id, part_bit, local_q, local_state,
                              read_arrays, local_size, nlq):
    """
    SWAP between a partition qubit and a local qubit.

    Let p = my_part_val = (card_id >> part_bit) & 1
    Let l = (i >> local_q) & 1 for local index i.

    After SWAP(part, local), new amp at (p, l) = old amp at (l, p).

    Case p == l: same card, same index â identity (no change).
    Case p != l: source card = partner (flipped partition bit),
                 source local index = i ^ (1 << local_q) (flipped local bit).
    """
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


# ============================================================================
# Global Gate Dispatcher
# ============================================================================

def _apply_global_gate(card_id, num_cards, npb, gate_name, gate_qubits,
                       gate_params, nq, nlq, partition_qubits,
                       local_state, read_arrays):
    """Apply gate touching partition qubits. Reads ONLY from read_arrays."""
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

        # --- Control = partition, target = local ---
        if q0 in pset and q1 not in pset:
            pb = q0 - nlq; lq = q1
            mc = (card_id >> pb) & 1

            if gate_name in ('cx', 'cnot'):
                return _apply_1q(local_state, 'x', lq, [], nlq) if mc else local_state.copy()
            elif gate_name == 'cz':
                return _apply_1q(local_state, 'z', lq, [], nlq) if mc else local_state.copy()
            elif gate_name == 'cy':
                return _apply_1q(local_state, 'y', lq, [], nlq) if mc else local_state.copy()
            elif gate_name == 'ch':
                return _apply_1q(local_state, 'h', lq, [], nlq) if mc else local_state.copy()
            elif gate_name == 'cp':
                a = float(gate_params[0]) if gate_params else 0.0
                return _apply_1q(local_state, 'p', lq, [a], nlq) if mc else local_state.copy()
            elif gate_name == 'swap':
                return _apply_swap_one_partition(card_id, pb, lq, local_state,
                                                read_arrays, ls, nlq)

        # --- Target = partition, control = local ---
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
                if mt == 1: ns[con] = -local_state[con]
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
                if mt == 1: ns[con] = ph * local_state[con]
                return ns.astype(np.complex64)
            elif gate_name == 'swap':
                return _apply_swap_one_partition(card_id, pb, lq, local_state,
                                                read_arrays, ls, nlq)

        # Fallback
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
            if mc == 0: return local_state.copy()
            pid = card_id ^ (1 << pb1)
            ps = _read_from_shared(read_arrays[pid], ls)
            if mt_val == 0:
                return (1j * ps).astype(np.complex64)
            else:
                return (-1j * ps).astype(np.complex64)
        elif gate_name == 'ch':
            mc = (card_id >> pb0) & 1; mt_val = (card_id >> pb1) & 1
            if mc == 0: return local_state.copy()
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
                return _apply_1q(local_state, 'x', tg, [], nlq)
            return local_state.copy()

        # One control partition, one local, target local
        elif (c0p != c1p) and not tgp:
            if c0p:
                pb, lc = c0 - nlq, c1
            else:
                pb, lc = c1 - nlq, c0
            if (card_id >> pb) & 1:
                return _apply_2q(local_state, 'cx', lc, tg, [], nlq)
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

    # ==== GENERIC FALLBACK ====
    return _apply_global_gate_generic(card_id, num_cards, npb, gate_name,
                                      gate_qubits, gate_params, nq, nlq,
                                      local_state, read_arrays)


def _apply_global_gate_generic(card_id, num_cards, npb, gate_name, gate_qubits,
                               gate_params, nq, nlq, local_state, read_arrays):
    """Reconstruct full state, apply gate, extract partition."""
    ls = len(local_state)
    full = np.zeros(ls * num_cards, dtype=np.complex64)
    for cid in range(num_cards):
        cs = _read_from_shared(read_arrays[cid], ls)
        full[cid * ls:(cid + 1) * ls] = cs

    if len(gate_qubits) == 1:
        full = _apply_1q(full, gate_name, gate_qubits[0], gate_params, nq)
    elif len(gate_qubits) == 2:
        full = _apply_2q(full, gate_name, gate_qubits[0], gate_qubits[1], gate_params, nq)
    elif len(gate_qubits) == 3:
        full = _apply_3q(full, gate_name, gate_qubits, gate_params, nq)

    off = card_id * ls
    return full[off:off + ls].astype(np.complex64)


# ============================================================================
# FPGA Helpers (reserved for future single-card full-statevector mode)
# ============================================================================

def _write_state_to_hbm(hbm_banks, state, nlq):
    ss = len(state)
    pb = (ss + NUM_HBM_BANKS - 1) // NUM_HBM_BANKS
    for bid, bank in enumerate(hbm_banks):
        buf = np.zeros(pb * 2, dtype=np.float32)
        indices = np.arange(bid, ss, NUM_HBM_BANKS)
        for li, gi in enumerate(indices):
            buf[li * 2] = state[gi].real
            buf[li * 2 + 1] = state[gi].imag
        bank.write(buf.tobytes(), 0)
        bank.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, len(buf.tobytes()), 0)


def _execute_gates_fpga(device, kernel, hbm_banks, state, gate_list, nlq):
    """Execute gates on FPGA kernel (single-card full-statevector mode)."""
    enc = []
    for gn, qb, pr in gate_list:
        gc = GATE_TYPES.get(gn, 255)
        if gc == 255: continue
        gd = [0] * 8
        gd[0] = gc
        gd[1] = qb[0] if len(qb) > 0 else 0
        gd[2] = qb[1] if len(qb) > 1 else -1
        gd[3] = qb[2] if len(qb) > 2 else -1
        if pr:
            for i, p in enumerate(pr[:3]):
                gd[4 + i] = int(np.float32(float(p)).view(np.int32))
        enc.extend(gd)
    if not enc: return state
    gs = np.array(enc, dtype=np.int32)
    ng = len(enc) // 8
    gsz = len(gs) * 4
    gbo = xrt.bo(device, gsz, xrt.bo.flags.normal, kernel.group_id(16))
    gbo.write(gs.tobytes(), 0)
    gbo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, gsz, 0)
    run = kernel(*hbm_banks[:16], gbo, ng, nlq)
    run.wait()
    ss = 2 ** nlq
    pb = (ss + NUM_HBM_BANKS - 1) // NUM_HBM_BANKS
    ns = np.zeros(ss, dtype=np.complex64)
    for bid, bank in enumerate(hbm_banks):
        bank.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, pb * 8, 0)
        buf = bank.map()
        raw = np.frombuffer(buf, dtype=np.float32, count=pb * 2)
        cplx = raw[0::2] + 1j * raw[1::2]
        indices = np.arange(bid, ss, NUM_HBM_BANKS)
        v = min(len(cplx), len(indices))
        ns[indices[:v]] = cplx[:v]
    return ns


# ============================================================================
# Worker Process
# ============================================================================

def distributed_worker(card_id, num_cards, npb, xclbin_path,
                       read_arrays, write_arrays, local_size,
                       barrier_a, barrier_b, barrier_c,
                       cmd_queue, result_queue, ready_event, shutdown_event):
    """
    Worker process for one FPGA card.
    
    IMPORTANT: FPGA initialization is DEFERRED â it happens on the first
    'init' command, not at process start. This ensures:
      1. No fork()+XRT conflict (we use 'spawn', but extra safety)
      2. HBM buffers can be sized for actual circuit, not MAX_QUBITS
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    local_state = None
    device = None
    kernel = None
    hbm_banks = []
    fpga_ready = False
    current_nlq = 0

    try:
        # Signal parent that we're alive and ready for commands.
        # FPGA init is deferred to 'init' command.
        ready_event.set()

        while not shutdown_event.is_set():
            try:
                cmd = cmd_queue.get(timeout=0.2)
            except Exception:
                continue

            if cmd is None:
                break

            ct = cmd[0]

            if ct == 'init':
                nq = cmd[1]
                nlq = nq - npb
                current_nlq = nlq
                lsz = 2 ** nlq

                # ---- Attempt FPGA init (deferred, per-circuit sizing) ----
                if XRT_AVAILABLE and not fpga_ready:
                    try:
                        device = xrt.device(card_id)
                        uuid = device.load_xclbin(xclbin_path)
                        kernel = xrt.kernel(device, uuid, "quantum_noisy_simulator_kernel")
                        # Size HBM buffers for THIS circuit's local partition,
                        # not MAX_QUBITS. Add small headroom (2x) for safety.
                        hbm_buf_size = max(lsz, 1024)
                        pb = (hbm_buf_size + NUM_HBM_BANKS - 1) // NUM_HBM_BANKS
                        for i in range(NUM_HBM_BANKS):
                            hbm_banks.append(
                                xrt.bo(device, pb * 8, xrt.bo.flags.normal, kernel.group_id(i))
                            )
                        fpga_ready = True
                        logger.info(f"Card {card_id}: FPGA initialized for {nq}-qubit circuit "
                                    f"({lsz} amps/card)")
                    except Exception as e:
                        logger.warning(f"Card {card_id}: FPGA init note ({e}), "
                                       f"using vectorized execution")
                        device = None; kernel = None; hbm_banks = []

                # ---- Initialize local state ----
                local_state = np.zeros(lsz, dtype=np.complex64)
                if card_id == 0:
                    local_state[0] = 1.0 + 0j

                _write_to_shared(read_arrays[card_id], local_state, local_size)
                result_queue.put(('init_done', card_id))

            elif ct == 'local_gates':
                gl, nlq = cmd[1], cmd[2]
                local_state = _execute_gates_vectorized(local_state, gl, nlq)
                _write_to_shared(read_arrays[card_id], local_state, local_size)
                result_queue.put(('local_done', card_id))

            elif ct == 'global_gate':
                gn, gq, gp, nq = cmd[1], cmd[2], cmd[3], cmd[4]
                nlq = nq - npb
                pq = list(range(nlq, nq))

                # 3-barrier double-buffered protocol
                _write_to_shared(read_arrays[card_id], local_state, local_size)
                barrier_a.wait()

                new_state = _apply_global_gate(
                    card_id, num_cards, npb, gn, gq, gp, nq, nlq, pq,
                    local_state, read_arrays)

                _write_to_shared(write_arrays[card_id], new_state, local_size)
                barrier_b.wait()

                local_state = new_state
                _write_to_shared(read_arrays[card_id], local_state, local_size)
                barrier_c.wait()

                result_queue.put(('global_done', card_id))

            elif ct == 'collect':
                result_queue.put(('collect_done', card_id,
                                  local_state.tobytes(), local_state.shape))

    except Exception as e:
        logger.error(f"Card {card_id}: Fatal - {e}")
        import traceback
        traceback.print_exc()
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

    Partitions one circuit's statevector across K cards.
    Top log2(K) qubits = partition index.
    3-barrier double-buffered protocol for race-free global gates.

    FIXED: Shared memory is allocated per-circuit (in run()), not at
    constructor time with MAX_QUBITS. Workers are spawned once and
    re-used across circuits.
    """

    def __init__(self, xclbin_path=None, num_cards=4):
        if num_cards not in (1, 2, 4, 8, 16):
            raise ValueError("num_cards must be power of 2 (1, 2, 4, 8, 16)")

        if xclbin_path is None:
            # Try several common locations
            candidates = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "quantum_noisy_simulator_kernel.xclbin"),
                "/home/abhishek/fpga/lib/python3.8/site-packages/qiskit_aer/backends/"
                "fpga_distributed_statevector/quantum_simulator_kernel.xclbin",
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
        self._current_nq = 0  # Track current circuit size

        # Shared memory, barriers, and workers are created per-circuit in run()
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

    def _ensure_workers(self, nq):
        """
        (Re-)create shared memory and workers sized for the given circuit.

        If workers are already running for the same circuit size, reuse them.
        If the circuit size changed, shut down old workers and create new ones.
        """
        nlq = nq - self.num_partition_bits
        local_size = 2 ** nlq

        if self._workers_spawned and self._current_nq == nq:
            # Workers already running for this circuit size â check health
            all_alive = all(w.is_alive() for w in self.workers)
            if all_alive:
                return
            # Some workers died â tear down and recreate
            logger.warning("Some workers died, recreating...")

        # Tear down existing workers if any
        if self._workers_spawned:
            self._teardown_workers()

        self._current_nq = nq
        self._local_size = local_size

        # ---- Allocate shared memory sized for THIS circuit ----
        # Each array: local_size * 2 (interleaved real/imag float32)
        shm_entries = local_size * 2
        logger.info(f"Allocating shared memory: {self.num_cards} cards Ã "
                    f"{local_size} amps Ã 8 bytes = "
                    f"{self.num_cards * local_size * 8 / (1024**2):.1f} MB per buffer set")

        self.read_arrays = [
            mp.Array(ctypes.c_float, shm_entries, lock=False)
            for _ in range(self.num_cards)
        ]
        self.write_arrays = [
            mp.Array(ctypes.c_float, shm_entries, lock=False)
            for _ in range(self.num_cards)
        ]

        # ---- Create barriers ----
        self.barrier_a = mp.Barrier(self.num_cards)
        self.barrier_b = mp.Barrier(self.num_cards)
        self.barrier_c = mp.Barrier(self.num_cards)

        # ---- Spawn workers ----
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
                      self.xclbin_path,
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

        # Wait for all workers to signal ready
        self._wait_for_workers(timeout=60.0)
        self._workers_spawned = True

        active = sum(1 for w in self.workers if w.is_alive())
        if active < self.num_cards:
            raise RuntimeError(
                f"Only {active}/{self.num_cards} workers started. "
                f"Check XRT installation and FPGA device availability."
            )
        logger.info(f"FPGADistributedSimulator ready: {active}/{self.num_cards} cards "
                    f"for {nq}-qubit circuits")

    def _wait_for_workers(self, timeout=60.0):
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
                w.join(timeout=3)
                if w.is_alive():
                    w.terminate()
                    w.join(timeout=1)
            except Exception:
                pass

        # Drain queues
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
        """Run a quantum circuit on the distributed FPGA simulator."""
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
            raise ValueError(f"{nq} qubits exceeds absolute max {ABSOLUTE_MAX_QUBITS}")

        # Validate memory requirements
        local_size = 2 ** nlq
        mem_per_card = local_size * 8  # complex64 = 8 bytes
        total_mem = mem_per_card * self.num_cards
        logger.info(f"Circuit requires {total_mem / (1024**3):.2f} GB total "
                    f"({mem_per_card / (1024**3):.2f} GB/card)")

        t0 = time.time()

        # ---- Ensure workers are sized for this circuit ----
        self._ensure_workers(nq)

        pset = set(range(nlq, nq))
        gates = self._parse_circuit(circuit)
        mq = self._get_measured_qubits(circuit)

        # ---- Initialize state on all cards ----
        self._broadcast(('init', nq))
        self._wait_all('init_done')

        # ---- Execute gates ----
        local_batch = []
        for gn, qb, pr in gates:
            if any(q in pset for q in qb):
                if local_batch:
                    self._exec_local(local_batch, nlq)
                    local_batch = []
                self._exec_global(gn, qb, pr, nq)
            else:
                local_batch.append((gn, qb, pr))
        if local_batch:
            self._exec_local(local_batch, nlq)

        # ---- Collect results ----
        self._broadcast(('collect',))
        parts = [None] * self.num_cards
        for _ in range(self.num_cards):
            r = self.result_queue.get(timeout=120)
            if r[0] == 'collect_done':
                parts[r[1]] = np.frombuffer(r[2], dtype=np.complex64).copy().reshape(r[3])

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

    def _wait_all(self, etype, timeout=120):
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
    print("FPGA Distributed Statevector Simulator (FIXED)")
    print("=" * 55)
    print(f"XRT Available:    {XRT_AVAILABLE}")
    print(f"Max Qubits:       {ABSOLUTE_MAX_QUBITS} (validation cap)")
    print(f"HBM Banks/Card:   {NUM_HBM_BANKS}")
    print(f"Start Method:     {mp.get_start_method()}")
    print()
    print("Key Fixes:")
    print("  - Dynamic shared memory (per-circuit, not MAX_QUBITS)")
    print("  - 'spawn' start method (no fork+XRT conflicts)")
    print("  - Deferred FPGA init (on first circuit, not constructor)")
    print("  - Proper resource cleanup and error handling")
    print()
    print("Architecture:")
    print("  - Statevector partitioned across K cards")
    print("  - Top log2(K) qubits = partition index")
    print("  - Local gates: vectorized execution on each partition")
    print("  - Global gates: 3-barrier double-buffered exchange")
    print()
    print("Usage:")
    print("  from fpga_distributed_statevector import FPGADistributedSimulator")
    print("  sim = FPGADistributedSimulator(num_cards=4)")
    print("  result = sim.run(circuit, shots=4096)")
    print("  sim.shutdown()")