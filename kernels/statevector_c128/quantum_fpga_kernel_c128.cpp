// quantum_fpga_kernel_c128.cpp
// Double-precision (complex128) statevector kernel for Xilinx Alveo U55C.
// 16-bank HBM interleave, amplitudes streamed from HBM (no BRAM state buffer).
// Gate parameters encoded as float32 IEEE 754 bits in int32 words; promoted to
// double on chip. Gate sequence: 8 int32 words / gate (unchanged from c64).
//
// Author: Nasir Ali, C-DAC Noida
// Layout: bank = idx % 16, slot = idx / 16
//         HBM[bank][slot*2]   = Re(amplitude)
//         HBM[bank][slot*2+1] = Im(amplitude)

#include <ap_int.h>
#include <hls_math.h>
#include <cstring>

#define NUM_HBM_BANKS 16

// Gate opcodes (identical to c64 reference)
#define GATE_H       0
#define GATE_X       1
#define GATE_Y       2
#define GATE_Z       3
#define GATE_S       4
#define GATE_T       5
#define GATE_SDG     6
#define GATE_TDG     7
#define GATE_RX      8
#define GATE_RY      9
#define GATE_RZ      10
#define GATE_P       11
#define GATE_CX      12
#define GATE_CY      13
#define GATE_CZ      14
#define GATE_CH      15
#define GATE_SWAP    16
#define GATE_CCX     17
#define GATE_SX      18
#define GATE_SXDG    19
#define GATE_U1      20
#define GATE_U2      21
#define GATE_U3      22
#define GATE_ID      23
#define GATE_CP      24
#define GATE_CRX     25
#define GATE_CRY     26
#define GATE_CRZ     27
#define GATE_ISWAP   28
#define GATE_ECR     29
#define GATE_RXX     30
#define GATE_RYY     31
#define GATE_RZZ     32
#define GATE_CSX     33
#define GATE_DCX     34

// ===========================================================
// HBM read/write helpers — double precision
// bank = global_idx & 0xF
// offset in bank = (global_idx >> 4) * 2
// ===========================================================

inline void read_amp_d(
    int global_idx,
    double &re, double &im,
    double* b0,  double* b1,  double* b2,  double* b3,
    double* b4,  double* b5,  double* b6,  double* b7,
    double* b8,  double* b9,  double* b10, double* b11,
    double* b12, double* b13, double* b14, double* b15
) {
    #pragma HLS INLINE
    int bank   = global_idx & 0xF;
    int offset = (global_idx >> 4) << 1;

    double* ptr;
    switch (bank) {
        case  0: ptr = b0;  break;
        case  1: ptr = b1;  break;
        case  2: ptr = b2;  break;
        case  3: ptr = b3;  break;
        case  4: ptr = b4;  break;
        case  5: ptr = b5;  break;
        case  6: ptr = b6;  break;
        case  7: ptr = b7;  break;
        case  8: ptr = b8;  break;
        case  9: ptr = b9;  break;
        case 10: ptr = b10; break;
        case 11: ptr = b11; break;
        case 12: ptr = b12; break;
        case 13: ptr = b13; break;
        case 14: ptr = b14; break;
        default: ptr = b15; break;
    }
    re = ptr[offset];
    im = ptr[offset + 1];
}

inline void write_amp_d(
    int global_idx,
    double re, double im,
    double* b0,  double* b1,  double* b2,  double* b3,
    double* b4,  double* b5,  double* b6,  double* b7,
    double* b8,  double* b9,  double* b10, double* b11,
    double* b12, double* b13, double* b14, double* b15
) {
    #pragma HLS INLINE
    int bank   = global_idx & 0xF;
    int offset = (global_idx >> 4) << 1;

    double* ptr;
    switch (bank) {
        case  0: ptr = b0;  break;
        case  1: ptr = b1;  break;
        case  2: ptr = b2;  break;
        case  3: ptr = b3;  break;
        case  4: ptr = b4;  break;
        case  5: ptr = b5;  break;
        case  6: ptr = b6;  break;
        case  7: ptr = b7;  break;
        case  8: ptr = b8;  break;
        case  9: ptr = b9;  break;
        case 10: ptr = b10; break;
        case 11: ptr = b11; break;
        case 12: ptr = b12; break;
        case 13: ptr = b13; break;
        case 14: ptr = b14; break;
        default: ptr = b15; break;
    }
    ptr[offset]     = re;
    ptr[offset + 1] = im;
}

#define BANK_PTRS_D \
    double* b0,  double* b1,  double* b2,  double* b3,  \
    double* b4,  double* b5,  double* b6,  double* b7,  \
    double* b8,  double* b9,  double* b10, double* b11, \
    double* b12, double* b13, double* b14, double* b15

#define BANK_ARGS_D \
    b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15

#define BANK_ARGS_K \
    bank0, bank1, bank2, bank3, bank4, bank5, bank6, bank7, \
    bank8, bank9, bank10, bank11, bank12, bank13, bank14, bank15

// ===========================================================
// Complex multiply: (ar+ai*i)(br+bi*i)
// ===========================================================
inline void cmul_d(double ar, double ai, double br, double bi,
                   double &rr, double &ri) {
    #pragma HLS INLINE
    rr = ar * br - ai * bi;
    ri = ar * bi + ai * br;
}

// ===========================================================
// Single-qubit gate: [[g00, g01],[g10, g11]] on target_qubit
// ===========================================================
static void apply_single_gate_d(
    BANK_PTRS_D,
    int target_qubit,
    int num_qubits,
    double g00r, double g00i, double g01r, double g01i,
    double g10r, double g10i, double g11r, double g11i
) {
    const int num_pairs = 1 << (num_qubits - 1);
    const int stride    = 1 << target_qubit;

    SG_LOOP: for (int p = 0; p < num_pairs; p++) {
        #pragma HLS PIPELINE II=16
        #pragma HLS LOOP_TRIPCOUNT min=8 max=33554432

        int idx0 = ((p >> target_qubit) << (target_qubit + 1)) |
                   (p & ((1 << target_qubit) - 1));
        int idx1 = idx0 + stride;

        double a0r, a0i, a1r, a1i;
        read_amp_d(idx0, a0r, a0i, BANK_ARGS_D);
        read_amp_d(idx1, a1r, a1i, BANK_ARGS_D);

        double t0r, t0i, t1r, t1i;
        cmul_d(g00r, g00i, a0r, a0i, t0r, t0i);
        cmul_d(g01r, g01i, a1r, a1i, t1r, t1i);
        double n0r = t0r + t1r;
        double n0i = t0i + t1i;

        cmul_d(g10r, g10i, a0r, a0i, t0r, t0i);
        cmul_d(g11r, g11i, a1r, a1i, t1r, t1i);
        double n1r = t0r + t1r;
        double n1i = t0i + t1i;

        write_amp_d(idx0, n0r, n0i, BANK_ARGS_D);
        write_amp_d(idx1, n1r, n1i, BANK_ARGS_D);
    }
}

// ===========================================================
// Controlled unitary: applies 2×2 to target when control=|1>
// ===========================================================
static void apply_controlled_gate_d(
    BANK_PTRS_D,
    int control_qubit,
    int target_qubit,
    int num_qubits,
    double g00r, double g00i, double g01r, double g01i,
    double g10r, double g10i, double g11r, double g11i
) {
    const int state_size = 1 << num_qubits;
    const int c_mask   = 1 << control_qubit;
    const int t_stride = 1 << target_qubit;

    CU_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=16
        #pragma HLS LOOP_TRIPCOUNT min=16 max=67108864

        if ((i & c_mask) && !(i & t_stride)) {
            int j = i | t_stride;

            double a0r, a0i, a1r, a1i;
            read_amp_d(i, a0r, a0i, BANK_ARGS_D);
            read_amp_d(j, a1r, a1i, BANK_ARGS_D);

            double t0r, t0i, t1r, t1i;
            cmul_d(g00r, g00i, a0r, a0i, t0r, t0i);
            cmul_d(g01r, g01i, a1r, a1i, t1r, t1i);
            double n0r = t0r + t1r;
            double n0i = t0i + t1i;

            cmul_d(g10r, g10i, a0r, a0i, t0r, t0i);
            cmul_d(g11r, g11i, a1r, a1i, t1r, t1i);
            double n1r = t0r + t1r;
            double n1i = t0i + t1i;

            write_amp_d(i, n0r, n0i, BANK_ARGS_D);
            write_amp_d(j, n1r, n1i, BANK_ARGS_D);
        }
    }
}

// ===========================================================
// CNOT
// ===========================================================
static void apply_cnot_d(
    BANK_PTRS_D,
    int control_qubit, int target_qubit, int num_qubits
) {
    const int state_size = 1 << num_qubits;
    const int c_mask = 1 << control_qubit;
    const int t_mask = 1 << target_qubit;

    CNOT_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=16
        #pragma HLS LOOP_TRIPCOUNT min=16 max=67108864

        if ((i & c_mask) && (i < (i ^ t_mask))) {
            int j = i ^ t_mask;

            double air, aii, ajr, aji;
            read_amp_d(i, air, aii, BANK_ARGS_D);
            read_amp_d(j, ajr, aji, BANK_ARGS_D);

            write_amp_d(i, ajr, aji, BANK_ARGS_D);
            write_amp_d(j, air, aii, BANK_ARGS_D);
        }
    }
}

// ===========================================================
// CZ
// ===========================================================
static void apply_cz_d(
    BANK_PTRS_D,
    int control_qubit, int target_qubit, int num_qubits
) {
    const int state_size = 1 << num_qubits;
    const int c_mask = 1 << control_qubit;
    const int t_mask = 1 << target_qubit;

    CZ_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=8
        #pragma HLS LOOP_TRIPCOUNT min=16 max=67108864

        if ((i & c_mask) && (i & t_mask)) {
            double re, im;
            read_amp_d(i, re, im, BANK_ARGS_D);
            write_amp_d(i, -re, -im, BANK_ARGS_D);
        }
    }
}

// ===========================================================
// SWAP
// ===========================================================
static void apply_swap_d(
    BANK_PTRS_D,
    int qubit0, int qubit1, int num_qubits
) {
    const int state_size = 1 << num_qubits;
    const int mask0 = 1 << qubit0;
    const int mask1 = 1 << qubit1;

    SWAP_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=16
        #pragma HLS LOOP_TRIPCOUNT min=16 max=67108864

        int bit0 = (i >> qubit0) & 1;
        int bit1 = (i >> qubit1) & 1;

        if (bit0 == 0 && bit1 == 1) {
            int j = (i ^ mask0) ^ mask1;

            double air, aii, ajr, aji;
            read_amp_d(i, air, aii, BANK_ARGS_D);
            read_amp_d(j, ajr, aji, BANK_ARGS_D);

            write_amp_d(i, ajr, aji, BANK_ARGS_D);
            write_amp_d(j, air, aii, BANK_ARGS_D);
        }
    }
}

// ===========================================================
// Toffoli (CCX)
// ===========================================================
static void apply_toffoli_d(
    BANK_PTRS_D,
    int control1, int control2, int target, int num_qubits
) {
    const int state_size = 1 << num_qubits;
    const int c1_mask = 1 << control1;
    const int c2_mask = 1 << control2;
    const int t_mask  = 1 << target;

    TOF_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=16
        #pragma HLS LOOP_TRIPCOUNT min=16 max=67108864

        if ((i & c1_mask) && (i & c2_mask) && (i < (i ^ t_mask))) {
            int j = i ^ t_mask;

            double air, aii, ajr, aji;
            read_amp_d(i, air, aii, BANK_ARGS_D);
            read_amp_d(j, ajr, aji, BANK_ARGS_D);

            write_amp_d(i, ajr, aji, BANK_ARGS_D);
            write_amp_d(j, air, aii, BANK_ARGS_D);
        }
    }
}

// ===========================================================
// iSWAP
// ===========================================================
static void apply_iswap_d(
    BANK_PTRS_D,
    int qubit0, int qubit1, int num_qubits
) {
    const int state_size = 1 << num_qubits;
    const int mask0 = 1 << qubit0;
    const int mask1 = 1 << qubit1;

    ISWAP_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=16
        #pragma HLS LOOP_TRIPCOUNT min=16 max=67108864

        int bit0 = (i >> qubit0) & 1;
        int bit1 = (i >> qubit1) & 1;

        if (bit0 == 0 && bit1 == 1) {
            int j = (i ^ mask0) ^ mask1;

            double air, aii, ajr, aji;
            read_amp_d(i, air, aii, BANK_ARGS_D);
            read_amp_d(j, ajr, aji, BANK_ARGS_D);

            write_amp_d(i, -aji, ajr, BANK_ARGS_D);
            write_amp_d(j, -aii, air, BANK_ARGS_D);
        }
    }
}

// ===========================================================
// ECR (Echoed Cross-Resonance)
// ===========================================================
static void apply_ecr_d(
    BANK_PTRS_D,
    int qubit0, int qubit1, int num_qubits
) {
    const int state_size = 1 << num_qubits;
    const int mask0 = 1 << qubit0;
    const int mask1 = 1 << qubit1;
    const double is2 = 0.7071067811865476;

    ECR_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=32
        #pragma HLS LOOP_TRIPCOUNT min=16 max=67108864

        if (!(i & mask0) && !(i & mask1)) {
            int i00 = i, i01 = i | mask1, i10 = i | mask0, i11 = i | mask0 | mask1;

            double a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i;
            read_amp_d(i00, a00r, a00i, BANK_ARGS_D);
            read_amp_d(i01, a01r, a01i, BANK_ARGS_D);
            read_amp_d(i10, a10r, a10i, BANK_ARGS_D);
            read_amp_d(i11, a11r, a11i, BANK_ARGS_D);

            write_amp_d(i00, is2*(a10r - a11i), is2*(a10i + a11r), BANK_ARGS_D);
            write_amp_d(i01, is2*(-a10i + a11r), is2*(a10r + a11i), BANK_ARGS_D);
            write_amp_d(i10, is2*(a00r + a01i), is2*(a00i - a01r), BANK_ARGS_D);
            write_amp_d(i11, is2*(a00i + a01r), is2*(-a00r + a01i), BANK_ARGS_D);
        }
    }
}

// ===========================================================
// RXX
// ===========================================================
static void apply_rxx_d(
    BANK_PTRS_D,
    int qubit0, int qubit1, int num_qubits, double theta
) {
    const int state_size = 1 << num_qubits;
    const int mask0 = 1 << qubit0;
    const int mask1 = 1 << qubit1;
    const double c = hls::cos(theta * 0.5);
    const double s = hls::sin(theta * 0.5);

    RXX_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=32
        #pragma HLS LOOP_TRIPCOUNT min=16 max=67108864

        if (!(i & mask0) && !(i & mask1)) {
            int i00 = i, i01 = i|mask1, i10 = i|mask0, i11 = i|mask0|mask1;

            double a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i;
            read_amp_d(i00, a00r, a00i, BANK_ARGS_D);
            read_amp_d(i01, a01r, a01i, BANK_ARGS_D);
            read_amp_d(i10, a10r, a10i, BANK_ARGS_D);
            read_amp_d(i11, a11r, a11i, BANK_ARGS_D);

            write_amp_d(i00, c*a00r + s*a11i, c*a00i - s*a11r, BANK_ARGS_D);
            write_amp_d(i01, c*a01r + s*a10i, c*a01i - s*a10r, BANK_ARGS_D);
            write_amp_d(i10, s*a01i + c*a10r, -s*a01r + c*a10i, BANK_ARGS_D);
            write_amp_d(i11, s*a00i + c*a11r, -s*a00r + c*a11i, BANK_ARGS_D);
        }
    }
}

// ===========================================================
// RYY
// ===========================================================
static void apply_ryy_d(
    BANK_PTRS_D,
    int qubit0, int qubit1, int num_qubits, double theta
) {
    const int state_size = 1 << num_qubits;
    const int mask0 = 1 << qubit0;
    const int mask1 = 1 << qubit1;
    const double c = hls::cos(theta * 0.5);
    const double s = hls::sin(theta * 0.5);

    RYY_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=32
        #pragma HLS LOOP_TRIPCOUNT min=16 max=67108864

        if (!(i & mask0) && !(i & mask1)) {
            int i00 = i, i01 = i|mask1, i10 = i|mask0, i11 = i|mask0|mask1;

            double a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i;
            read_amp_d(i00, a00r, a00i, BANK_ARGS_D);
            read_amp_d(i01, a01r, a01i, BANK_ARGS_D);
            read_amp_d(i10, a10r, a10i, BANK_ARGS_D);
            read_amp_d(i11, a11r, a11i, BANK_ARGS_D);

            write_amp_d(i00, c*a00r - s*a11i, c*a00i + s*a11r, BANK_ARGS_D);
            write_amp_d(i01, c*a01r + s*a10i, c*a01i - s*a10r, BANK_ARGS_D);
            write_amp_d(i10, s*a01i + c*a10r, -s*a01r + c*a10i, BANK_ARGS_D);
            write_amp_d(i11, -s*a00i + c*a11r, s*a00r + c*a11i, BANK_ARGS_D);
        }
    }
}

// ===========================================================
// RZZ (diagonal — only touches amplitude at index i)
// ===========================================================
static void apply_rzz_d(
    BANK_PTRS_D,
    int qubit0, int qubit1, int num_qubits, double theta
) {
    const int state_size = 1 << num_qubits;
    const double c  = hls::cos(theta * 0.5);
    const double s  = hls::sin(theta * 0.5);
    const double ps_r = c, ps_i = -s;  // same-parity phase: e^{-i theta/2}
    const double pd_r = c, pd_i =  s;  // diff-parity phase: e^{+i theta/2}

    RZZ_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=8
        #pragma HLS LOOP_TRIPCOUNT min=16 max=67108864

        int bit0 = (i >> qubit0) & 1;
        int bit1 = (i >> qubit1) & 1;

        double pr = (bit0 == bit1) ? ps_r : pd_r;
        double pi = (bit0 == bit1) ? ps_i : pd_i;

        double ar, ai;
        read_amp_d(i, ar, ai, BANK_ARGS_D);
        write_amp_d(i, pr*ar - pi*ai, pr*ai + pi*ar, BANK_ARGS_D);
    }
}

// DCX = CX(0,1) then CX(1,0)
static void apply_dcx_d(BANK_PTRS_D, int q0, int q1, int nq) {
    apply_cnot_d(BANK_ARGS_D, q0, q1, nq);
    apply_cnot_d(BANK_ARGS_D, q1, q0, nq);
}

// ===========================================================
// Top-level kernel — quantum_fpga_kernel_c128
// 17 AXI masters: bank0..bank15 (HBM) + gate_sequence (HBM)
// See docs/DECISIONS.md D-01 for routing-risk note.
// ===========================================================
extern "C" void quantum_fpga_kernel_c128(
    double* bank0,  double* bank1,  double* bank2,  double* bank3,
    double* bank4,  double* bank5,  double* bank6,  double* bank7,
    double* bank8,  double* bank9,  double* bank10, double* bank11,
    double* bank12, double* bank13, double* bank14, double* bank15,
    int*    gate_sequence,
    int     num_gates,
    int     num_qubits
) {
    #pragma HLS INTERFACE m_axi port=bank0  offset=slave bundle=bank0
    #pragma HLS INTERFACE m_axi port=bank1  offset=slave bundle=bank1
    #pragma HLS INTERFACE m_axi port=bank2  offset=slave bundle=bank2
    #pragma HLS INTERFACE m_axi port=bank3  offset=slave bundle=bank3
    #pragma HLS INTERFACE m_axi port=bank4  offset=slave bundle=bank4
    #pragma HLS INTERFACE m_axi port=bank5  offset=slave bundle=bank5
    #pragma HLS INTERFACE m_axi port=bank6  offset=slave bundle=bank6
    #pragma HLS INTERFACE m_axi port=bank7  offset=slave bundle=bank7
    #pragma HLS INTERFACE m_axi port=bank8  offset=slave bundle=bank8
    #pragma HLS INTERFACE m_axi port=bank9  offset=slave bundle=bank9
    #pragma HLS INTERFACE m_axi port=bank10 offset=slave bundle=bank10
    #pragma HLS INTERFACE m_axi port=bank11 offset=slave bundle=bank11
    #pragma HLS INTERFACE m_axi port=bank12 offset=slave bundle=bank12
    #pragma HLS INTERFACE m_axi port=bank13 offset=slave bundle=bank13
    #pragma HLS INTERFACE m_axi port=bank14 offset=slave bundle=bank14
    #pragma HLS INTERFACE m_axi port=bank15 offset=slave bundle=bank15
    #pragma HLS INTERFACE m_axi port=gate_sequence offset=slave bundle=gmem0

    #pragma HLS INTERFACE s_axilite port=bank0        bundle=control
    #pragma HLS INTERFACE s_axilite port=bank1        bundle=control
    #pragma HLS INTERFACE s_axilite port=bank2        bundle=control
    #pragma HLS INTERFACE s_axilite port=bank3        bundle=control
    #pragma HLS INTERFACE s_axilite port=bank4        bundle=control
    #pragma HLS INTERFACE s_axilite port=bank5        bundle=control
    #pragma HLS INTERFACE s_axilite port=bank6        bundle=control
    #pragma HLS INTERFACE s_axilite port=bank7        bundle=control
    #pragma HLS INTERFACE s_axilite port=bank8        bundle=control
    #pragma HLS INTERFACE s_axilite port=bank9        bundle=control
    #pragma HLS INTERFACE s_axilite port=bank10       bundle=control
    #pragma HLS INTERFACE s_axilite port=bank11       bundle=control
    #pragma HLS INTERFACE s_axilite port=bank12       bundle=control
    #pragma HLS INTERFACE s_axilite port=bank13       bundle=control
    #pragma HLS INTERFACE s_axilite port=bank14       bundle=control
    #pragma HLS INTERFACE s_axilite port=bank15       bundle=control
    #pragma HLS INTERFACE s_axilite port=gate_sequence bundle=control
    #pragma HLS INTERFACE s_axilite port=num_gates    bundle=control
    #pragma HLS INTERFACE s_axilite port=num_qubits   bundle=control
    #pragma HLS INTERFACE s_axilite port=return       bundle=control

    // Gate buffer in BRAM: 8 int32 words per gate, max 512 gates
    int gate_buf[512 * 8];
    #pragma HLS BIND_STORAGE variable=gate_buf type=ram_2p impl=bram

    int total_ints = num_gates * 8;
    READ_GATES: for (int i = 0; i < total_ints; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=8 max=4096
        gate_buf[i] = gate_sequence[i];
    }

    GATE_SEQ: for (int g = 0; g < num_gates; g++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=512

        int base = g * 8;
        int gate_type = gate_buf[base];
        int qubit0    = gate_buf[base + 1];
        int qubit1    = gate_buf[base + 2];
        int qubit2    = gate_buf[base + 3];

        // Parameters: float32 IEEE bits → promoted to double
        union { int i; float f; } p0, p1, p2;
        p0.i = gate_buf[base + 4];
        p1.i = gate_buf[base + 5];
        p2.i = gate_buf[base + 6];
        double d0 = (double)p0.f;
        double d1 = (double)p1.f;
        double d2 = (double)p2.f;

        const double H  = 0.7071067811865476;
        const double PI = 3.141592653589793;

        if (gate_type == GATE_ID) {
            // no-op
        } else if (gate_type == GATE_H) {
            apply_single_gate_d(BANK_ARGS_K, qubit0, num_qubits,
                H, 0,  H, 0,
                H, 0, -H, 0);
        } else if (gate_type == GATE_X) {
            apply_single_gate_d(BANK_ARGS_K, qubit0, num_qubits,
                0, 0, 1, 0,
                1, 0, 0, 0);
        } else if (gate_type == GATE_Y) {
            apply_single_gate_d(BANK_ARGS_K, qubit0, num_qubits,
                0, 0,  0, -1,
                0, 1,  0,  0);
        } else if (gate_type == GATE_Z) {
            apply_single_gate_d(BANK_ARGS_K, qubit0, num_qubits,
                1, 0, 0, 0,
                0, 0, -1, 0);
        } else if (gate_type == GATE_S) {
            apply_single_gate_d(BANK_ARGS_K, qubit0, num_qubits,
                1, 0, 0, 0,
                0, 0, 0, 1);
        } else if (gate_type == GATE_T) {
            apply_single_gate_d(BANK_ARGS_K, qubit0, num_qubits,
                1, 0, 0, 0,
                0, 0, H, H);
        } else if (gate_type == GATE_SDG) {
            apply_single_gate_d(BANK_ARGS_K, qubit0, num_qubits,
                1, 0, 0,  0,
                0, 0, 0, -1);
        } else if (gate_type == GATE_TDG) {
            apply_single_gate_d(BANK_ARGS_K, qubit0, num_qubits,
                1, 0, 0, 0,
                0, 0, H, -H);
        } else if (gate_type == GATE_SX) {
            apply_single_gate_d(BANK_ARGS_K, qubit0, num_qubits,
                0.5, 0.5,  0.5, -0.5,
                0.5, -0.5, 0.5,  0.5);
        } else if (gate_type == GATE_SXDG) {
            apply_single_gate_d(BANK_ARGS_K, qubit0, num_qubits,
                0.5, -0.5, 0.5,  0.5,
                0.5,  0.5, 0.5, -0.5);
        } else if (gate_type == GATE_RX) {
            double c = hls::cos(d0 * 0.5);
            double s = hls::sin(d0 * 0.5);
            apply_single_gate_d(BANK_ARGS_K, qubit0, num_qubits,
                c, 0, 0, -s,
                0, -s, c, 0);
        } else if (gate_type == GATE_RY) {
            double c = hls::cos(d0 * 0.5);
            double s = hls::sin(d0 * 0.5);
            apply_single_gate_d(BANK_ARGS_K, qubit0, num_qubits,
                c, 0, -s, 0,
                s, 0,  c, 0);
        } else if (gate_type == GATE_RZ) {
            double c = hls::cos(d0 * 0.5);
            double s = hls::sin(d0 * 0.5);
            apply_single_gate_d(BANK_ARGS_K, qubit0, num_qubits,
                c, -s, 0, 0,
                0,  0, c, s);
        } else if (gate_type == GATE_P || gate_type == GATE_U1) {
            double c = hls::cos(d0);
            double s = hls::sin(d0);
            apply_single_gate_d(BANK_ARGS_K, qubit0, num_qubits,
                1, 0, 0, 0,
                0, 0, c, s);
        } else if (gate_type == GATE_U2) {
            double cphi = hls::cos(d0), sphi = hls::sin(d0);
            double clam = hls::cos(d1), slam = hls::sin(d1);
            double cpl  = hls::cos(d0 + d1), spl = hls::sin(d0 + d1);
            apply_single_gate_d(BANK_ARGS_K, qubit0, num_qubits,
                H, 0,
                -H*clam, -H*slam,
                 H*cphi,  H*sphi,
                 H*cpl,   H*spl);
        } else if (gate_type == GATE_U3) {
            double ct  = hls::cos(d0 * 0.5), st = hls::sin(d0 * 0.5);
            double cphi = hls::cos(d1), sphi = hls::sin(d1);
            double clam = hls::cos(d2), slam = hls::sin(d2);
            double cpl  = hls::cos(d1 + d2), spl = hls::sin(d1 + d2);
            apply_single_gate_d(BANK_ARGS_K, qubit0, num_qubits,
                ct,   0,
                -st*clam, -st*slam,
                 st*cphi,  st*sphi,
                 ct*cpl,   ct*spl);
        } else if (gate_type == GATE_CX) {
            apply_cnot_d(BANK_ARGS_K, qubit0, qubit1, num_qubits);
        } else if (gate_type == GATE_CY) {
            apply_controlled_gate_d(BANK_ARGS_K, qubit0, qubit1, num_qubits,
                0, 0, 0, -1,
                0, 1, 0,  0);
        } else if (gate_type == GATE_CZ) {
            apply_cz_d(BANK_ARGS_K, qubit0, qubit1, num_qubits);
        } else if (gate_type == GATE_CH) {
            apply_controlled_gate_d(BANK_ARGS_K, qubit0, qubit1, num_qubits,
                H, 0,  H, 0,
                H, 0, -H, 0);
        } else if (gate_type == GATE_SWAP) {
            apply_swap_d(BANK_ARGS_K, qubit0, qubit1, num_qubits);
        } else if (gate_type == GATE_CP) {
            double c = hls::cos(d0), s = hls::sin(d0);
            apply_controlled_gate_d(BANK_ARGS_K, qubit0, qubit1, num_qubits,
                1, 0, 0, 0,
                0, 0, c, s);
        } else if (gate_type == GATE_CRX) {
            double c = hls::cos(d0 * 0.5), s = hls::sin(d0 * 0.5);
            apply_controlled_gate_d(BANK_ARGS_K, qubit0, qubit1, num_qubits,
                c, 0, 0, -s,
                0, -s, c, 0);
        } else if (gate_type == GATE_CRY) {
            double c = hls::cos(d0 * 0.5), s = hls::sin(d0 * 0.5);
            apply_controlled_gate_d(BANK_ARGS_K, qubit0, qubit1, num_qubits,
                c, 0, -s, 0,
                s, 0,  c, 0);
        } else if (gate_type == GATE_CRZ) {
            double c = hls::cos(d0 * 0.5), s = hls::sin(d0 * 0.5);
            apply_controlled_gate_d(BANK_ARGS_K, qubit0, qubit1, num_qubits,
                c, -s, 0, 0,
                0,  0, c, s);
        } else if (gate_type == GATE_CSX) {
            apply_controlled_gate_d(BANK_ARGS_K, qubit0, qubit1, num_qubits,
                0.5,  0.5, 0.5, -0.5,
                0.5, -0.5, 0.5,  0.5);
        } else if (gate_type == GATE_ISWAP) {
            apply_iswap_d(BANK_ARGS_K, qubit0, qubit1, num_qubits);
        } else if (gate_type == GATE_ECR) {
            apply_ecr_d(BANK_ARGS_K, qubit0, qubit1, num_qubits);
        } else if (gate_type == GATE_RXX) {
            apply_rxx_d(BANK_ARGS_K, qubit0, qubit1, num_qubits, d0);
        } else if (gate_type == GATE_RYY) {
            apply_ryy_d(BANK_ARGS_K, qubit0, qubit1, num_qubits, d0);
        } else if (gate_type == GATE_RZZ) {
            apply_rzz_d(BANK_ARGS_K, qubit0, qubit1, num_qubits, d0);
        } else if (gate_type == GATE_DCX) {
            apply_dcx_d(BANK_ARGS_K, qubit0, qubit1, num_qubits);
        } else if (gate_type == GATE_CCX) {
            apply_toffoli_d(BANK_ARGS_K, qubit0, qubit1, qubit2, num_qubits);
        }
    }
}
