#include <ap_int.h>
#include <hls_math.h>
#include <cstring>

// ============================================================
// Configuration
// ============================================================
#define NUM_HBM_BANKS 16
#define MAX_QUBITS 16
#define MAX_STATE_SIZE (1 << MAX_QUBITS)  // 65536

// Use separate float for real/imaginary to avoid std::complex HLS overhead
// Each HBM bank stores interleaved [re0, im0, re1, im1, ...]
// So each amplitude takes 2 floats = 8 bytes

// ============================================================
// Gate type encoding (unchanged)
// ============================================================
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
#define GATE_SX      18
#define GATE_SXDG    19
#define GATE_U1      20
#define GATE_U2      21
#define GATE_U3      22
#define GATE_ID      23
#define GATE_CX      12
#define GATE_CY      13
#define GATE_CZ      14
#define GATE_CH      15
#define GATE_SWAP    16
#define GATE_CCX     17
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

// ============================================================
// Amplitude read/write using interleaved real/imag floats
// Global index i -> bank = i % 16, offset_in_bank = i / 16
// Each amplitude = 2 floats, so float offset = (i/16)*2
// ============================================================

inline void read_amp(
    int global_idx,
    float &re, float &im,
    float* b0,  float* b1,  float* b2,  float* b3,
    float* b4,  float* b5,  float* b6,  float* b7,
    float* b8,  float* b9,  float* b10, float* b11,
    float* b12, float* b13, float* b14, float* b15
) {
    #pragma HLS INLINE
    int bank   = global_idx & 0xF;       // % 16
    int offset = (global_idx >> 4) << 1;  // (idx/16)*2
    
    float* ptr;
    switch(bank) {
        case 0:  ptr = b0;  break;
        case 1:  ptr = b1;  break;
        case 2:  ptr = b2;  break;
        case 3:  ptr = b3;  break;
        case 4:  ptr = b4;  break;
        case 5:  ptr = b5;  break;
        case 6:  ptr = b6;  break;
        case 7:  ptr = b7;  break;
        case 8:  ptr = b8;  break;
        case 9:  ptr = b9;  break;
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

inline void write_amp(
    int global_idx,
    float re, float im,
    float* b0,  float* b1,  float* b2,  float* b3,
    float* b4,  float* b5,  float* b6,  float* b7,
    float* b8,  float* b9,  float* b10, float* b11,
    float* b12, float* b13, float* b14, float* b15
) {
    #pragma HLS INLINE
    int bank   = global_idx & 0xF;
    int offset = (global_idx >> 4) << 1;
    
    float* ptr;
    switch(bank) {
        case 0:  ptr = b0;  break;
        case 1:  ptr = b1;  break;
        case 2:  ptr = b2;  break;
        case 3:  ptr = b3;  break;
        case 4:  ptr = b4;  break;
        case 5:  ptr = b5;  break;
        case 6:  ptr = b6;  break;
        case 7:  ptr = b7;  break;
        case 8:  ptr = b8;  break;
        case 9:  ptr = b9;  break;
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

// Macro for bank pointer parameter list
#define BANK_PTRS \
    float* b0,  float* b1,  float* b2,  float* b3,  \
    float* b4,  float* b5,  float* b6,  float* b7,  \
    float* b8,  float* b9,  float* b10, float* b11, \
    float* b12, float* b13, float* b14, float* b15

#define BANK_ARGS \
    b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15

#define BANK_ARGS_K \
    bank0, bank1, bank2, bank3, bank4, bank5, bank6, bank7, \
    bank8, bank9, bank10, bank11, bank12, bank13, bank14, bank15

// ============================================================
// Complex multiply helper: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
// ============================================================
inline void cmul(float ar, float ai, float br, float bi,
                 float &rr, float &ri) {
    #pragma HLS INLINE
    rr = ar * br - ai * bi;
    ri = ar * bi + ai * br;
}

// ============================================================
// Single-qubit gate: applies 2x2 unitary [[g00, g01],[g10, g11]]
// ============================================================
static void apply_single_gate(
    BANK_PTRS,
    int target_qubit,
    int num_qubits,
    float g00r, float g00i, float g01r, float g01i,
    float g10r, float g10i, float g11r, float g11i
) {
    const int num_pairs = 1 << (num_qubits - 1);
    const int stride    = 1 << target_qubit;
    
    SINGLE_GATE_LOOP: for (int p = 0; p < num_pairs; p++) {
        #pragma HLS PIPELINE II=16
        #pragma HLS LOOP_TRIPCOUNT min=8 max=32768
        
        int idx0 = ((p >> target_qubit) << (target_qubit + 1)) |
                   (p & ((1 << target_qubit) - 1));
        int idx1 = idx0 + stride;
        
        float a0r, a0i, a1r, a1i;
        read_amp(idx0, a0r, a0i, BANK_ARGS);
        read_amp(idx1, a1r, a1i, BANK_ARGS);
        
        // new0 = g00*a0 + g01*a1
        float t0r, t0i, t1r, t1i;
        cmul(g00r, g00i, a0r, a0i, t0r, t0i);
        cmul(g01r, g01i, a1r, a1i, t1r, t1i);
        float n0r = t0r + t1r;
        float n0i = t0i + t1i;
        
        // new1 = g10*a0 + g11*a1
        cmul(g10r, g10i, a0r, a0i, t0r, t0i);
        cmul(g11r, g11i, a1r, a1i, t1r, t1i);
        float n1r = t0r + t1r;
        float n1i = t0i + t1i;
        
        write_amp(idx0, n0r, n0i, BANK_ARGS);
        write_amp(idx1, n1r, n1i, BANK_ARGS);
    }
}

// ============================================================
// Controlled unitary: applies 2x2 to target when control=|1>
// ============================================================
static void apply_controlled_gate(
    BANK_PTRS,
    int control_qubit,
    int target_qubit,
    int num_qubits,
    float g00r, float g00i, float g01r, float g01i,
    float g10r, float g10i, float g11r, float g11i
) {
    const int state_size = 1 << num_qubits;
    const int c_mask   = 1 << control_qubit;
    const int t_stride = 1 << target_qubit;
    
    CU_GATE_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=16
        #pragma HLS LOOP_TRIPCOUNT min=16 max=65536
        
        if ((i & c_mask) && !(i & t_stride)) {
            int j = i | t_stride;
            
            float a0r, a0i, a1r, a1i;
            read_amp(i, a0r, a0i, BANK_ARGS);
            read_amp(j, a1r, a1i, BANK_ARGS);
            
            float t0r, t0i, t1r, t1i;
            cmul(g00r, g00i, a0r, a0i, t0r, t0i);
            cmul(g01r, g01i, a1r, a1i, t1r, t1i);
            float n0r = t0r + t1r;
            float n0i = t0i + t1i;
            
            cmul(g10r, g10i, a0r, a0i, t0r, t0i);
            cmul(g11r, g11i, a1r, a1i, t1r, t1i);
            float n1r = t0r + t1r;
            float n1i = t0i + t1i;
            
            write_amp(i, n0r, n0i, BANK_ARGS);
            write_amp(j, n1r, n1i, BANK_ARGS);
        }
    }
}

// ============================================================
// CNOT: swap amplitudes where control=1
// ============================================================
static void apply_cnot(
    BANK_PTRS,
    int control_qubit,
    int target_qubit,
    int num_qubits
) {
    const int state_size = 1 << num_qubits;
    const int c_mask = 1 << control_qubit;
    const int t_mask = 1 << target_qubit;
    
    CNOT_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=16
        #pragma HLS LOOP_TRIPCOUNT min=16 max=65536
        
        if ((i & c_mask) && (i < (i ^ t_mask))) {
            int j = i ^ t_mask;
            
            float air, aii, ajr, aji;
            read_amp(i, air, aii, BANK_ARGS);
            read_amp(j, ajr, aji, BANK_ARGS);
            
            write_amp(i, ajr, aji, BANK_ARGS);
            write_amp(j, air, aii, BANK_ARGS);
        }
    }
}

// ============================================================
// CZ gate: negate phase when both control and target = 1
// ============================================================
static void apply_cz(
    BANK_PTRS,
    int control_qubit,
    int target_qubit,
    int num_qubits
) {
    const int state_size = 1 << num_qubits;
    const int c_mask = 1 << control_qubit;
    const int t_mask = 1 << target_qubit;
    
    CZ_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=8
        #pragma HLS LOOP_TRIPCOUNT min=16 max=65536
        
        if ((i & c_mask) && (i & t_mask)) {
            float re, im;
            read_amp(i, re, im, BANK_ARGS);
            write_amp(i, -re, -im, BANK_ARGS);
        }
    }
}

// ============================================================
// SWAP gate
// ============================================================
static void apply_swap(
    BANK_PTRS,
    int qubit0,
    int qubit1,
    int num_qubits
) {
    const int state_size = 1 << num_qubits;
    const int mask0 = 1 << qubit0;
    const int mask1 = 1 << qubit1;
    
    SWAP_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=16
        #pragma HLS LOOP_TRIPCOUNT min=16 max=65536
        
        int bit0 = (i >> qubit0) & 1;
        int bit1 = (i >> qubit1) & 1;
        
        if (bit0 == 0 && bit1 == 1) {
            int j = (i ^ mask0) ^ mask1;
            
            float air, aii, ajr, aji;
            read_amp(i, air, aii, BANK_ARGS);
            read_amp(j, ajr, aji, BANK_ARGS);
            
            write_amp(i, ajr, aji, BANK_ARGS);
            write_amp(j, air, aii, BANK_ARGS);
        }
    }
}

// ============================================================
// Toffoli (CCX)
// ============================================================
static void apply_toffoli(
    BANK_PTRS,
    int control1, int control2, int target,
    int num_qubits
) {
    const int state_size = 1 << num_qubits;
    const int c1_mask = 1 << control1;
    const int c2_mask = 1 << control2;
    const int t_mask  = 1 << target;
    
    TOFFOLI_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=16
        #pragma HLS LOOP_TRIPCOUNT min=16 max=65536
        
        if ((i & c1_mask) && (i & c2_mask) && (i < (i ^ t_mask))) {
            int j = i ^ t_mask;
            
            float air, aii, ajr, aji;
            read_amp(i, air, aii, BANK_ARGS);
            read_amp(j, ajr, aji, BANK_ARGS);
            
            write_amp(i, ajr, aji, BANK_ARGS);
            write_amp(j, air, aii, BANK_ARGS);
        }
    }
}

// ============================================================
// iSWAP gate
// ============================================================
static void apply_iswap(
    BANK_PTRS,
    int qubit0, int qubit1,
    int num_qubits
) {
    const int state_size = 1 << num_qubits;
    const int mask0 = 1 << qubit0;
    const int mask1 = 1 << qubit1;
    
    ISWAP_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=16
        #pragma HLS LOOP_TRIPCOUNT min=16 max=65536
        
        int bit0 = (i >> qubit0) & 1;
        int bit1 = (i >> qubit1) & 1;
        
        if (bit0 == 0 && bit1 == 1) {
            int j = (i ^ mask0) ^ mask1;
            
            float air, aii, ajr, aji;
            read_amp(i, air, aii, BANK_ARGS);
            read_amp(j, ajr, aji, BANK_ARGS);
            
            // i * (ajr + aji*i) = -aji + ajr*i
            write_amp(i, -aji, ajr, BANK_ARGS);
            write_amp(j, -aii, air, BANK_ARGS);
        }
    }
}

// ============================================================
// ECR gate (Echoed Cross-Resonance)
// ============================================================
static void apply_ecr(
    BANK_PTRS,
    int qubit0, int qubit1,
    int num_qubits
) {
    const int state_size = 1 << num_qubits;
    const int mask0 = 1 << qubit0;
    const int mask1 = 1 << qubit1;
    const float inv_sqrt2 = 0.7071067811865476f;
    
    ECR_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=32
        #pragma HLS LOOP_TRIPCOUNT min=16 max=65536
        
        if (!(i & mask0) && !(i & mask1)) {
            int i00 = i;
            int i01 = i | mask1;
            int i10 = i | mask0;
            int i11 = i | mask0 | mask1;
            
            float a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i;
            read_amp(i00, a00r, a00i, BANK_ARGS);
            read_amp(i01, a01r, a01i, BANK_ARGS);
            read_amp(i10, a10r, a10i, BANK_ARGS);
            read_amp(i11, a11r, a11i, BANK_ARGS);
            
            // n00 = inv_sqrt2 * (a10 + i*a11)
            float n00r = inv_sqrt2 * (a10r - a11i);
            float n00i = inv_sqrt2 * (a10i + a11r);
            // n01 = inv_sqrt2 * (i*a10 + a11)
            float n01r = inv_sqrt2 * (-a10i + a11r);
            float n01i = inv_sqrt2 * (a10r + a11i);
            // n10 = inv_sqrt2 * (a00 - i*a01)
            float n10r = inv_sqrt2 * (a00r + a01i);
            float n10i = inv_sqrt2 * (a00i - a01r);
            // n11 = inv_sqrt2 * (-i*a00 + a01)
            float n11r = inv_sqrt2 * (a00i + a01r);
            float n11i = inv_sqrt2 * (-a00r + a01i);
            
            write_amp(i00, n00r, n00i, BANK_ARGS);
            write_amp(i01, n01r, n01i, BANK_ARGS);
            write_amp(i10, n10r, n10i, BANK_ARGS);
            write_amp(i11, n11r, n11i, BANK_ARGS);
        }
    }
}

// ============================================================
// RXX gate
// ============================================================
static void apply_rxx(
    BANK_PTRS,
    int qubit0, int qubit1,
    int num_qubits, float theta
) {
    const int state_size = 1 << num_qubits;
    const int mask0 = 1 << qubit0;
    const int mask1 = 1 << qubit1;
    const float c = hls::cosf(theta * 0.5f);
    const float s = hls::sinf(theta * 0.5f);
    
    RXX_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=32
        #pragma HLS LOOP_TRIPCOUNT min=16 max=65536
        
        if (!(i & mask0) && !(i & mask1)) {
            int i00 = i;
            int i01 = i | mask1;
            int i10 = i | mask0;
            int i11 = i | mask0 | mask1;
            
            float a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i;
            read_amp(i00, a00r, a00i, BANK_ARGS);
            read_amp(i01, a01r, a01i, BANK_ARGS);
            read_amp(i10, a10r, a10i, BANK_ARGS);
            read_amp(i11, a11r, a11i, BANK_ARGS);
            
            // n00 = c*a00 + (-is)*a11 = c*a00 + (s*a11i - i*s*a11r) -- wait
            // (-i*s)*(a11r+i*a11i) = -i*s*a11r + s*a11i = s*a11i + i*(-s*a11r)
            float n00r = c * a00r + s * a11i;
            float n00i = c * a00i - s * a11r;
            float n01r = c * a01r + s * a10i;
            float n01i = c * a01i - s * a10r;
            float n10r = s * a01i + c * a10r;
            float n10i = -s * a01r + c * a10i;
            float n11r = s * a00i + c * a11r;
            float n11i = -s * a00r + c * a11i;
            
            write_amp(i00, n00r, n00i, BANK_ARGS);
            write_amp(i01, n01r, n01i, BANK_ARGS);
            write_amp(i10, n10r, n10i, BANK_ARGS);
            write_amp(i11, n11r, n11i, BANK_ARGS);
        }
    }
}

// ============================================================
// RYY gate
// ============================================================
static void apply_ryy(
    BANK_PTRS,
    int qubit0, int qubit1,
    int num_qubits, float theta
) {
    const int state_size = 1 << num_qubits;
    const int mask0 = 1 << qubit0;
    const int mask1 = 1 << qubit1;
    const float c = hls::cosf(theta * 0.5f);
    const float s = hls::sinf(theta * 0.5f);
    
    RYY_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=32
        #pragma HLS LOOP_TRIPCOUNT min=16 max=65536
        
        if (!(i & mask0) && !(i & mask1)) {
            int i00 = i;
            int i01 = i | mask1;
            int i10 = i | mask0;
            int i11 = i | mask0 | mask1;
            
            float a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i;
            read_amp(i00, a00r, a00i, BANK_ARGS);
            read_amp(i01, a01r, a01i, BANK_ARGS);
            read_amp(i10, a10r, a10i, BANK_ARGS);
            read_amp(i11, a11r, a11i, BANK_ARGS);
            
            // (i*s)*(a11r+i*a11i) = -s*a11i + i*s*a11r
            float n00r = c * a00r - s * a11i;
            float n00i = c * a00i + s * a11r;
            // (-i*s)*(a10r+i*a10i) = s*a10i - i*s*a10r
            float n01r = c * a01r + s * a10i;
            float n01i = c * a01i - s * a10r;
            float n10r = s * a01i + c * a10r;
            float n10i = -s * a01r + c * a10i;
            // (i*s)*(a00r+i*a00i) = -s*a00i + i*s*a00r
            float n11r = -s * a00i + c * a11r;
            float n11i = s * a00r + c * a11i;
            
            write_amp(i00, n00r, n00i, BANK_ARGS);
            write_amp(i01, n01r, n01i, BANK_ARGS);
            write_amp(i10, n10r, n10i, BANK_ARGS);
            write_amp(i11, n11r, n11i, BANK_ARGS);
        }
    }
}

// ============================================================
// RZZ gate (diagonal)
// ============================================================
static void apply_rzz(
    BANK_PTRS,
    int qubit0, int qubit1,
    int num_qubits, float theta
) {
    const int state_size = 1 << num_qubits;
    const float c = hls::cosf(theta * 0.5f);
    const float s = hls::sinf(theta * 0.5f);
    
    // same parity: e^{-iÎ¸/2} = (c, -s), diff parity: e^{+iÎ¸/2} = (c, s)
    const float ps_r = c, ps_i = -s;  // phase_same
    const float pd_r = c, pd_i =  s;  // phase_diff
    
    RZZ_LOOP: for (int i = 0; i < state_size; i++) {
        #pragma HLS PIPELINE II=8
        #pragma HLS LOOP_TRIPCOUNT min=16 max=65536
        
        int bit0 = (i >> qubit0) & 1;
        int bit1 = (i >> qubit1) & 1;
        
        float pr = (bit0 == bit1) ? ps_r : pd_r;
        float pi = (bit0 == bit1) ? ps_i : pd_i;
        
        float ar, ai;
        read_amp(i, ar, ai, BANK_ARGS);
        
        // phase * amp
        float nr = pr * ar - pi * ai;
        float ni = pr * ai + pi * ar;
        
        write_amp(i, nr, ni, BANK_ARGS);
    }
}

// ============================================================
// DCX = CX(q0,q1) then CX(q1,q0)
// ============================================================
static void apply_dcx(
    BANK_PTRS,
    int qubit0, int qubit1,
    int num_qubits
) {
    apply_cnot(BANK_ARGS, qubit0, qubit1, num_qubits);
    apply_cnot(BANK_ARGS, qubit1, qubit0, num_qubits);
}

// ============================================================
// Top-level kernel
// ============================================================
extern "C" void quantum_simulator_kernel(
    float* bank0,  float* bank1,  float* bank2,  float* bank3,
    float* bank4,  float* bank5,  float* bank6,  float* bank7,
    float* bank8,  float* bank9,  float* bank10, float* bank11,
    float* bank12, float* bank13, float* bank14, float* bank15,
    int* gate_sequence,
    int num_gates,
    int num_qubits
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
    
    #pragma HLS INTERFACE s_axilite port=bank0       bundle=control
    #pragma HLS INTERFACE s_axilite port=bank1       bundle=control
    #pragma HLS INTERFACE s_axilite port=bank2       bundle=control
    #pragma HLS INTERFACE s_axilite port=bank3       bundle=control
    #pragma HLS INTERFACE s_axilite port=bank4       bundle=control
    #pragma HLS INTERFACE s_axilite port=bank5       bundle=control
    #pragma HLS INTERFACE s_axilite port=bank6       bundle=control
    #pragma HLS INTERFACE s_axilite port=bank7       bundle=control
    #pragma HLS INTERFACE s_axilite port=bank8       bundle=control
    #pragma HLS INTERFACE s_axilite port=bank9       bundle=control
    #pragma HLS INTERFACE s_axilite port=bank10      bundle=control
    #pragma HLS INTERFACE s_axilite port=bank11      bundle=control
    #pragma HLS INTERFACE s_axilite port=bank12      bundle=control
    #pragma HLS INTERFACE s_axilite port=bank13      bundle=control
    #pragma HLS INTERFACE s_axilite port=bank14      bundle=control
    #pragma HLS INTERFACE s_axilite port=bank15      bundle=control
    #pragma HLS INTERFACE s_axilite port=gate_sequence bundle=control
    #pragma HLS INTERFACE s_axilite port=num_gates   bundle=control
    #pragma HLS INTERFACE s_axilite port=num_qubits  bundle=control
    #pragma HLS INTERFACE s_axilite port=return      bundle=control
    
    // Read gate sequence into local buffer to reduce HBM accesses
    int gate_buf[512 * 8];  // max 512 gates
    #pragma HLS BIND_STORAGE variable=gate_buf type=ram_2p impl=bram
    
    int total_ints = num_gates * 8;
    READ_GATES: for (int i = 0; i < total_ints; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=8 max=4096
        gate_buf[i] = gate_sequence[i];
    }
    
    // Process gates sequentially
    GATE_SEQ_LOOP: for (int g = 0; g < num_gates; g++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=512
        
        int base = g * 8;
        int gate_type = gate_buf[base];
        int qubit0    = gate_buf[base + 1];
        int qubit1    = gate_buf[base + 2];
        int qubit2    = gate_buf[base + 3];
        
        // Reinterpret int bits as float for parameters
        union { int i; float f; } p0, p1, p2;
        p0.i = gate_buf[base + 4];
        p1.i = gate_buf[base + 5];
        p2.i = gate_buf[base + 6];
        
        // ==== Identity ====
        if (gate_type == GATE_ID) {
            // no-op
        }
        // ==== Hadamard ====
        else if (gate_type == GATE_H) {
            const float h = 0.7071067811865476f;
            apply_single_gate(BANK_ARGS_K, qubit0, num_qubits,
                h, 0, h, 0,   h, 0, -h, 0);
        }
        // ==== Pauli-X ====
        else if (gate_type == GATE_X) {
            apply_single_gate(BANK_ARGS_K, qubit0, num_qubits,
                0, 0, 1, 0,   1, 0, 0, 0);
        }
        // ==== Pauli-Y ====
        else if (gate_type == GATE_Y) {
            apply_single_gate(BANK_ARGS_K, qubit0, num_qubits,
                0, 0, 0, -1,  0, 1, 0, 0);
        }
        // ==== Pauli-Z ====
        else if (gate_type == GATE_Z) {
            apply_single_gate(BANK_ARGS_K, qubit0, num_qubits,
                1, 0, 0, 0,   0, 0, -1, 0);
        }
        // ==== S gate ====
        else if (gate_type == GATE_S) {
            apply_single_gate(BANK_ARGS_K, qubit0, num_qubits,
                1, 0, 0, 0,   0, 0, 0, 1);
        }
        // ==== T gate ====
        else if (gate_type == GATE_T) {
            const float t = 0.7071067811865476f;
            apply_single_gate(BANK_ARGS_K, qubit0, num_qubits,
                1, 0, 0, 0,   0, 0, t, t);
        }
        // ==== Sâ  gate ====
        else if (gate_type == GATE_SDG) {
            apply_single_gate(BANK_ARGS_K, qubit0, num_qubits,
                1, 0, 0, 0,   0, 0, 0, -1);
        }
        // ==== Tâ  gate ====
        else if (gate_type == GATE_TDG) {
            const float t = 0.7071067811865476f;
            apply_single_gate(BANK_ARGS_K, qubit0, num_qubits,
                1, 0, 0, 0,   0, 0, t, -t);
        }
        // ==== SX gate ====
        else if (gate_type == GATE_SX) {
            apply_single_gate(BANK_ARGS_K, qubit0, num_qubits,
                0.5f, 0.5f, 0.5f, -0.5f,
                0.5f, -0.5f, 0.5f, 0.5f);
        }
        // ==== SXâ  gate ====
        else if (gate_type == GATE_SXDG) {
            apply_single_gate(BANK_ARGS_K, qubit0, num_qubits,
                0.5f, -0.5f, 0.5f, 0.5f,
                0.5f, 0.5f, 0.5f, -0.5f);
        }
        // ==== RX gate ====
        else if (gate_type == GATE_RX) {
            float c = hls::cosf(p0.f * 0.5f);
            float s = hls::sinf(p0.f * 0.5f);
            apply_single_gate(BANK_ARGS_K, qubit0, num_qubits,
                c, 0, 0, -s,  0, -s, c, 0);
        }
        // ==== RY gate ====
        else if (gate_type == GATE_RY) {
            float c = hls::cosf(p0.f * 0.5f);
            float s = hls::sinf(p0.f * 0.5f);
            apply_single_gate(BANK_ARGS_K, qubit0, num_qubits,
                c, 0, -s, 0,  s, 0, c, 0);
        }
        // ==== RZ gate ====
        else if (gate_type == GATE_RZ) {
            float c = hls::cosf(p0.f * 0.5f);
            float s = hls::sinf(p0.f * 0.5f);
            apply_single_gate(BANK_ARGS_K, qubit0, num_qubits,
                c, -s, 0, 0,  0, 0, c, s);
        }
        // ==== Phase / U1 gate ====
        else if (gate_type == GATE_P || gate_type == GATE_U1) {
            float c = hls::cosf(p0.f);
            float s = hls::sinf(p0.f);
            apply_single_gate(BANK_ARGS_K, qubit0, num_qubits,
                1, 0, 0, 0,   0, 0, c, s);
        }
        // ==== U2 gate ====
        else if (gate_type == GATE_U2) {
            float inv_sqrt2 = 0.7071067811865476f;
            float cphi = hls::cosf(p0.f);
            float sphi = hls::sinf(p0.f);
            float clam = hls::cosf(p1.f);
            float slam = hls::sinf(p1.f);
            float cpl  = hls::cosf(p0.f + p1.f);
            float spl  = hls::sinf(p0.f + p1.f);
            apply_single_gate(BANK_ARGS_K, qubit0, num_qubits,
                inv_sqrt2, 0,
                -inv_sqrt2 * clam, -inv_sqrt2 * slam,
                inv_sqrt2 * cphi, inv_sqrt2 * sphi,
                inv_sqrt2 * cpl, inv_sqrt2 * spl);
        }
        // ==== U3 gate ====
        else if (gate_type == GATE_U3) {
            float ct = hls::cosf(p0.f * 0.5f);
            float st = hls::sinf(p0.f * 0.5f);
            float cphi = hls::cosf(p1.f);
            float sphi = hls::sinf(p1.f);
            float clam = hls::cosf(p2.f);
            float slam = hls::sinf(p2.f);
            float cpl  = hls::cosf(p1.f + p2.f);
            float spl  = hls::sinf(p1.f + p2.f);
            apply_single_gate(BANK_ARGS_K, qubit0, num_qubits,
                ct, 0,
                -st * clam, -st * slam,
                st * cphi, st * sphi,
                ct * cpl, ct * spl);
        }
        
        // ==== Two-qubit gates ====
        
        else if (gate_type == GATE_CX) {
            apply_cnot(BANK_ARGS_K, qubit0, qubit1, num_qubits);
        }
        else if (gate_type == GATE_CY) {
            apply_controlled_gate(BANK_ARGS_K, qubit0, qubit1, num_qubits,
                0, 0, 0, -1,  0, 1, 0, 0);
        }
        else if (gate_type == GATE_CZ) {
            apply_cz(BANK_ARGS_K, qubit0, qubit1, num_qubits);
        }
        else if (gate_type == GATE_CH) {
            const float h = 0.7071067811865476f;
            apply_controlled_gate(BANK_ARGS_K, qubit0, qubit1, num_qubits,
                h, 0, h, 0,  h, 0, -h, 0);
        }
        else if (gate_type == GATE_SWAP) {
            apply_swap(BANK_ARGS_K, qubit0, qubit1, num_qubits);
        }
        else if (gate_type == GATE_CP) {
            float c = hls::cosf(p0.f);
            float s = hls::sinf(p0.f);
            apply_controlled_gate(BANK_ARGS_K, qubit0, qubit1, num_qubits,
                1, 0, 0, 0,  0, 0, c, s);
        }
        else if (gate_type == GATE_CRX) {
            float c = hls::cosf(p0.f * 0.5f);
            float s = hls::sinf(p0.f * 0.5f);
            apply_controlled_gate(BANK_ARGS_K, qubit0, qubit1, num_qubits,
                c, 0, 0, -s,  0, -s, c, 0);
        }
        else if (gate_type == GATE_CRY) {
            float c = hls::cosf(p0.f * 0.5f);
            float s = hls::sinf(p0.f * 0.5f);
            apply_controlled_gate(BANK_ARGS_K, qubit0, qubit1, num_qubits,
                c, 0, -s, 0,  s, 0, c, 0);
        }
        else if (gate_type == GATE_CRZ) {
            float c = hls::cosf(p0.f * 0.5f);
            float s = hls::sinf(p0.f * 0.5f);
            apply_controlled_gate(BANK_ARGS_K, qubit0, qubit1, num_qubits,
                c, -s, 0, 0,  0, 0, c, s);
        }
        else if (gate_type == GATE_CSX) {
            apply_controlled_gate(BANK_ARGS_K, qubit0, qubit1, num_qubits,
                0.5f, 0.5f, 0.5f, -0.5f,
                0.5f, -0.5f, 0.5f, 0.5f);
        }
        else if (gate_type == GATE_ISWAP) {
            apply_iswap(BANK_ARGS_K, qubit0, qubit1, num_qubits);
        }
        else if (gate_type == GATE_ECR) {
            apply_ecr(BANK_ARGS_K, qubit0, qubit1, num_qubits);
        }
        else if (gate_type == GATE_RXX) {
            apply_rxx(BANK_ARGS_K, qubit0, qubit1, num_qubits, p0.f);
        }
        else if (gate_type == GATE_RYY) {
            apply_ryy(BANK_ARGS_K, qubit0, qubit1, num_qubits, p0.f);
        }
        else if (gate_type == GATE_RZZ) {
            apply_rzz(BANK_ARGS_K, qubit0, qubit1, num_qubits, p0.f);
        }
        else if (gate_type == GATE_DCX) {
            apply_dcx(BANK_ARGS_K, qubit0, qubit1, num_qubits);
        }
        
        // ==== Three-qubit gates ====
        else if (gate_type == GATE_CCX) {
            apply_toffoli(BANK_ARGS_K, qubit0, qubit1, qubit2, num_qubits);
        }
    }
}