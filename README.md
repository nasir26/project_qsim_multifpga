# QSim Multi-FPGA — Dual-Engine Quantum Circuit Simulator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Dual-engine FPGA quantum circuit simulator: distributed exact statevector (28 qubits,
complex128) combined with MPS engine (500+ qubits) on 4× Xilinx Alveo U55C.

**Author**: Nasir Ali — Centre for Development of Advanced Computing (C-DAC), Noida

## Overview

This is the scaffold for a dual-engine quantum simulator that automatically routes
circuits to the most efficient backend:

- **Statevector engine**: exact simulation, ≤ 28 qubits, complex128 precision
- **MPS engine**: approximate simulation, 500+ qubits, bond dim ≤ χ_max

## Architecture

```
Circuit input
  → Engine selector (n_qubits, entanglement structure)
       ├── n ≤ 28, high entanglement → Statevector engine
       │     └── 4× Alveo U55C (HBM2, 8 banks, complex128)
       └── n > 28, low entanglement → MPS engine
             └── Alveo U55C (HBM2, site tensors, SVD truncation)
```

## Status

- [x] Phase A: complex128 statevector scaffold + QFT diagonal-local path
- [ ] Phase B: MPS engine integration
- [ ] Phase C: Hardware gate implementation

## Requirements

- XRT 2.16.204, Python 3.8+, NumPy, Qiskit

## Usage

```python
from qsim_engine import QSimMultiFPGA

sim = QSimMultiFPGA()
result = sim.run(circuit)   # auto-selects engine
print(result.fidelity)
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{nasirali_project_qsim_multifp,
  author    = {Nasir Ali},
  title     = {project qsim multifpga},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/nasir26/project_qsim_multifpga},
  note      = {Centre for Development of Advanced Computing (C-DAC), Noida, India}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.\
© 2026 Nasir Ali, C-DAC Noida. All rights reserved.
