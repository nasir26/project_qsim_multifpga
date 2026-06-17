#!/bin/bash
# Environment setup for qsim_multifpga
# Run once per shell session before any Python or v++ commands.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# XRT
if [ -f /opt/xilinx/xrt/setup.sh ]; then
    source /opt/xilinx/xrt/setup.sh
    echo "[setup] XRT loaded"
else
    echo "[WARN] XRT not found at /opt/xilinx/xrt/setup.sh"
fi

# Vitis
VITIS_SETTINGS="/tools/Xilinx/Vitis/2023.2/settings64.sh"
if [ -f "$VITIS_SETTINGS" ]; then
    source "$VITIS_SETTINGS"
    echo "[setup] Vitis 2023.2 loaded"
else
    echo "[WARN] Vitis not found at $VITIS_SETTINGS"
fi

# XRT Python bindings
export PYTHONPATH=/opt/xilinx/xrt/python:${PYTHONPATH:-}

# U55C platform
export PLATFORM=xilinx_u55c_gen3x16_xdma_3_202210_1

# Virtual environment
VENV="$ROOT/.venv"
if [ ! -d "$VENV" ]; then
    echo "[setup] Creating venv at $VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip -q
    "$VENV/bin/pip" install numpy scipy qiskit qiskit-aer -q
    echo "[setup] Python packages installed"
fi
source "$VENV/bin/activate"
echo "[setup] venv activated: $(python --version)"

# Results directory
mkdir -p "$ROOT/results"
echo "[setup] Done. PLATFORM=$PLATFORM"
