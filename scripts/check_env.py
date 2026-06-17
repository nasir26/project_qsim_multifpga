"""Pre-flight check — run before any FPGA benchmark."""
import sys, importlib, subprocess, os

REQUIRED_PY = (3, 8)
REQUIRED_PACKAGES = ["numpy", "scipy", "qiskit", "qiskit_aer"]

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

ok = True

# Python version
if sys.version_info >= REQUIRED_PY:
    print(f"{PASS} Python {sys.version.split()[0]}")
else:
    print(f"{FAIL} Python {sys.version.split()[0]} — need >= {REQUIRED_PY[0]}.{REQUIRED_PY[1]}")
    ok = False

# Packages
for pkg in REQUIRED_PACKAGES:
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, "__version__", "?")
        print(f"{PASS} {pkg} {ver}")
    except ImportError:
        print(f"{FAIL} {pkg} not found")
        ok = False

# numpy complex128 sanity
import numpy as np
a = np.array([1+2j], dtype=np.complex128)
assert a.dtype == np.complex128, "complex128 dtype broken"
print(f"{PASS} numpy complex128 dtype OK")

# XRT Python
try:
    import pyxrt  # noqa
    print(f"{PASS} pyxrt available")
except ImportError:
    try:
        sys.path.insert(0, "/opt/xilinx/xrt/python")
        import pyxrt  # noqa
        print(f"{WARN} pyxrt available (needed sys.path patch)")
    except ImportError:
        print(f"{WARN} pyxrt not found — FPGA paths will raise RuntimeError")

# XRT devices
try:
    import pyxrt
    n = pyxrt.num_devices()
    if n > 0:
        print(f"{PASS} {n} XRT device(s) found")
        for i in range(n):
            d = pyxrt.device(i)
            print(f"       card {i}: {d.get_info(pyxrt.xclbin_info.interface_uuid)}")
    else:
        print(f"{WARN} XRT available but 0 devices found")
except Exception as e:
    print(f"{WARN} XRT device enumeration failed: {e}")

# Vitis v++
try:
    r = subprocess.run(["v++", "--version"], capture_output=True, text=True, timeout=5)
    ver_line = r.stdout.split("\n")[0]
    print(f"{PASS} v++ found: {ver_line.strip()}")
except (FileNotFoundError, subprocess.TimeoutExpired):
    print(f"{WARN} v++ not in PATH — cannot build kernels")

# PLATFORM env var
plat = os.environ.get("PLATFORM", "")
if plat:
    print(f"{PASS} PLATFORM={plat}")
else:
    print(f"{WARN} PLATFORM not set — needed for v++ build")

# Results dir
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results = os.path.join(root, "results")
os.makedirs(results, exist_ok=True)
print(f"{PASS} results/ dir ready at {results}")

print()
if ok:
    print("All required checks passed.")
else:
    print("One or more required checks FAILED. Fix before running benchmarks.")
    sys.exit(1)
