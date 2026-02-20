#!/usr/bin/env python3
"""
Fusion Scaling Experiment: How speedup grows with pipeline depth.

NumPy does N separate operations (N memory passes).
Ea fuses all N operations into a single pass (1 memory pass).

Measures 1, 2, 4, 6, 8 operations on 47M pixels (real MNIST data).
"""

import sys
import time
import ctypes
import subprocess
from pathlib import Path
import numpy as np

DEMO_DIR = Path(__file__).parent
EA_ROOT = DEMO_DIR / ".." / ".."

FLOAT_PTR = ctypes.POINTER(ctypes.c_float)

# Parameters (realistic ML preprocessing values)
SCALE = 1.0 / 255.0
MEAN = 0.1307
INV_STD = 1.0 / 0.3081
BIAS = 0.01
CONTRAST = 1.2


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data():
    """Load MNIST or generate synthetic."""
    npy_path = DEMO_DIR / "mnist_data" / "train_images.npy"
    if npy_path.exists():
        return np.load(str(npy_path))
    # Fallback
    print("  No MNIST data found. Run 'python run.py' first to download.")
    print("  Using synthetic data.")
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (60000, 784)).astype(np.float32)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_scaling_kernel():
    """Compile scaling.ea to scaling.so if needed."""
    so_path = DEMO_DIR / "scaling.so"
    ea_path = DEMO_DIR / "scaling.ea"

    if so_path.exists() and so_path.stat().st_mtime > ea_path.stat().st_mtime:
        return so_path

    print("Building scaling kernels...")
    result = subprocess.run(
        ["cargo", "run", "--features=llvm", "--release", "--",
         str(ea_path), "--lib"],
        capture_output=True, text=True, cwd=str(EA_ROOT),
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)

    built = EA_ROOT / "scaling.so"
    if built.exists():
        built.rename(so_path)

    print(f"  Built: {so_path}")
    return so_path


# ---------------------------------------------------------------------------
# NumPy pipelines (N separate operations = N memory passes)
# ---------------------------------------------------------------------------

def numpy_1op(data):
    x = data * SCALE
    return x

def numpy_2op(data):
    x = data * SCALE
    x = x - MEAN
    return x

def numpy_4op(data):
    x = data * SCALE
    x = x - MEAN
    x = x * INV_STD
    x = np.maximum(x, 0.0)
    return x

def numpy_6op(data):
    x = data * SCALE
    x = x - MEAN
    x = x * INV_STD
    x = np.maximum(x, 0.0)
    x = np.minimum(x, 1.0)
    x = x + BIAS
    return x

def numpy_8op(data):
    x = data * SCALE
    x = x - MEAN
    x = x * INV_STD
    x = np.maximum(x, 0.0)
    x = np.minimum(x, 1.0)
    x = x + BIAS
    x = x * CONTRAST
    x = np.maximum(x, 0.0)
    return x


# ---------------------------------------------------------------------------
# Ea fused pipelines (all operations in 1 pass)
# ---------------------------------------------------------------------------

def make_ea_funcs(so_path):
    """Set up ctypes wrappers for all fused kernels."""
    lib = ctypes.CDLL(str(so_path))

    # fused_1(input, out, len, s1)
    lib.fused_1.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int32,
                            ctypes.c_float]
    lib.fused_1.restype = None

    # fused_2(input, out, len, s1, s2)
    lib.fused_2.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int32,
                            ctypes.c_float, ctypes.c_float]
    lib.fused_2.restype = None

    # fused_4(input, out, len, s1, s2, s3)
    lib.fused_4.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int32,
                            ctypes.c_float, ctypes.c_float, ctypes.c_float]
    lib.fused_4.restype = None

    # fused_6(input, out, len, s1, s2, s3, s4)
    lib.fused_6.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int32,
                            ctypes.c_float, ctypes.c_float, ctypes.c_float,
                            ctypes.c_float]
    lib.fused_6.restype = None

    # fused_8(input, out, len, s1, s2, s3, s4, s5)
    lib.fused_8.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int32,
                            ctypes.c_float, ctypes.c_float, ctypes.c_float,
                            ctypes.c_float, ctypes.c_float]
    lib.fused_8.restype = None

    return lib


def ea_call(lib, func_name, flat, out, n, *params):
    """Call an Ea fused kernel."""
    func = getattr(lib, func_name)
    func(
        flat.ctypes.data_as(FLOAT_PTR),
        out.ctypes.data_as(FLOAT_PTR),
        n,
        *[ctypes.c_float(p) for p in params],
    )
    return out


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark(func, *args, warmup=5, runs=50):
    for _ in range(warmup):
        func(*args)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Fusion Scaling: How speedup grows with pipeline depth")
    print()

    data = load_data()
    flat = np.ascontiguousarray(data, dtype=np.float32).ravel()
    out = np.empty_like(flat)
    n = len(flat)
    total_mb = n * 4 / 1e6

    print(f"  Data: {data.shape[0]} MNIST images, {n:,} pixels ({total_mb:.0f} MB)")
    print()

    so_path = build_scaling_kernel()
    lib = make_ea_funcs(so_path)

    # Verify correctness for each level
    print("=== Correctness ===")
    numpy_funcs = [numpy_1op, numpy_2op, numpy_4op, numpy_6op, numpy_8op]
    ea_configs = [
        ("fused_1", [SCALE]),
        ("fused_2", [SCALE, MEAN]),
        ("fused_4", [SCALE, MEAN, INV_STD]),
        ("fused_6", [SCALE, MEAN, INV_STD, BIAS]),
        ("fused_8", [SCALE, MEAN, INV_STD, BIAS, CONTRAST]),
    ]
    op_counts = [1, 2, 4, 6, 8]

    for np_func, (ea_name, params), nops in zip(numpy_funcs, ea_configs, op_counts):
        np_result = np_func(data).ravel()
        ea_call(lib, ea_name, flat, out, n, *params)
        max_diff = np.abs(out - np_result).max()
        status = "OK" if max_diff < 1e-5 else f"DIFF={max_diff:.6f}"
        print(f"  {nops} ops: {status} (max diff {max_diff:.2e})")

    print()

    # Benchmark
    print("=== Fusion Scaling ===")
    print()
    print(f"  {'Ops':>4}  {'NumPy (ms)':>10}  {'Ea fused (ms)':>13}  {'Speedup':>8}  {'NumPy passes':>12}  {'Ea passes':>9}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*13}  {'─'*8}  {'─'*12}  {'─'*9}")

    results = []

    for np_func, (ea_name, params), nops in zip(numpy_funcs, ea_configs, op_counts):
        t_np = benchmark(np_func, data)
        ea_bench = lambda: ea_call(lib, ea_name, flat, out, n, *params)
        t_ea = benchmark(ea_bench)
        speedup = t_np / t_ea
        results.append((nops, t_np, t_ea, speedup))
        print(f"  {nops:>4}  {t_np:>10.1f}  {t_ea:>13.1f}  {speedup:>7.1f}x  {nops:>12}  {'1':>9}")

    print()

    # ASCII chart
    print("=== Speedup vs Pipeline Depth ===")
    print()
    max_speedup = max(r[3] for r in results)
    bar_width = 40
    for nops, t_np, t_ea, speedup in results:
        bar_len = int(speedup / max_speedup * bar_width)
        bar = "█" * bar_len
        print(f"  {nops} ops │{bar} {speedup:.1f}x")

    print()
    print("  NumPy: N ops = N memory passes (each reads + writes ~180 MB)")
    print("  Ea:    N ops = 1 memory pass   (all ops in SIMD registers)")
    print()
    print("  Speedup scales with pipeline depth because each additional")
    print("  NumPy operation adds another memory pass. Ea adds zero.")


if __name__ == "__main__":
    main()
