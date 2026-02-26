#!/usr/bin/env python3
"""
FMA Kernel Benchmark: Ea vs hand-optimized C and competitors

Measures performance of fused multiply-add operations:
result[i] = a[i] * b[i] + c[i]

Tests f32x4 (SSE) and f32x8 (AVX2) implementations.
Competitors (Clang, ISPC, Rust std::simd) are included when available.
"""

import os
import sys
import subprocess
import time
import platform
import numpy as np
import ctypes
from pathlib import Path

# Make bench_common importable
sys.path.insert(0, str(Path(__file__).parent.parent))
import bench_common

# Configuration
ARRAY_SIZE = 1_000_000  # 1M elements for meaningful timing
NUM_RUNS = 100          # Average over many runs
WARMUP_RUNS = 10        # Warmup iterations

BENCH_DIR = Path(__file__).parent
FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
I32 = ctypes.c_int32
FMA_ARGTYPES = [FLOAT_PTR, FLOAT_PTR, FLOAT_PTR, FLOAT_PTR, I32]


def print_environment():
    """Print CPU, compiler, and OS info for reproducibility"""
    print("=== Environment ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Arch: {platform.machine()}")

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    print(f"CPU: {line.split(':')[1].strip()}")
                    break
    except FileNotFoundError:
        print(f"CPU: {platform.processor() or 'unknown'}")

    try:
        gcc = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
        print(f"GCC: {gcc.stdout.splitlines()[0]}")
    except FileNotFoundError:
        print("GCC: not found")

    try:
        llvm = subprocess.run(["llvm-config-14", "--version"],
                              capture_output=True, text=True)
        print(f"LLVM: {llvm.stdout.strip()}")
    except FileNotFoundError:
        print("LLVM: llvm-config-14 not found")

    bench_common.print_competitor_versions()


def compile_ea_kernel():
    """Compile Ea kernel to shared library"""
    print("Compiling Ea kernel...")

    result = subprocess.run([
        "cargo", "run", "--features=llvm", "--",
        "kernel.ea", "--lib"
    ], capture_output=True, text=True, cwd=BENCH_DIR)

    if result.returncode != 0:
        print(f"Ea compilation failed:")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)

    if not (BENCH_DIR / "kernel.so").exists():
        print("Error: kernel.so not created")
        sys.exit(1)

    print("Ea kernel compiled successfully")


def compile_c_reference():
    """Compile C reference with GCC and maximum optimization"""
    print("Compiling C reference (GCC)...")

    result = subprocess.run([
        "gcc", "-O3", "-march=native", "-ffast-math",
        "-shared", "-fPIC",
        "reference.c", "-o", "reference.so"
    ], capture_output=True, text=True, cwd=BENCH_DIR)

    if result.returncode != 0:
        print(f"C compilation failed:")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)

    print("C reference compiled successfully")


def load_libraries():
    """Load Ea and GCC libraries"""
    ea_lib = ctypes.CDLL(str(BENCH_DIR / "kernel.so"))
    c_lib = ctypes.CDLL(str(BENCH_DIR / "reference.so"))

    for name in ["fma_kernel_f32x4", "fma_kernel_f32x8",
                  "fma_kernel_foreach", "fma_kernel_foreach_unroll"]:
        fn = getattr(ea_lib, name)
        fn.argtypes = FMA_ARGTYPES
        fn.restype = None

    c_lib.fma_kernel_f32x4_c.argtypes = FMA_ARGTYPES
    c_lib.fma_kernel_f32x4_c.restype = None

    c_lib.fma_kernel_f32x8_c.argtypes = FMA_ARGTYPES
    c_lib.fma_kernel_f32x8_c.restype = None

    c_lib.fma_kernel_scalar_c.argtypes = FMA_ARGTYPES
    c_lib.fma_kernel_scalar_c.restype = None

    return ea_lib, c_lib


def compile_ea_kernel_at_opt(opt_level):
    """Compile Ea kernel at a specific optimization level, return .so path"""
    so_name = f"kernel_O{opt_level}.so"
    ea_root = BENCH_DIR.parent.parent
    result = subprocess.run([
        "cargo", "run", "--features=llvm", "--",
        str(BENCH_DIR / "kernel.ea"), "--lib",
        f"--opt-level={opt_level}", "-o", str(BENCH_DIR / so_name),
    ], capture_output=True, text=True, cwd=ea_root)
    if result.returncode != 0:
        print(f"  Ea O{opt_level} compilation failed: {result.stderr[:200]}")
        return None
    return BENCH_DIR / so_name


def compile_and_load_competitors():
    """Try to compile and load Clang, ISPC, and Rust competitors.

    Returns a list of (label, func) tuples for each available competitor.
    """
    competitors = []

    # --- Clang versions ---
    for ver in [14, 16, 17, 18]:
        clang = bench_common.has_clang(ver)
        if clang is None:
            continue
        so_name = f"reference_clang{ver}.so"
        print(f"Compiling with {clang}...")
        ok = bench_common.compile_with_clang(
            clang, "reference.c", so_name, BENCH_DIR
        )
        if not ok:
            continue
        lib = bench_common.try_load(BENCH_DIR / so_name, f"Clang-{ver}")
        if lib is None:
            continue
        # Set up signatures — same functions as GCC reference
        for fname in ["fma_kernel_f32x4_c", "fma_kernel_f32x8_c",
                       "fma_kernel_scalar_c"]:
            fn = getattr(lib, fname)
            fn.argtypes = FMA_ARGTYPES
            fn.restype = None
        competitors.append((f"Clang-{ver} f32x8", lib.fma_kernel_f32x8_c))
        competitors.append((f"Clang-{ver} f32x4", lib.fma_kernel_f32x4_c))
        break  # Use first available clang version

    # --- ISPC ---
    if bench_common.has_ispc():
        print("Compiling ISPC kernel...")
        ok = bench_common.compile_ispc(
            "fma_kernel.ispc", "fma_kernel_ispc.so", BENCH_DIR
        )
        if ok:
            lib = bench_common.try_load(
                BENCH_DIR / "fma_kernel_ispc.so", "ISPC"
            )
            if lib:
                lib.fma_kernel_ispc.argtypes = FMA_ARGTYPES
                lib.fma_kernel_ispc.restype = None
                competitors.append(("ISPC", lib.fma_kernel_ispc))

    # --- Rust std::simd ---
    if bench_common.has_rust_nightly():
        crate_dir = BENCH_DIR.parent / "rust_competitors"
        print("Building Rust competitors...")
        so_path = bench_common.compile_rust_competitors(crate_dir)
        if so_path:
            lib = bench_common.try_load(so_path, "Rust std::simd")
            if lib:
                lib.fma_kernel_f32x4_rust.argtypes = FMA_ARGTYPES
                lib.fma_kernel_f32x4_rust.restype = None
                competitors.append(("Rust std::simd", lib.fma_kernel_f32x4_rust))

    return competitors


def create_test_data():
    """Create test arrays with meaningful data"""
    np.random.seed(42)
    a = np.random.uniform(-1.0, 1.0, ARRAY_SIZE).astype(np.float32)
    b = np.random.uniform(-1.0, 1.0, ARRAY_SIZE).astype(np.float32)
    c = np.random.uniform(-1.0, 1.0, ARRAY_SIZE).astype(np.float32)
    return a, b, c


def benchmark_function(func, a, b, c, result, description):
    """Benchmark a single FMA function"""
    print(f"  Benchmarking {description}...")

    for _ in range(WARMUP_RUNS):
        func(a, b, c, result, ARRAY_SIZE)

    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        func(a, b, c, result, ARRAY_SIZE)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    return avg_time, min_time


def verify_correctness(ea_lib, c_lib, a, b, c):
    """Verify that Ea and C produce identical results"""
    print("Verifying correctness...")

    test_size = 100
    a_test = a[:test_size]
    b_test = b[:test_size]
    c_test = c[:test_size]

    a_ptr = a_test.ctypes.data_as(FLOAT_PTR)
    b_ptr = b_test.ctypes.data_as(FLOAT_PTR)
    c_ptr = c_test.ctypes.data_as(FLOAT_PTR)

    ea_result = np.zeros(test_size, dtype=np.float32)
    c_result = np.zeros(test_size, dtype=np.float32)

    ea_result_ptr = ea_result.ctypes.data_as(FLOAT_PTR)
    c_result_ptr = c_result.ctypes.data_as(FLOAT_PTR)

    ea_lib.fma_kernel_f32x4(a_ptr, b_ptr, c_ptr, ea_result_ptr, test_size)
    c_lib.fma_kernel_f32x4_c(a_ptr, b_ptr, c_ptr, c_result_ptr, test_size)

    if not np.allclose(ea_result, c_result, rtol=1e-5):
        print(f"ERROR: f32x4 results don't match!")
        idx = np.argmax(np.abs(ea_result - c_result))
        print(f"First difference at index: {idx}")
        print(f"Ea result: {ea_result[:10]}")
        print(f"C result: {c_result[:10]}")
        sys.exit(1)

    print("  Correctness verified")


def main():
    os.chdir(BENCH_DIR)

    print("=== FMA Kernel Benchmark ===")
    print_environment()
    print(f"Array size: {ARRAY_SIZE:,} elements")
    print(f"Runs per test: {NUM_RUNS}")
    print()

    # Compile core libraries
    compile_ea_kernel()
    compile_c_reference()
    ea_lib, c_lib = load_libraries()

    # Compile competitors (graceful — missing tools are skipped)
    competitors = compile_and_load_competitors()

    # Create test data
    a, b, c = create_test_data()
    a_ptr = a.ctypes.data_as(FLOAT_PTR)
    b_ptr = b.ctypes.data_as(FLOAT_PTR)
    c_ptr = c.ctypes.data_as(FLOAT_PTR)
    result = np.zeros(ARRAY_SIZE, dtype=np.float32)
    result_ptr = result.ctypes.data_as(FLOAT_PTR)

    verify_correctness(ea_lib, c_lib, a, b, c)

    print("\n=== Performance Results ===")

    # Build the full benchmark list: core + foreach + competitors
    bench_list = [
        ("GCC f32x8 (AVX2)", c_lib.fma_kernel_f32x8_c),
        ("GCC f32x4 (SSE)",  c_lib.fma_kernel_f32x4_c),
        ("GCC scalar",       c_lib.fma_kernel_scalar_c),
        ("Ea f32x8",         ea_lib.fma_kernel_f32x8),
        ("Ea f32x4",         ea_lib.fma_kernel_f32x4),
        ("Ea foreach",       ea_lib.fma_kernel_foreach),
        ("Ea foreach+unroll", ea_lib.fma_kernel_foreach_unroll),
    ]
    bench_list.extend(competitors)

    results = {}
    for label, func in bench_list:
        avg, mint = benchmark_function(
            func, a_ptr, b_ptr, c_ptr, result_ptr, label
        )
        results[label] = (avg, mint)

    # Find baseline: fastest C/SIMD implementation (GCC or Clang f32x8)
    c_simd_labels = [l for l in results if "f32x8" in l or "f32x4" in l]
    c_simd_labels = [l for l in c_simd_labels if not l.startswith("Ea")]
    baseline_label = min(c_simd_labels, key=lambda l: results[l][0])
    baseline_avg = results[baseline_label][0]

    # Print unified table
    print(f"\n  {'Implementation':<22} | {'Avg (us)':>10} | {'Min (us)':>10} "
          f"| {'vs Best C':>10}")
    print(f"  {'-'*22}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for label, _ in bench_list:
        avg, mint = results[label]
        ratio = avg / baseline_avg
        print(f"  {label:<22} | {avg*1e6:>10.1f} | {mint*1e6:>10.1f} "
              f"| {ratio:>9.3f}x")

    # Key ratios
    ea4_avg = results["Ea f32x4"][0]
    gcc4_avg = results["GCC f32x4 (SSE)"][0]
    ea4_vs_gcc4 = ea4_avg / gcc4_avg
    print(f"\nKey Result: Ea f32x4 vs GCC f32x4 = {ea4_vs_gcc4:.3f}x")

    if ea4_vs_gcc4 <= 1.1:
        print("  Within 10% of hand-optimized C!")
    else:
        print("  More than 10% slower than C")

    scalar_avg = results["GCC scalar"][0]
    speedup = scalar_avg / gcc4_avg
    print(f"SIMD speedup vs scalar = {speedup:.1f}x")

    # --- Opt-level comparison ---
    print("\n=== Optimization Level Comparison (Ea foreach) ===")
    print(f"  {'Opt Level':<12} | {'Avg (us)':>10} | {'Min (us)':>10} "
          f"| {'vs Best C':>10}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for opt in [0, 1, 2, 3]:
        so_path = compile_ea_kernel_at_opt(opt)
        if so_path is None:
            continue
        opt_lib = ctypes.CDLL(str(so_path))
        opt_lib.fma_kernel_foreach.argtypes = FMA_ARGTYPES
        opt_lib.fma_kernel_foreach.restype = None
        avg, mint = benchmark_function(
            opt_lib.fma_kernel_foreach, a_ptr, b_ptr, c_ptr, result_ptr,
            f"Ea foreach O{opt}"
        )
        ratio = avg / baseline_avg
        print(f"  O{opt:<11} | {avg*1e6:>10.1f} | {mint*1e6:>10.1f} "
              f"| {ratio:>9.3f}x")


if __name__ == "__main__":
    main()
