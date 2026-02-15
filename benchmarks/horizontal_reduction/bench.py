#!/usr/bin/env python3
"""
Horizontal Reduction Benchmark: Ea vs hand-optimized C and competitors

Measures performance of reduction operations (sum, max, min) across arrays.
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
ARRAY_SIZE = 1_000_000
NUM_RUNS = 200
WARMUP_RUNS = 20

BENCH_DIR = Path(__file__).parent
FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
I32 = ctypes.c_int32
F32 = ctypes.c_float
REDUCE_ARGTYPES = [FLOAT_PTR, I32]


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
    ea_root = BENCH_DIR.parent.parent
    result = subprocess.run(
        [
            "cargo", "run", "--features=llvm", "--",
            str(BENCH_DIR / "kernel.ea"), "--lib",
        ],
        capture_output=True, text=True, cwd=ea_root,
    )
    if result.returncode != 0:
        print(f"Ea compilation failed:\nstdout: {result.stdout}\n"
              f"stderr: {result.stderr}")
        sys.exit(1)

    so_src = ea_root / "kernel.so"
    so_dst = BENCH_DIR / "kernel.so"
    if so_src.exists():
        so_src.rename(so_dst)
    elif not so_dst.exists():
        print("Error: kernel.so not created")
        sys.exit(1)

    print("Ea kernel compiled successfully")


def compile_c_reference():
    """Compile C reference with GCC and maximum optimization"""
    print("Compiling C reference (GCC)...")
    result = subprocess.run(
        [
            "gcc", "-O3", "-march=native", "-ffast-math",
            "-shared", "-fPIC",
            "reference.c", "-o", "reference.so",
        ],
        capture_output=True, text=True, cwd=BENCH_DIR,
    )
    if result.returncode != 0:
        print(f"C compilation failed:\nstdout: {result.stdout}\n"
              f"stderr: {result.stderr}")
        sys.exit(1)
    print("C reference compiled successfully")


def _setup_reduce_fn(lib, name):
    """Configure a reduction function's ctypes signature."""
    fn = getattr(lib, name)
    fn.argtypes = REDUCE_ARGTYPES
    fn.restype = F32
    return fn


def load_libraries():
    """Load Ea and GCC libraries and set up function signatures"""
    ea_lib = ctypes.CDLL(str(BENCH_DIR / "kernel.so"))
    c_lib = ctypes.CDLL(str(BENCH_DIR / "reference.so"))

    for name in ["sum_f32x4", "sum_f32x8", "max_f32x4", "min_f32x4"]:
        _setup_reduce_fn(ea_lib, name)

    for name in [
        "sum_f32x8_c", "sum_f32x4_c", "sum_scalar_c",
        "max_f32x4_c", "max_scalar_c",
        "min_f32x4_c", "min_scalar_c",
    ]:
        _setup_reduce_fn(c_lib, name)

    return ea_lib, c_lib


def compile_and_load_competitors():
    """Try to compile and load Clang, ISPC, and Rust competitors.

    Returns a dict with keys 'sum', 'max', 'min', each mapping to a list
    of (label, func) tuples.
    """
    competitors = {"sum": [], "max": [], "min": []}

    # --- Clang ---
    for ver in [14, 16, 17, 18]:
        clang = bench_common.has_clang(ver)
        if clang is None:
            continue
        so_name = f"reference_clang{ver}.so"
        print(f"Compiling reductions with {clang}...")
        ok = bench_common.compile_with_clang(
            clang, "reference.c", so_name, BENCH_DIR
        )
        if not ok:
            continue
        lib = bench_common.try_load(BENCH_DIR / so_name, f"Clang-{ver}")
        if lib is None:
            continue
        for name in [
            "sum_f32x8_c", "sum_f32x4_c", "sum_scalar_c",
            "max_f32x4_c", "max_scalar_c",
            "min_f32x4_c", "min_scalar_c",
        ]:
            _setup_reduce_fn(lib, name)
        tag = f"Clang-{ver}"
        competitors["sum"].append((f"{tag} f32x8", lib.sum_f32x8_c))
        competitors["sum"].append((f"{tag} f32x4", lib.sum_f32x4_c))
        competitors["max"].append((f"{tag} f32x4", lib.max_f32x4_c))
        competitors["min"].append((f"{tag} f32x4", lib.min_f32x4_c))
        break  # Use first available clang version

    # --- ISPC ---
    if bench_common.has_ispc():
        print("Compiling ISPC reduction kernels...")
        ok = bench_common.compile_ispc(
            "reductions.ispc", "reductions_ispc.so", BENCH_DIR
        )
        if ok:
            lib = bench_common.try_load(
                BENCH_DIR / "reductions_ispc.so", "ISPC"
            )
            if lib:
                _setup_reduce_fn(lib, "sum_ispc")
                _setup_reduce_fn(lib, "max_ispc")
                _setup_reduce_fn(lib, "min_ispc")
                competitors["sum"].append(("ISPC", lib.sum_ispc))
                competitors["max"].append(("ISPC", lib.max_ispc))
                competitors["min"].append(("ISPC", lib.min_ispc))

    # --- Rust std::simd ---
    if bench_common.has_rust_nightly():
        crate_dir = BENCH_DIR.parent / "rust_competitors"
        print("Building Rust competitors...")
        so_path = bench_common.compile_rust_competitors(crate_dir)
        if so_path:
            lib = bench_common.try_load(so_path, "Rust std::simd")
            if lib:
                _setup_reduce_fn(lib, "sum_f32x4_rust")
                _setup_reduce_fn(lib, "sum_f32x8_rust")
                _setup_reduce_fn(lib, "max_f32x4_rust")
                _setup_reduce_fn(lib, "min_f32x4_rust")
                competitors["sum"].append(
                    ("Rust f32x8", lib.sum_f32x8_rust))
                competitors["sum"].append(
                    ("Rust f32x4", lib.sum_f32x4_rust))
                competitors["max"].append(
                    ("Rust f32x4", lib.max_f32x4_rust))
                competitors["min"].append(
                    ("Rust f32x4", lib.min_f32x4_rust))

    return competitors


def benchmark_reduction(func, data_ptr, size, description):
    """Benchmark a single reduction function, return (avg_time, min_time, result)"""
    for _ in range(WARMUP_RUNS):
        func(data_ptr, size)

    times = []
    result = None
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        result = func(data_ptr, size)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    return avg_time, min_time, result


def verify_correctness(ea_lib, c_lib, data_ptr, size):
    """Verify Ea and C produce matching results"""
    print("Verifying correctness...")

    ea_sum4 = ea_lib.sum_f32x4(data_ptr, size)
    ea_sum8 = ea_lib.sum_f32x8(data_ptr, size)
    c_sum = c_lib.sum_scalar_c(data_ptr, size)

    ea_max = ea_lib.max_f32x4(data_ptr, size)
    c_max = c_lib.max_scalar_c(data_ptr, size)

    ea_min = ea_lib.min_f32x4(data_ptr, size)
    c_min = c_lib.min_scalar_c(data_ptr, size)

    def check(name, ea_val, c_val, rtol=1e-2):
        if abs(ea_val - c_val) > abs(c_val) * rtol + 1e-5:
            print(f"  MISMATCH {name}: Ea={ea_val}, C={c_val}")
            return False
        return True

    ok = True
    ok &= check("sum_f32x4", ea_sum4, c_sum)
    ok &= check("sum_f32x8", ea_sum8, c_sum)
    ok &= check("max_f32x4", ea_max, c_max)
    ok &= check("min_f32x4", ea_min, c_min)

    if ok:
        print("  Correctness verified")
    else:
        print("  WARNING: some results differ (may be FP ordering)")


def run_benchmark_group(name, variants, data_ptr, size):
    """Run a group of benchmarks and print a unified result table."""
    print(f"\n--- {name} ---")

    results = {}
    for label, func in variants:
        avg, mint, val = benchmark_reduction(func, data_ptr, size, label)
        results[label] = (avg, mint, val)
        sys.stdout.write(f"  {label}: done\n")

    # Baseline = fastest C SIMD entry (labels starting with "C " or "GCC ")
    c_labels = [l for l, _ in variants
                if l.startswith("C ") or l.startswith("GCC ")]
    if not c_labels:
        c_labels = [l for l, _ in variants]
    baseline_label = min(c_labels, key=lambda l: results[l][0])
    baseline_avg = results[baseline_label][0]

    print(f"\n  {'Implementation':<22} | {'Avg (us)':>10} | {'Min (us)':>10} "
          f"| {'vs Best C':>10}")
    print(f"  {'-'*22}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for label, _ in variants:
        avg, mint, _ = results[label]
        ratio = avg / baseline_avg
        print(f"  {label:<22} | {avg*1e6:>10.1f} | {mint*1e6:>10.1f} "
              f"| {ratio:>9.3f}x")

    return results


def main():
    os.chdir(BENCH_DIR)

    print("=== Horizontal Reduction Benchmark ===")
    print_environment()
    print(f"Array size: {ARRAY_SIZE:,} elements")
    print(f"Runs per test: {NUM_RUNS}")
    print()

    compile_ea_kernel()
    compile_c_reference()
    ea_lib, c_lib = load_libraries()

    # Compile competitors (graceful — missing tools are skipped)
    comp = compile_and_load_competitors()

    np.random.seed(42)
    data = np.random.uniform(-100.0, 100.0, ARRAY_SIZE).astype(np.float32)
    data_ptr = data.ctypes.data_as(FLOAT_PTR)

    verify_correctness(ea_lib, c_lib, data_ptr, ARRAY_SIZE)

    # --- Sum benchmarks ---
    sum_variants = [
        ("C f32x8 (AVX2)",   c_lib.sum_f32x8_c),
        ("C f32x4 (SSE)",    c_lib.sum_f32x4_c),
        ("C scalar",         c_lib.sum_scalar_c),
        ("Ea f32x8",         ea_lib.sum_f32x8),
        ("Ea f32x4",         ea_lib.sum_f32x4),
    ]
    sum_variants.extend(comp["sum"])
    sum_results = run_benchmark_group(
        "Sum Reduction", sum_variants, data_ptr, ARRAY_SIZE
    )

    # --- Max benchmarks ---
    max_variants = [
        ("C f32x4 (SSE)",    c_lib.max_f32x4_c),
        ("C scalar",         c_lib.max_scalar_c),
        ("Ea f32x4",         ea_lib.max_f32x4),
    ]
    max_variants.extend(comp["max"])
    max_results = run_benchmark_group(
        "Max Reduction", max_variants, data_ptr, ARRAY_SIZE
    )

    # --- Min benchmarks ---
    min_variants = [
        ("C f32x4 (SSE)",    c_lib.min_f32x4_c),
        ("C scalar",         c_lib.min_scalar_c),
        ("Ea f32x4",         ea_lib.min_f32x4),
    ]
    min_variants.extend(comp["min"])
    min_results = run_benchmark_group(
        "Min Reduction", min_variants, data_ptr, ARRAY_SIZE
    )

    # --- Summary ---
    print("\n=== Summary ===")

    ea_sum8_avg = sum_results["Ea f32x8"][0]
    c_sum8_avg = sum_results["C f32x8 (AVX2)"][0]
    print(f"Sum:  Ea f32x8 vs C f32x8 = {ea_sum8_avg / c_sum8_avg:.3f}x")

    ea_sum4_avg = sum_results["Ea f32x4"][0]
    c_sum4_avg = sum_results["C f32x4 (SSE)"][0]
    print(f"Sum:  Ea f32x4 vs C f32x4 = {ea_sum4_avg / c_sum4_avg:.3f}x")

    ea_max_avg = max_results["Ea f32x4"][0]
    c_max_avg = max_results["C f32x4 (SSE)"][0]
    print(f"Max:  Ea f32x4 vs C f32x4 = {ea_max_avg / c_max_avg:.3f}x")

    ea_min_avg = min_results["Ea f32x4"][0]
    c_min_avg = min_results["C f32x4 (SSE)"][0]
    print(f"Min:  Ea f32x4 vs C f32x4 = {ea_min_avg / c_min_avg:.3f}x")

    scalar_avg = sum_results["C scalar"][0]
    simd_speedup = scalar_avg / c_sum8_avg
    print(f"\nSIMD speedup (C scalar vs C f32x8): {simd_speedup:.1f}x")


if __name__ == "__main__":
    main()
