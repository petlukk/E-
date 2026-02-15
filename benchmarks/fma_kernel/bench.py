#!/usr/bin/env python3
"""
FMA Kernel Benchmark: Eä vs hand-optimized C

Measures performance of fused multiply-add operations:
result[i] = a[i] * b[i] + c[i]

Tests both f32x4 (SSE) and f32x8 (AVX2) implementations.
"""

import os
import sys
import subprocess
import time
import platform
import numpy as np
import ctypes
from pathlib import Path

# Configuration
ARRAY_SIZE = 1_000_000  # 1M elements for meaningful timing
NUM_RUNS = 100          # Average over many runs
WARMUP_RUNS = 10        # Warmup iterations

def print_environment():
    """Print CPU, compiler, and OS info for reproducibility"""
    print("=== Environment ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Arch: {platform.machine()}")

    # CPU model
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    print(f"CPU: {line.split(':')[1].strip()}")
                    break
    except FileNotFoundError:
        print(f"CPU: {platform.processor() or 'unknown'}")

    # GCC version
    try:
        gcc = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
        print(f"GCC: {gcc.stdout.splitlines()[0]}")
    except FileNotFoundError:
        print("GCC: not found")

    # LLVM version
    try:
        llvm = subprocess.run(["llvm-config-14", "--version"], capture_output=True, text=True)
        print(f"LLVM: {llvm.stdout.strip()}")
    except FileNotFoundError:
        print("LLVM: llvm-config-14 not found")

    print()


def compile_ea_kernel():
    """Compile Eä kernel to shared library"""
    print("Compiling Eä kernel...")
    
    # Compile to .so
    result = subprocess.run([
        "cargo", "run", "--features=llvm", "--", 
        "kernel.ea", "--lib"
    ], capture_output=True, text=True, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"Eä compilation failed:")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    
    # Should create kernel.so
    if not os.path.exists("kernel.so"):
        print("Error: kernel.so not created")
        sys.exit(1)
    
    print("Eä kernel compiled successfully")

def compile_c_reference():
    """Compile C reference with maximum optimization"""
    print("Compiling C reference...")
    
    result = subprocess.run([
        "gcc", "-O3", "-march=native", "-ffast-math",
        "-shared", "-fPIC", 
        "reference.c", "-o", "reference.so"
    ], capture_output=True, text=True, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"C compilation failed:")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    
    print("C reference compiled successfully")

def load_libraries():
    """Load compiled libraries"""
    ea_lib = ctypes.CDLL("./kernel.so")
    c_lib = ctypes.CDLL("./reference.so") 
    
    # Configure function signatures
    
    # f32x4 versions
    ea_lib.fma_kernel_f32x4.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32
    ]
    ea_lib.fma_kernel_f32x4.restype = None
    
    c_lib.fma_kernel_f32x4_c.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), 
        ctypes.c_int32
    ]
    c_lib.fma_kernel_f32x4_c.restype = None
    
    # Scalar version
    c_lib.fma_kernel_scalar_c.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32
    ]
    c_lib.fma_kernel_scalar_c.restype = None
    
    return ea_lib, c_lib

def create_test_data():
    """Create test arrays with meaningful data"""
    np.random.seed(42)  # Reproducible results
    
    a = np.random.uniform(-1.0, 1.0, ARRAY_SIZE).astype(np.float32)
    b = np.random.uniform(-1.0, 1.0, ARRAY_SIZE).astype(np.float32)
    c = np.random.uniform(-1.0, 1.0, ARRAY_SIZE).astype(np.float32)
    
    return a, b, c

def benchmark_function(func, a, b, c, result, description):
    """Benchmark a single function"""
    print(f"  Benchmarking {description}...")
    
    # Warmup
    for _ in range(WARMUP_RUNS):
        func(a, b, c, result, ARRAY_SIZE)
    
    # Actual timing
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
    """Verify that Eä and C produce identical results"""
    print("Verifying correctness...")
    
    # Small test arrays
    test_size = 100
    a_test = a[:test_size]
    b_test = b[:test_size] 
    c_test = c[:test_size]
    
    # Convert to ctypes arrays
    a_ptr = a_test.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b_test.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = c_test.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Results arrays
    ea_result = np.zeros(test_size, dtype=np.float32)
    c_result = np.zeros(test_size, dtype=np.float32)
    
    ea_result_ptr = ea_result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_result_ptr = c_result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Test f32x4 versions
    ea_lib.fma_kernel_f32x4(a_ptr, b_ptr, c_ptr, ea_result_ptr, test_size)
    c_lib.fma_kernel_f32x4_c(a_ptr, b_ptr, c_ptr, c_result_ptr, test_size)
    
    if not np.allclose(ea_result, c_result, rtol=1e-5):
        print(f"ERROR: f32x4 results don't match!")
        print(f"First difference at index: {np.argmax(np.abs(ea_result - c_result))}")
        print(f"Eä result: {ea_result[:10]}")
        print(f"C result: {c_result[:10]}")
        sys.exit(1)
    
    print("✓ Correctness verified")

def main():
    os.chdir(Path(__file__).parent)
    
    print("=== FMA Kernel Benchmark ===")
    print_environment()
    print(f"Array size: {ARRAY_SIZE:,} elements")
    print(f"Runs per test: {NUM_RUNS}")
    print()
    
    # Compile
    compile_ea_kernel()
    compile_c_reference()
    
    # Load libraries
    ea_lib, c_lib = load_libraries()
    
    # Create test data
    a, b, c = create_test_data()
    
    # Convert to ctypes arrays
    a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Result arrays
    result = np.zeros(ARRAY_SIZE, dtype=np.float32)
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Verify correctness
    verify_correctness(ea_lib, c_lib, a, b, c)
    
    print("\n=== Performance Results ===")
    
    # Benchmark all versions
    results = {}
    
    # f32x8 (AVX2) - C reference
    c_avg8, c_min8 = benchmark_function(
        c_lib.fma_kernel_f32x8_c, a_ptr, b_ptr, c_ptr, result_ptr,
        "C f32x8"  
    )
    results['c_f32x8'] = (c_avg8, c_min8)
    
    # f32x4 (SSE)
    ea_avg4, ea_min4 = benchmark_function(
        ea_lib.fma_kernel_f32x4, a_ptr, b_ptr, c_ptr, result_ptr,
        "Eä f32x4"
    )
    results['ea_f32x4'] = (ea_avg4, ea_min4)
    
    c_avg4, c_min4 = benchmark_function(
        c_lib.fma_kernel_f32x4_c, a_ptr, b_ptr, c_ptr, result_ptr,
        "C f32x4"
    )
    results['c_f32x4'] = (c_avg4, c_min4)
    
    # Scalar baseline
    scalar_avg, scalar_min = benchmark_function(
        c_lib.fma_kernel_scalar_c, a_ptr, b_ptr, c_ptr, result_ptr,
        "C scalar"
    )
    results['scalar'] = (scalar_avg, scalar_min)
    
    # Report results
    print("\nTiming Results (seconds):")
    print("Implementation     | Avg Time  | Min Time  | vs C f32x8")
    print("-------------------|-----------|-----------|----------")
    
    baseline_avg = results['c_f32x8'][0]
    
    for name, (avg, min_t) in results.items():
        ratio = avg / baseline_avg
        print(f"{name:18} | {avg:.6f} | {min_t:.6f} | {ratio:.3f}x")
    
    # Calculate key ratios
    
    ea4_vs_c4_ratio = results['ea_f32x4'][0] / results['c_f32x4'][0]
    print(f"\n🎯 Key Result: Eä f32x4 vs C f32x4 = {ea4_vs_c4_ratio:.3f}x")
    
    if ea4_vs_c4_ratio <= 1.1:
        print("✅ Within 10% of hand-optimized C!")
    else:
        print("❌ More than 10% slower than C")
    
    speedup_vs_scalar = results['scalar'][0] / results['c_f32x4'][0]
    print(f"📈 SIMD speedup vs scalar = {speedup_vs_scalar:.1f}x")

if __name__ == "__main__":
    main()