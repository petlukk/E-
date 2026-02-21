#!/usr/bin/env python3
"""
Conv2D Demo: Eä vs NumPy — Quantized Inference Kernels

Demonstrates maddubs (SSSE3 pmaddubsw): the SIMD instruction that powers
int8 quantized inference in TFLite, ONNX Runtime, XNNPACK.

  dot_u8i8(act, wt, n)      — uint8 activations × int8 weights → i16 sum
  conv1d_u8i8(src, wt, dst, n, k)  — sliding k-wide dot product

NumPy computes in int32; Eä accumulates in i16x8 using maddubs.
Keep activation values ≤ 10 and weight values ≤ 5 to avoid i16 overflow.

Usage:
    python run.py
"""

import sys
import time
import ctypes
import subprocess
from pathlib import Path
import numpy as np

DEMO_DIR = Path(__file__).parent
EA_ROOT  = DEMO_DIR / ".." / ".."

U8PTR  = ctypes.POINTER(ctypes.c_uint8)
I8PTR  = ctypes.POINTER(ctypes.c_int8)
I16PTR = ctypes.POINTER(ctypes.c_int16)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_conv():
    so_path = DEMO_DIR / "conv.so"
    ea_path = DEMO_DIR / "conv.ea"

    if so_path.exists() and so_path.stat().st_mtime > ea_path.stat().st_mtime:
        return so_path

    print("Building Eä kernel (conv)...")
    result = subprocess.run(
        ["cargo", "run", "--features=llvm", "--release", "--",
         str(ea_path), "--lib"],
        capture_output=True, text=True, cwd=str(EA_ROOT),
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)

    built = EA_ROOT / "conv.so"
    if built.exists():
        built.rename(so_path)

    print(f"  Built: {so_path}")
    return so_path


def load_lib(so_path):
    lib = ctypes.CDLL(str(so_path))

    lib.dot_u8i8.argtypes  = [U8PTR, I8PTR, ctypes.c_int32]
    lib.dot_u8i8.restype   = ctypes.c_int16

    lib.conv1d_u8i8.argtypes = [U8PTR, I8PTR, I16PTR, ctypes.c_int32, ctypes.c_int32]
    lib.conv1d_u8i8.restype  = None

    return lib


# ---------------------------------------------------------------------------
# Dot product
# ---------------------------------------------------------------------------

N_DOT = 512       # kept small so i16 accumulator doesn't overflow
N_BENCH = 65536   # for timing only — values chosen to avoid overflow too


def dot_numpy(act, wt):
    return int(np.dot(act.astype(np.int32), wt.astype(np.int32)))


def dot_ea(lib, act, wt):
    act_c = np.ascontiguousarray(act, dtype=np.uint8)
    wt_c  = np.ascontiguousarray(wt, dtype=np.int8)
    return int(lib.dot_u8i8(
        act_c.ctypes.data_as(U8PTR),
        wt_c.ctypes.data_as(I8PTR),
        ctypes.c_int32(len(act_c)),
    ))


# ---------------------------------------------------------------------------
# 1-D convolution
# ---------------------------------------------------------------------------

K = 16   # kernel width — must be multiple of 16
N_CONV = 4096


def conv1d_numpy(src, wt, n, k):
    # Correlate: output[j] = sum(src[j:j+k] * wt)
    src32 = src.astype(np.int32)
    wt32  = wt.astype(np.int32)
    out   = np.zeros(n, dtype=np.int32)
    for j in range(n):
        out[j] = int(np.dot(src32[j:j+k], wt32))
    return out


def conv1d_numpy_fast(src, wt, n, k):
    # Vectorised version using stride tricks
    src32 = src[:n + k - 1].astype(np.int32)
    wt32  = wt.astype(np.int32)
    shape   = (n, k)
    strides = (src32.strides[0], src32.strides[0])
    windows = np.lib.stride_tricks.as_strided(src32, shape=shape, strides=strides)
    return windows @ wt32


def conv1d_ea(lib, src, wt, n, k):
    src_c = np.ascontiguousarray(src, dtype=np.uint8)
    wt_c  = np.ascontiguousarray(wt, dtype=np.int8)
    dst   = np.zeros(n, dtype=np.int16)
    lib.conv1d_u8i8(
        src_c.ctypes.data_as(U8PTR),
        wt_c.ctypes.data_as(I8PTR),
        dst.ctypes.data_as(I16PTR),
        ctypes.c_int32(n),
        ctypes.c_int32(k),
    )
    return dst


# ---------------------------------------------------------------------------
# Benchmark
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
    return times[len(times) // 2], float(np.std(times))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Quantized Inference Kernels: uint8 × int8 (SSSE3 pmaddubsw)")
    print()

    so_path = build_conv()
    lib = load_lib(so_path)
    print()

    rng = np.random.RandomState(42)

    # ==========================================================================
    print("=" * 62)
    print("  KERNEL 1: dot_u8i8 — dot product (n=512, act∈[0,10], wt∈[-5,5])")
    print("=" * 62)
    print()
    print("  Eä:   maddubs(u8x16, i8x16) → i16x8, reduce_add → i16")
    print("  NumPy: np.dot(act.astype(int32), wt.astype(int32))")
    print()

    act_small = rng.randint(0, 11, N_DOT).astype(np.uint8)
    wt_small  = rng.randint(-5, 6, N_DOT).astype(np.int8)

    ref_dot = dot_numpy(act_small, wt_small)
    ea_dot  = dot_ea(lib, act_small, wt_small)
    match   = ref_dot == ea_dot
    print(f"  Correctness: NumPy={ref_dot}  Eä={ea_dot}  "
          f"{'PASS — exact match' if match else 'FAIL'}")
    print()

    # Performance on larger array (still no overflow with values ≤ 5)
    act_bench = rng.randint(0, 3, N_BENCH).astype(np.uint8)
    wt_bench  = rng.randint(-2, 3, N_BENCH).astype(np.int8)

    print(f"  Performance: n={N_BENCH:,} elements, 50 runs")
    t_np, s_np = benchmark(dot_numpy, act_bench, wt_bench)
    t_ea, s_ea = benchmark(dot_ea, lib, act_bench, wt_bench)
    tag = f"{t_np/t_ea:.2f}x faster" if t_ea < t_np else f"{t_ea/t_np:.2f}x slower"
    print(f"  NumPy (int32): {t_np:7.3f} ms  ±{s_np:.3f}")
    print(f"  Eä (maddubs) : {t_ea:7.3f} ms  ±{s_ea:.3f}")
    print(f"  Eä vs NumPy  : {tag}")
    print()

    # ==========================================================================
    print("=" * 62)
    print(f"  KERNEL 2: conv1d_u8i8 — 1-D convolution (n={N_CONV}, k={K})")
    print("=" * 62)
    print()
    print(f"  Eä: inner loop = maddubs per 16 bytes, store i16 output")
    print(f"  NumPy: stride-trick windows @ wt  (vectorised)")
    print()

    src_conv = rng.randint(0, 3, N_CONV + K - 1).astype(np.uint8)
    wt_conv  = rng.randint(-2, 3, K).astype(np.int8)

    ref_conv = conv1d_numpy_fast(src_conv, wt_conv, N_CONV, K).astype(np.int16)
    ea_conv  = conv1d_ea(lib, src_conv, wt_conv, N_CONV, K)
    close    = np.array_equal(ref_conv, ea_conv)
    print(f"  Correctness: all {N_CONV} outputs match: {'PASS' if close else 'FAIL'}")
    if not close:
        diff_idx = np.where(ref_conv != ea_conv)[0]
        print(f"    First mismatch at index {diff_idx[0]}: "
              f"NumPy={ref_conv[diff_idx[0]]}  Eä={ea_conv[diff_idx[0]]}")
    print()

    t_np2, s_np2 = benchmark(conv1d_numpy_fast, src_conv, wt_conv, N_CONV, K)
    t_ea2, s_ea2 = benchmark(conv1d_ea, lib, src_conv, wt_conv, N_CONV, K)
    tag2 = f"{t_np2/t_ea2:.2f}x faster" if t_ea2 < t_np2 else f"{t_ea2/t_np2:.2f}x slower"
    print(f"  NumPy (stride): {t_np2:7.3f} ms  ±{s_np2:.3f}")
    print(f"  Eä (maddubs)  : {t_ea2:7.3f} ms  ±{s_ea2:.3f}")
    print(f"  Eä vs NumPy   : {tag2}")
    print()

    # ==========================================================================
    print("=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    print()
    print(f"  dot_u8i8  : Eä {t_ea:.3f} ms  NumPy {t_np:.3f} ms  → {tag}")
    print(f"  conv1d    : Eä {t_ea2:.3f} ms  NumPy {t_np2:.3f} ms  → {tag2}")
    print()
    print(f"  Instruction: SSSE3 pmaddubsw (maddubs) — 1 cycle throughput")
    print(f"  New in v0.3.0: i16x8, i16x16 vectors + maddubs(u8x16, i8x16) → i16x8")


if __name__ == "__main__":
    main()
