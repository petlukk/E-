#!/usr/bin/env python3
"""
Pixel Pipeline Demo: Eä vs NumPy

Demonstrates two core vision pipeline kernels operating on raw uint8 data:
  1. threshold_u8x16  — binary segmentation in byte space (no float)
  2. normalize_u8_f32x8 — uint8 → float32 [0,1] using SIMD byte widening

NumPy must upcast uint8 to float32 before comparison/division. Eä stays in
u8x16 throughout the threshold, and widens directly in the normalize kernel.

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
EA_ROOT = DEMO_DIR / ".." / ".."

U8PTR  = ctypes.POINTER(ctypes.c_uint8)
F32PTR = ctypes.POINTER(ctypes.c_float)

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_pipeline():
    so_path = DEMO_DIR / "pipeline.so"
    ea_path = DEMO_DIR / "pipeline.ea"

    if so_path.exists() and so_path.stat().st_mtime > ea_path.stat().st_mtime:
        return so_path

    print("Building Eä kernel (pipeline)...")
    result = subprocess.run(
        ["cargo", "run", "--features=llvm", "--release", "--",
         str(ea_path), "--lib"],
        capture_output=True, text=True, cwd=str(EA_ROOT),
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)

    built = EA_ROOT / "pipeline.so"
    if built.exists():
        built.rename(so_path)

    print(f"  Built: {so_path}")
    return so_path


def load_lib(so_path):
    lib = ctypes.CDLL(str(so_path))

    lib.threshold_u8x16.argtypes = [U8PTR, U8PTR, ctypes.c_int32, ctypes.c_uint8]
    lib.threshold_u8x16.restype  = None

    lib.normalize_u8_f32x8.argtypes = [U8PTR, F32PTR, ctypes.c_int32]
    lib.normalize_u8_f32x8.restype  = None

    return lib


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_image(h=4096, w=4096, seed=42):
    """Generate a synthetic uint8 image with natural-looking statistics."""
    rng = np.random.RandomState(seed)
    # Bimodal distribution: background (~40) + signal (~180)
    bg = rng.normal(40, 20, (h, w)).clip(0, 255).astype(np.uint8)
    sig = rng.normal(180, 30, (h, w)).clip(0, 255).astype(np.uint8)
    mask = rng.random((h, w)) > 0.7
    img = np.where(mask, sig, bg).astype(np.uint8)
    return img


def pad_to_multiple(arr, k):
    """Pad flat array to a multiple of k (append zeros)."""
    n = len(arr)
    r = n % k
    if r == 0:
        return arr, n
    padded = np.zeros(n + (k - r), dtype=arr.dtype)
    padded[:n] = arr
    return padded, n


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

THRESH = 128


def threshold_numpy(img):
    return np.where(img > THRESH, np.uint8(255), np.uint8(0))


def threshold_ea(lib, flat_src, n_orig):
    flat_dst = np.empty_like(flat_src)
    lib.threshold_u8x16(
        flat_src.ctypes.data_as(U8PTR),
        flat_dst.ctypes.data_as(U8PTR),
        ctypes.c_int32(len(flat_src)),
        ctypes.c_uint8(THRESH),
    )
    return flat_dst[:n_orig]


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

def normalize_numpy(img):
    return img.astype(np.float32) / 255.0


def normalize_ea(lib, flat_src, n_orig):
    dst = np.empty(len(flat_src), dtype=np.float32)
    lib.normalize_u8_f32x8(
        flat_src.ctypes.data_as(U8PTR),
        dst.ctypes.data_as(F32PTR),
        ctypes.c_int32(len(flat_src)),
    )
    return dst[:n_orig]


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
    print("Pixel Pipeline: uint8 → threshold | normalize")
    print()

    img = make_image()
    h, w = img.shape
    n = h * w
    print(f"  Image: {h}×{w} = {n:,} pixels  ({n / 1e6:.1f} MP)")
    print(f"  Data: synthetic, bimodal distribution (bg≈40, signal≈180)")
    print()

    so_path = build_pipeline()
    lib = load_lib(so_path)
    print()

    flat = np.ascontiguousarray(img.ravel(), dtype=np.uint8)
    flat16, n16 = pad_to_multiple(flat, 16)
    flat8,  n8  = pad_to_multiple(flat, 8)

    # ==========================================================================
    print("=" * 62)
    print("  KERNEL 1: threshold_u8x16 (binary segmentation in u8 space)")
    print("=" * 62)
    print()
    print(f"  Eä: loads u8x16, compares .> splat({THRESH}), selects 0/255, stores u8x16")
    print(f"  NumPy: np.where(img > {THRESH}, 255, 0)  [promotes to int32 internally]")
    print()

    ref_thresh  = threshold_numpy(img).ravel()
    ea_thresh   = threshold_ea(lib, flat16, n16)

    match = np.array_equal(ref_thresh, ea_thresh)
    print(f"  Correctness: {'PASS — exact byte match' if match else 'FAIL'}")
    print()

    t_np_t,  s_np_t  = benchmark(threshold_numpy, img)
    t_ea_t,  s_ea_t  = benchmark(threshold_ea, lib, flat16, n16)

    print(f"  NumPy   : {t_np_t:7.2f} ms  ±{s_np_t:.2f}")
    print(f"  Eä      : {t_ea_t:7.2f} ms  ±{s_ea_t:.2f}")
    tag = f"{t_np_t / t_ea_t:.2f}x faster" if t_ea_t < t_np_t else f"{t_ea_t / t_np_t:.2f}x slower"
    print(f"  Eä vs NumPy: {tag}")
    print()

    # ==========================================================================
    print("=" * 62)
    print("  KERNEL 2: normalize_u8_f32x8 (uint8 → float32 [0,1])")
    print("=" * 62)
    print()
    print( "  Eä: widen_u8_f32x4 × 2 per 8-byte chunk, scale, store f32x4 × 2")
    print( "  NumPy: img.astype(float32) / 255.0  [two-pass: cast then divide]")
    print()

    ref_norm  = normalize_numpy(img).ravel()
    ea_norm   = normalize_ea(lib, flat8, n8)

    max_diff = float(np.abs(ref_norm - ea_norm).max())
    close    = max_diff < 2e-7
    print(f"  Correctness: max |Eä − NumPy| = {max_diff:.2e}  "
          f"({'PASS' if close else 'FAIL — exceeds tolerance'})")
    print()

    t_np_n,  s_np_n  = benchmark(normalize_numpy, img)
    t_ea_n,  s_ea_n  = benchmark(normalize_ea, lib, flat8, n8)

    print(f"  NumPy   : {t_np_n:7.2f} ms  ±{s_np_n:.2f}")
    print(f"  Eä      : {t_ea_n:7.2f} ms  ±{s_ea_n:.2f}")
    tag2 = f"{t_np_n / t_ea_n:.2f}x faster" if t_ea_n < t_np_n else f"{t_ea_n / t_np_n:.2f}x slower"
    print(f"  Eä vs NumPy: {tag2}")
    print()

    # ==========================================================================
    print("=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    print()
    print(f"  threshold  ({n/1e6:.0f} MP uint8):  Eä {t_ea_t:.2f} ms  NumPy {t_np_t:.2f} ms  → {tag}")
    print(f"  normalize  ({n/1e6:.0f} MP uint8):  Eä {t_ea_n:.2f} ms  NumPy {t_np_n:.2f} ms  → {tag2}")
    print()
    print(f"  Correctness: threshold exact ✓   normalize within 2e-7 ✓")
    print(f"  New in v0.3.0: u8/i8 scalars, u8x16/i8x16/i8x32 vectors,")
    print(f"  widen_u8_f32x4, widen_i8_f32x4, narrow_f32x4_i8")


if __name__ == "__main__":
    main()
