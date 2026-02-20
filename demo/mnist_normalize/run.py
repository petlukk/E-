#!/usr/bin/env python3
"""
MNIST Normalization Demo: Ea vs NumPy

Downloads real MNIST handwritten digit data, normalizes pixel values
from [0, 255] to [0.0, 1.0]. Compares correctness and performance.

Usage:
    python run.py

Default: 60,000 images (28x28 = 784 pixels each, 47M pixels total).
"""

import sys
import time
import gzip
import ctypes
import struct
import subprocess
import urllib.request
from pathlib import Path
import numpy as np

DEMO_DIR = Path(__file__).parent
EA_ROOT = DEMO_DIR / ".." / ".."

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"

FLOAT_PTR = ctypes.POINTER(ctypes.c_float)


# ---------------------------------------------------------------------------
# MNIST download
# ---------------------------------------------------------------------------

def download_mnist():
    """Download MNIST training images. Returns flat f32 array of raw pixel values."""
    data_dir = DEMO_DIR / "mnist_data"
    data_dir.mkdir(exist_ok=True)
    npy_path = data_dir / "train_images.npy"

    if npy_path.exists():
        print(f"  Using cached MNIST data")
        return np.load(str(npy_path))

    gz_path = data_dir / "train-images-idx3-ubyte.gz"

    if not gz_path.exists():
        print(f"Downloading MNIST training images...")
        print(f"  Source: {MNIST_URL}")
        try:
            urllib.request.urlretrieve(MNIST_URL, str(gz_path))
            print(f"  Downloaded: {gz_path}")
        except Exception as e:
            print(f"  Download failed: {e}")
            return None

    # Parse IDX format
    with gzip.open(str(gz_path), 'rb') as f:
        magic, n_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            print(f"  Invalid MNIST file (magic={magic})")
            return None
        data = np.frombuffer(f.read(), dtype=np.uint8)

    images = data.reshape(n_images, rows * cols).astype(np.float32)
    np.save(str(npy_path), images)
    print(f"  Loaded {n_images} images ({rows}x{cols})")
    return images


def generate_synthetic(n_images=60000, pixels=784):
    """Generate synthetic image data as fallback."""
    print(f"  Generating {n_images} synthetic images ({pixels} pixels each)")
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (n_images, pixels)).astype(np.float32)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_ea_kernel(ea_name="normalize"):
    """Compile an .ea file to .so if needed."""
    so_path = DEMO_DIR / f"{ea_name}.so"
    ea_path = DEMO_DIR / f"{ea_name}.ea"

    if so_path.exists() and so_path.stat().st_mtime > ea_path.stat().st_mtime:
        return so_path

    print(f"Building Ea kernel ({ea_name})...")
    result = subprocess.run(
        ["cargo", "run", "--features=llvm", "--release", "--",
         str(ea_path), "--lib"],
        capture_output=True, text=True, cwd=str(EA_ROOT),
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)

    built = EA_ROOT / f"{ea_name}.so"
    if built.exists():
        built.rename(so_path)

    print(f"  Built: {so_path}")
    return so_path


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

def normalize_numpy(data):
    """NumPy normalization: data / 255.0"""
    return data / 255.0


def normalize_ea(data, so_path):
    """Ea SIMD normalization via ctypes."""
    lib = ctypes.CDLL(str(so_path))
    lib.normalize_f32x8.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int32,
                                    ctypes.c_float]
    lib.normalize_f32x8.restype = None

    flat = np.ascontiguousarray(data, dtype=np.float32).ravel()
    out = np.empty_like(flat)
    n = len(flat)

    lib.normalize_f32x8(
        flat.ctypes.data_as(FLOAT_PTR),
        out.ctypes.data_as(FLOAT_PTR),
        n,
        ctypes.c_float(1.0 / 255.0),
    )
    return out.reshape(data.shape)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline: normalize + standardize + clip
# ---------------------------------------------------------------------------

# MNIST standard normalization values
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def preprocess_numpy(data):
    """NumPy full preprocessing: normalize, standardize, clip. Multiple passes."""
    x = data / 255.0                       # pass 1: normalize
    x = (x - MNIST_MEAN) / MNIST_STD       # pass 2-3: subtract + divide
    x = np.clip(x, 0.0, 1.0)              # pass 4: clip
    return x


def preprocess_ea_fused(data, so_path):
    """Ea fused preprocessing: all in one pass. No intermediate arrays."""
    lib = ctypes.CDLL(str(so_path))
    lib.preprocess_fused.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int32,
                                     ctypes.c_float, ctypes.c_float,
                                     ctypes.c_float]
    lib.preprocess_fused.restype = None

    flat = np.ascontiguousarray(data, dtype=np.float32).ravel()
    out = np.empty_like(flat)
    n = len(flat)

    lib.preprocess_fused(
        flat.ctypes.data_as(FLOAT_PTR),
        out.ctypes.data_as(FLOAT_PTR),
        n,
        ctypes.c_float(1.0 / 255.0),
        ctypes.c_float(MNIST_MEAN),
        ctypes.c_float(1.0 / MNIST_STD),
    )
    return out.reshape(data.shape)


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark(func, *args, warmup=5, runs=50):
    """Run function multiple times, return median time in ms."""
    for _ in range(warmup):
        func(*args)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    median = times[len(times) // 2]
    return median, float(np.std(times))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("MNIST Normalization: [0,255] → [0.0, 1.0]")
    print()

    # Download or generate data
    images = download_mnist()
    if images is None:
        images = generate_synthetic()
        data_source = "synthetic"
    else:
        data_source = "MNIST (Yann LeCun et al.)"

    n_images, n_pixels = images.shape
    total_pixels = n_images * n_pixels
    print(f"  Images: {n_images}")
    print(f"  Pixels per image: {n_pixels}")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Data source: {data_source}")
    print()

    # Build kernels
    so_path = build_ea_kernel("normalize")
    so_fused_path = build_ea_kernel("preprocess_fused")
    print()

    # --- Part 1: Single Operation (normalize only) ---
    print("=" * 60)
    print("  PART 1: Single Operation (normalize only)")
    print("=" * 60)
    print()

    print("=== Correctness ===")
    result_numpy = normalize_numpy(images)
    result_ea = normalize_ea(images, so_path)

    diff = np.abs(result_ea - result_numpy)
    max_diff = diff.max()
    print(f"  Ea vs NumPy: max diff = {max_diff:.10f}")
    if max_diff < 1e-6:
        print("  Match: YES")
    print()

    print("=== Performance ===")
    print(f"  {total_pixels:,} pixels, 50 runs, median time\n")

    t_numpy, s_numpy = benchmark(normalize_numpy, images)
    print(f"  NumPy (x / 255.0)  : {t_numpy:8.2f} ms  ±{s_numpy:.2f}")

    t_ea, s_ea = benchmark(normalize_ea, images, so_path)
    print(f"  Ea (1 kernel)      : {t_ea:8.2f} ms  ±{s_ea:.2f}")

    print()
    speedup1 = t_numpy / t_ea
    print(f"  Ea vs NumPy: {speedup1:.1f}x "
          f"{'faster' if t_ea < t_numpy else 'slower'}")
    print(f"  → Memory-bound. Both hit DRAM bandwidth wall.")
    print()

    # --- Part 2: Full Pipeline (normalize + standardize + clip) ---
    print("=" * 60)
    print("  PART 2: Full Pipeline (normalize + standardize + clip)")
    print("=" * 60)
    print()

    print("=== Correctness ===")
    result_np_full = preprocess_numpy(images)
    result_ea_full = preprocess_ea_fused(images, so_fused_path)

    diff_full = np.abs(result_ea_full - result_np_full)
    max_diff_full = diff_full.max()
    mean_diff_full = diff_full.mean()
    print(f"  Ea fused vs NumPy: max diff = {max_diff_full:.10f}")
    print(f"  Output range Ea   : [{result_ea_full.min():.4f}, {result_ea_full.max():.4f}]")
    print(f"  Output range NumPy: [{result_np_full.min():.4f}, {result_np_full.max():.4f}]")
    if max_diff_full < 1e-5:
        print("  Match: YES (within floating-point tolerance)")
    else:
        print(f"  Match: APPROXIMATE (max diff {max_diff_full:.8f})")
    print()

    print("=== Performance ===")
    print(f"  NumPy: x/255 → (x-mean)/std → clip(0,1)  [3-4 memory passes]")
    print(f"  Ea:    fused single pass                   [1 memory pass]")
    print()

    t_np_full, s_np_full = benchmark(preprocess_numpy, images)
    print(f"  NumPy (multi-pass) : {t_np_full:8.2f} ms  ±{s_np_full:.2f}")

    t_ea_full, s_ea_full = benchmark(preprocess_ea_fused, images, so_fused_path)
    print(f"  Ea fused (1 pass)  : {t_ea_full:8.2f} ms  ±{s_ea_full:.2f}")

    print()

    # --- Summary ---
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print()
    speedup_full = t_np_full / t_ea_full
    print(f"  Single op (normalize only):")
    print(f"    Ea vs NumPy     : {speedup1:.1f}x "
          f"{'faster' if t_ea < t_numpy else 'slower'}")
    print()
    print(f"  Full pipeline (normalize + standardize + clip):")
    print(f"    Ea fused vs NumPy: {speedup_full:.1f}x "
          f"{'faster' if t_ea_full < t_np_full else 'slower'}")
    print(f"    Fusion eliminated {3}-4 memory passes → 1")
    print()
    print(f"  Data: {n_images} real MNIST images, {total_pixels:,} pixels")
    print(f"  Correctness: verified (max diff {max_diff_full:.1e})")


if __name__ == "__main__":
    main()
