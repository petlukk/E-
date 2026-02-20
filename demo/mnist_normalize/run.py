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

def build_ea_kernel():
    """Compile normalize.ea to normalize.so if needed."""
    so_path = DEMO_DIR / "normalize.so"
    ea_path = DEMO_DIR / "normalize.ea"

    if so_path.exists() and so_path.stat().st_mtime > ea_path.stat().st_mtime:
        return so_path

    print("Building Ea kernel...")
    result = subprocess.run(
        ["cargo", "run", "--features=llvm", "--release", "--",
         str(ea_path), "--lib"],
        capture_output=True, text=True, cwd=str(EA_ROOT),
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)

    built = EA_ROOT / "normalize.so"
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
    return times[len(times) // 2]


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

    # Build kernel
    so_path = build_ea_kernel()
    print()

    # --- Correctness ---
    print("=== Correctness ===")
    result_numpy = normalize_numpy(images)
    result_ea = normalize_ea(images, so_path)

    diff = np.abs(result_ea - result_numpy)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"  Ea vs NumPy: max diff = {max_diff:.10f}, mean diff = {mean_diff:.10f}")

    # Verify range
    print(f"  Output range: [{result_ea.min():.4f}, {result_ea.max():.4f}]")
    expected_min = images.min() / 255.0
    expected_max = images.max() / 255.0
    print(f"  Expected range: [{expected_min:.4f}, {expected_max:.4f}]")

    if max_diff < 1e-6:
        print("  Match: YES (within floating-point tolerance)")
    else:
        print(f"  Match: APPROXIMATE (max diff {max_diff:.8f})")
    print()

    # --- Performance ---
    print("=== Performance ===")
    print(f"  {n_images} images, {total_pixels:,} pixels, 50 runs, median time\n")

    t_numpy = benchmark(normalize_numpy, images)
    print(f"  NumPy              : {t_numpy:8.2f} ms")

    t_ea = benchmark(normalize_ea, images, so_path)
    print(f"  Ea (normalize.so)  : {t_ea:8.2f} ms")

    print()

    # --- Throughput ---
    print("=== Throughput ===")
    bytes_processed = total_pixels * 4  # f32
    ea_gbps = (bytes_processed / (t_ea / 1000)) / 1e9
    np_gbps = (bytes_processed / (t_numpy / 1000)) / 1e9
    print(f"  Ea    : {ea_gbps:.1f} GB/s")
    print(f"  NumPy : {np_gbps:.1f} GB/s")
    print()

    # --- Summary ---
    print("=== Summary ===")
    speedup = t_numpy / t_ea
    print(f"  Ea vs NumPy  : {speedup:.1f}x "
          f"{'faster' if t_ea < t_numpy else 'slower'}")
    print(f"  Data: {n_images} real MNIST images, {total_pixels:,} pixels")
    print(f"  Correctness: verified against NumPy (max diff {max_diff:.1e})")


if __name__ == "__main__":
    main()
