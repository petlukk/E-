#!/usr/bin/env python3
"""
Astronomy Frame Stacking Demo: Ea vs NumPy

Stacks N noisy exposures of a starfield to reduce noise.
Signal reinforces, noise cancels by sqrt(N).

Usage:
    python run.py [N_frames]

Default: 16 frames, 1024x1024 synthetic starfield.
"""

import sys
import time
import ctypes
import subprocess
from pathlib import Path
import numpy as np

DEMO_DIR = Path(__file__).parent
EA_ROOT = DEMO_DIR / ".." / ".."

N_FRAMES = 16
NOISE_SIGMA = 0.05

FLOAT_PTR = ctypes.POINTER(ctypes.c_float)


# ---------------------------------------------------------------------------
# Starfield generation
# ---------------------------------------------------------------------------

def generate_starfield(width=1024, height=1024, seed=42):
    """Generate a synthetic starfield: stars as 2D gaussians, a nebula, sky glow."""
    rng = np.random.RandomState(seed)
    img = np.zeros((height, width), dtype=np.float32)

    # Sky glow background
    yy, xx = np.mgrid[0:height, 0:width]
    img += 0.02 + 0.01 * np.exp(-((yy - height * 0.5) ** 2) / (2 * (height * 0.4) ** 2))

    # Nebula: broad elliptical glow
    cx_neb, cy_neb = width * 0.35, height * 0.6
    sx_neb, sy_neb = width * 0.15, height * 0.10
    nebula = 0.08 * np.exp(-(((xx - cx_neb) / sx_neb) ** 2 +
                              ((yy - cy_neb) / sy_neb) ** 2) / 2)
    img += nebula.astype(np.float32)

    # Stars: 60-80 point sources as 2D gaussians
    n_stars = rng.randint(60, 81)
    for _ in range(n_stars):
        cx = rng.uniform(0, width)
        cy = rng.uniform(0, height)
        brightness = rng.uniform(0.3, 1.0)
        sigma = rng.uniform(1.0, 3.0)
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        star = brightness * np.exp(-r2 / (2 * sigma ** 2))
        img += star.astype(np.float32)

    img = np.clip(img, 0.0, 1.0)
    return img


# ---------------------------------------------------------------------------
# Noise generation
# ---------------------------------------------------------------------------

def generate_noisy_frames(reference, n_frames, sigma, seed=100):
    """Generate n_frames noisy copies of reference, each with gaussian noise."""
    rng = np.random.RandomState(seed)
    frames = []
    for _ in range(n_frames):
        noise = rng.normal(0, sigma, reference.shape).astype(np.float32)
        noisy = np.clip(reference + noise, 0.0, 1.0)
        frames.append(noisy)
    return frames


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path):
    """Load image as grayscale float32 [0, 1]."""
    try:
        from PIL import Image
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.float32) / 255.0
    except ImportError:
        pass
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        return img.astype(np.float32) / 255.0
    except ImportError:
        print("Error: need Pillow or OpenCV to load images")
        print("  pip install Pillow  OR  pip install opencv-python")
        sys.exit(1)


def save_image(data, path):
    """Save float32 array as grayscale PNG."""
    clamped = np.clip(data, 0, None)
    if clamped.max() > 0:
        clamped = clamped / clamped.max()
    uint8 = (clamped * 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(uint8, mode="L").save(str(path))
        return
    except ImportError:
        pass
    try:
        import cv2
        cv2.imwrite(str(path), uint8)
        return
    except ImportError:
        pass
    print(f"  Warning: could not save {path} (need Pillow or OpenCV)")


# ---------------------------------------------------------------------------
# Build Ea kernel
# ---------------------------------------------------------------------------

def build_ea_kernel():
    """Compile stack.ea to stack.so if needed."""
    so_path = DEMO_DIR / "stack.so"
    ea_path = DEMO_DIR / "stack.ea"

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

    # Move from project root to demo dir
    built = EA_ROOT / "stack.so"
    if built.exists():
        built.rename(so_path)

    print(f"  Built: {so_path}")
    return so_path


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

def stack_numpy(frames):
    """Stack frames by averaging. Uses O(N*pixels) memory."""
    return np.mean(np.array(frames), axis=0)


def stack_ea(frames, so_path):
    """Stack frames using Ea SIMD kernels. Uses O(pixels) extra memory."""
    lib = ctypes.CDLL(str(so_path))
    lib.accumulate_f32x8.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int32]
    lib.accumulate_f32x8.restype = None
    lib.scale_f32x8.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int32, ctypes.c_float]
    lib.scale_f32x8.restype = None

    h, w = frames[0].shape
    n = h * w
    acc = np.zeros(n, dtype=np.float32)

    for frame in frames:
        flat = np.ascontiguousarray(frame, dtype=np.float32).ravel()
        lib.accumulate_f32x8(
            acc.ctypes.data_as(FLOAT_PTR),
            flat.ctypes.data_as(FLOAT_PTR),
            n,
        )

    out = np.zeros(n, dtype=np.float32)
    lib.scale_f32x8(
        acc.ctypes.data_as(FLOAT_PTR),
        out.ctypes.data_as(FLOAT_PTR),
        n,
        ctypes.c_float(1.0 / len(frames)),
    )
    return out.reshape(h, w)


# ---------------------------------------------------------------------------
# SNR measurement
# ---------------------------------------------------------------------------

def compute_snr(image, reference):
    """Compute signal-to-noise ratio in dB."""
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean((image - reference) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


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
    # Parse args
    n_frames = N_FRAMES
    if len(sys.argv) > 1:
        n_frames = int(sys.argv[1])

    print(f"Astronomy Frame Stacking: {n_frames} frames, 1024x1024")
    print()

    # Generate starfield and noisy frames
    print("Generating starfield reference...")
    reference = generate_starfield()
    h, w = reference.shape
    print(f"  Image: {w}x{h} ({w*h:,} pixels)")

    print(f"Generating {n_frames} noisy frames (sigma={NOISE_SIGMA})...")
    frames = generate_noisy_frames(reference, n_frames, NOISE_SIGMA)
    print()

    # Build Ea kernel
    so_path = build_ea_kernel()
    print()

    # --- Correctness ---
    print("=== Correctness ===")
    result_numpy = stack_numpy(frames)
    result_ea = stack_ea(frames, so_path)

    diff = np.abs(result_ea - result_numpy)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"  Ea vs NumPy: max diff = {max_diff:.8f}, mean diff = {mean_diff:.8f}")
    if max_diff < 1e-5:
        print("  Match: YES (within floating-point tolerance)")
    else:
        print(f"  Match: APPROXIMATE (max diff {max_diff:.6f})")
    print()

    # --- SNR Analysis ---
    print("=== SNR Analysis ===")
    snr_single = compute_snr(frames[0], reference)
    snr_numpy = compute_snr(result_numpy, reference)
    snr_ea = compute_snr(result_ea, reference)
    expected_improvement_db = 10 * np.log10(n_frames) / 2

    print(f"  Single noisy frame SNR : {snr_single:6.2f} dB")
    print(f"  Stacked (NumPy) SNR    : {snr_numpy:6.2f} dB")
    print(f"  Stacked (Ea) SNR       : {snr_ea:6.2f} dB")
    print(f"  Improvement (Ea)       : {snr_ea - snr_single:+.2f} dB")
    print(f"  Expected improvement   : ~{expected_improvement_db:+.2f} dB "
          f"(sqrt({n_frames}) = {np.sqrt(n_frames):.1f}x noise reduction)")
    print()

    # Save output images
    save_image(frames[0], DEMO_DIR / "output_single.png")
    save_image(result_ea, DEMO_DIR / "output_stacked.png")
    print("  Saved: output_single.png (one noisy frame)")
    print("  Saved: output_stacked.png (Ea stacked result)")
    print()

    # --- Performance ---
    print("=== Performance ===")
    print(f"  {n_frames} frames, {w}x{h} image, 50 runs, median time\n")

    t_numpy = benchmark(stack_numpy, frames)
    print(f"  NumPy               : {t_numpy:8.2f} ms")

    t_ea = benchmark(stack_ea, frames, so_path)
    print(f"  Ea (stack.so)       : {t_ea:8.2f} ms")
    print()

    # Memory note
    pixels = w * h
    print("=== Memory Usage ===")
    print(f"  Ea    : O(pixels) extra = {pixels * 4 / 1024:.0f} KB "
          f"(single accumulator)")
    print(f"  NumPy : O(N*pixels)     = {n_frames * pixels * 4 / 1024:.0f} KB "
          f"(np.array stacks all frames)")
    print()

    # --- Summary ---
    print("=== Summary ===")
    speedup = t_numpy / t_ea
    print(f"  Ea vs NumPy  : {speedup:.1f}x "
          f"{'faster' if t_ea < t_numpy else 'slower'}")
    print(f"  SNR gain     : {snr_ea - snr_single:+.2f} dB from stacking {n_frames} frames")
    print(f"  Memory       : Ea uses {n_frames}x less memory than NumPy")
    print()
    print("Output images saved to demo/astro_stack/")


if __name__ == "__main__":
    main()
