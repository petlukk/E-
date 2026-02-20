#!/usr/bin/env python3
"""
Astronomy Frame Stacking Demo: Ea vs NumPy

Stacks N noisy exposures of a starfield to reduce noise.
Signal reinforces, noise cancels by sqrt(N).

Usage:
    python run.py [N_frames]

Default: 16 frames from NASA SkyView (or synthetic if unavailable).
"""

import sys
import time
import ctypes
import subprocess
import urllib.request
from pathlib import Path
import numpy as np

DEMO_DIR = Path(__file__).parent
EA_ROOT = DEMO_DIR / ".." / ".."

N_FRAMES = 16
NOISE_SIGMA = 0.05

FLOAT_PTR = ctypes.POINTER(ctypes.c_float)

NASA_SKYVIEW_URL = "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"


# ---------------------------------------------------------------------------
# NASA SkyView download
# ---------------------------------------------------------------------------

def download_nasa_frames(n_frames=16):
    """Download real telescope data from NASA SkyView.

    Downloads multiple survey images of the same sky region (M31 / Andromeda).
    Different surveys have different noise characteristics, simulating
    multiple exposures.
    """
    data_dir = DEMO_DIR / "nasa_data"
    data_dir.mkdir(exist_ok=True)

    # Check if we already have frames
    existing = sorted(data_dir.glob("frame_*.npy"))
    if len(existing) >= n_frames:
        print(f"  Using cached NASA data ({len(existing)} frames)")
        return [np.load(str(f)) for f in existing[:n_frames]]

    # Download a DSS image of M31 (Andromeda galaxy)
    print("Downloading NASA SkyView data (M31 / Andromeda)...")
    print(f"  Source: {NASA_SKYVIEW_URL}")

    try:
        # Download FITS file
        params = "Survey=DSS&Position=M31&Size=0.5&Pixels=1024&Return=FITS"
        url = f"{NASA_SKYVIEW_URL}?{params}"
        fits_path = data_dir / "m31.fits"

        if not fits_path.exists():
            urllib.request.urlretrieve(url, str(fits_path))
            print(f"  Downloaded: {fits_path}")

        # Try to read FITS
        try:
            from astropy.io import fits as pyfits
            with pyfits.open(str(fits_path)) as hdul:
                img = hdul[0].data.astype(np.float32)
        except ImportError:
            # Fallback: read FITS manually (simple 2D case)
            img = _read_simple_fits(fits_path)

        if img is None:
            return None

        # Normalize to [0, 1]
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()

        # Make square 1024x1024 (crop or pad)
        h, w = img.shape
        size = min(h, w, 1024)
        cy, cx = h // 2, w // 2
        img = img[cy - size//2:cy + size//2, cx - size//2:cx + size//2]

        # Generate N "exposures" by adding realistic noise to the real image
        # This simulates multiple telescope exposures of the same field
        rng = np.random.RandomState(100)
        frames = []
        for i in range(n_frames):
            noise = rng.normal(0, NOISE_SIGMA, img.shape).astype(np.float32)
            frame = np.clip(img + noise, 0.0, 1.0)
            frame_path = data_dir / f"frame_{i:03d}.npy"
            np.save(str(frame_path), frame)
            frames.append(frame)

        # Save reference
        np.save(str(data_dir / "reference.npy"), img)
        print(f"  Generated {n_frames} noisy exposures from real telescope data")
        return frames

    except Exception as e:
        print(f"  Download failed: {e}")
        print(f"  Falling back to synthetic starfield")
        return None


def _read_simple_fits(path):
    """Minimal FITS reader for simple 2D images. No dependencies."""
    try:
        with open(str(path), 'rb') as f:
            # Read primary header
            header = {}
            while True:
                block = f.read(2880)
                if not block:
                    return None
                text = block.decode('ascii', errors='replace')
                for i in range(0, len(text), 80):
                    card = text[i:i+80]
                    if card.startswith('END'):
                        break
                    if len(card) > 8 and card[8] == '=':
                        key = card[:8].strip()
                        val = card[10:30].strip().strip("'").strip()
                        header[key] = val
                if 'END' in text:
                    break

            naxis = int(header.get('NAXIS', 0))
            if naxis != 2:
                return None

            naxis1 = int(header['NAXIS1'])
            naxis2 = int(header['NAXIS2'])
            bitpix = int(header['BITPIX'])

            if bitpix == -32:
                dtype = np.float32
            elif bitpix == -64:
                dtype = np.float64
            elif bitpix == 16:
                dtype = np.int16
            elif bitpix == 32:
                dtype = np.int32
            else:
                return None

            # FITS stores data in big-endian byte order
            be_dtype = np.dtype(dtype).newbyteorder('>')
            data = np.frombuffer(
                f.read(naxis1 * naxis2 * abs(bitpix) // 8), dtype=be_dtype
            )
            data = data.astype(dtype)
            return data.reshape(naxis2, naxis1).astype(np.float32)
    except Exception:
        return None


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
    median = times[len(times) // 2]
    return median, float(np.std(times))


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

    # Try real NASA data first, fall back to synthetic
    frames = download_nasa_frames(n_frames)
    if frames is not None:
        reference_path = DEMO_DIR / "nasa_data" / "reference.npy"
        if reference_path.exists():
            reference = np.load(str(reference_path))
        else:
            reference = generate_starfield()
        data_source = "NASA SkyView (M31 / Andromeda)"
    else:
        print("Generating starfield reference...")
        reference = generate_starfield()
        print(f"Generating {n_frames} noisy frames (sigma={NOISE_SIGMA})...")
        frames = generate_noisy_frames(reference, n_frames, NOISE_SIGMA)
        data_source = "synthetic starfield"

    h, w = reference.shape
    print(f"  Image: {w}x{h} ({w*h:,} pixels)")
    print(f"  Data source: {data_source}")
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
    # Power SNR improvement from averaging N frames: noise power drops by N,
    # so SNR improves by 10*log10(N). (The /2 form is for amplitude SNR only.)
    expected_improvement_db = 10 * np.log10(n_frames)

    print(f"  Single noisy frame SNR : {snr_single:6.2f} dB")
    print(f"  Stacked (NumPy) SNR    : {snr_numpy:6.2f} dB")
    print(f"  Stacked (Ea) SNR       : {snr_ea:6.2f} dB")
    print(f"  Improvement (Ea)       : {snr_ea - snr_single:+.2f} dB")
    print(f"  Expected improvement   : ~{expected_improvement_db:+.2f} dB "
          f"(noise power / {n_frames} → power SNR + 10·log₁₀({n_frames}))")
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

    t_numpy, s_numpy = benchmark(stack_numpy, frames)
    print(f"  NumPy               : {t_numpy:8.2f} ms  ±{s_numpy:.2f}")

    t_ea, s_ea = benchmark(stack_ea, frames, so_path)
    print(f"  Ea (stack.so)       : {t_ea:8.2f} ms  ±{s_ea:.2f}")
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
