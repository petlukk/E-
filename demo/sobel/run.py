#!/usr/bin/env python3
"""
Sobel Edge Detection Demo: Eä vs NumPy vs OpenCV

Loads a grayscale image, runs Sobel edge detection three ways,
compares correctness and performance.

Usage:
    python run.py [image_path]

If no image is given, downloads a Kodak benchmark image (or generates synthetic).
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

KODAK_URL = "https://r0k.us/graphics/kodak/kodak/kodim23.png"


def download_test_image():
    """Download a Kodak benchmark image if no input image exists."""
    dest = DEMO_DIR / "input.png"
    if dest.exists():
        return dest
    print(f"Downloading Kodak benchmark image...")
    print(f"  Source: {KODAK_URL}")
    try:
        urllib.request.urlretrieve(KODAK_URL, str(dest))
        print(f"  Saved: {dest}")
        return dest
    except Exception as e:
        print(f"  Download failed: {e}")
        print(f"  Falling back to synthetic image")
        return None


# ---------------------------------------------------------------------------
# Image loading
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


def generate_test_image(width=1920, height=1080):
    """Generate a synthetic test image with geometric shapes."""
    img = np.zeros((height, width), dtype=np.float32)

    # Gradient background
    for y in range(height):
        img[y, :] = y / height * 0.3

    # Rectangles
    img[100:300, 200:500] = 0.8
    img[150:250, 250:450] = 0.2
    img[400:700, 800:1200] = 0.7
    img[500:600, 900:1100] = 0.3

    # Circles
    yy, xx = np.ogrid[:height, :width]
    for cx, cy, r, val in [(960, 540, 200, 0.9), (960, 540, 100, 0.1),
                           (400, 800, 150, 0.6), (1500, 300, 120, 0.75)]:
        mask = (xx - cx)**2 + (yy - cy)**2 < r**2
        img[mask] = val

    # Diagonal lines
    for offset in range(0, width, 80):
        for t in range(min(height, width - offset)):
            if t + offset < width:
                img[min(t, height-1), t + offset] = 1.0

    return img


def save_image(data, path):
    """Save float32 array as grayscale PNG."""
    # Clamp and convert
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
# Implementations
# ---------------------------------------------------------------------------

def sobel_numpy(img):
    """Pure NumPy Sobel using array slicing. No scipy, no OpenCV."""
    # Gx = [-1 0 1; -2 0 2; -1 0 1]
    gx = (img[:-2, 2:] - img[:-2, :-2]
          + 2.0 * (img[1:-1, 2:] - img[1:-1, :-2])
          + img[2:, 2:] - img[2:, :-2])

    # Gy = [-1 -2 -1; 0 0 0; 1 2 1]
    gy = (img[2:, :-2] - img[:-2, :-2]
          + 2.0 * (img[2:, 1:-1] - img[:-2, 1:-1])
          + img[2:, 2:] - img[:-2, 2:])

    mag = np.abs(gx) + np.abs(gy)

    # Pad back to original size (border = 0)
    out = np.zeros_like(img)
    out[1:-1, 1:-1] = mag
    return out


def sobel_opencv(img):
    """OpenCV Sobel. L1 norm for fair comparison."""
    import cv2
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(gx) + np.abs(gy)


def build_ea_kernel():
    """Compile sobel.ea to sobel.so if needed."""
    so_path = DEMO_DIR / "sobel.so"
    ea_path = DEMO_DIR / "sobel.ea"

    if so_path.exists() and so_path.stat().st_mtime > ea_path.stat().st_mtime:
        return so_path

    print("Building Eä kernel...")
    result = subprocess.run(
        ["cargo", "run", "--features=llvm", "--release", "--",
         str(ea_path), "--lib"],
        capture_output=True, text=True, cwd=str(EA_ROOT),
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)

    # Move from project root to demo dir
    built = EA_ROOT / "sobel.so"
    if built.exists():
        built.rename(so_path)

    print(f"  Built: {so_path}")
    return so_path


def sobel_ea(img, so_path):
    """Run Eä Sobel kernel via ctypes."""
    lib = ctypes.CDLL(str(so_path))
    lib.sobel.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int32,                  # width
        ctypes.c_int32,                  # height
    ]
    lib.sobel.restype = None

    height, width = img.shape
    flat_in = np.ascontiguousarray(img, dtype=np.float32)
    flat_out = np.zeros_like(flat_in)

    lib.sobel(
        flat_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        flat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(width),
        ctypes.c_int32(height),
    )
    return flat_out


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark(func, *args, warmup=5, runs=50):
    """Run function multiple times, return median time in ms."""
    # Warmup
    for _ in range(warmup):
        func(*args)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    return times[len(times) // 2]  # median


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load or generate image
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"Loading: {img_path}")
        img = load_image(img_path)
    else:
        img_path = download_test_image()
        if img_path is not None:
            print(f"Using Kodak benchmark image: {img_path.name}")
            img = load_image(img_path)
        else:
            print("Generating 1920x1080 synthetic test image")
            img = generate_test_image()
            save_image(img, DEMO_DIR / "input.png")

    h, w = img.shape
    print(f"Image: {w}x{h} ({w*h:,} pixels)\n")

    # Build Eä kernel
    so_path = build_ea_kernel()

    # --- Correctness ---
    print("=== Correctness ===")
    result_numpy = sobel_numpy(img)
    result_ea = sobel_ea(img, so_path)

    # Compare Eä vs NumPy (both use same Sobel kernel definition)
    diff = np.abs(result_ea[1:-1, 1:-1] - result_numpy[1:-1, 1:-1])
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"  Eä vs NumPy: max diff = {max_diff:.6f}, mean diff = {mean_diff:.6f}")
    if max_diff < 0.001:
        print("  Match: YES (within floating-point tolerance)")
    else:
        print(f"  Match: APPROXIMATE (max diff {max_diff:.4f})")

    has_opencv = False
    try:
        import cv2
        has_opencv = True
        result_cv = sobel_opencv(img)
        # OpenCV Sobel uses different scaling, normalize for comparison
        # OpenCV ksize=3 applies a 1/1 scale by default, but the kernel
        # values differ slightly. Compare pattern, not absolute values.
        print(f"  OpenCV output range: [{result_cv.min():.2f}, {result_cv.max():.2f}]")
        print(f"  Eä output range:     [{result_ea.min():.2f}, {result_ea.max():.2f}]")
    except ImportError:
        print("  OpenCV not installed, skipping (pip install opencv-python)")

    # Save outputs
    save_image(result_ea, DEMO_DIR / "output_ea.png")
    save_image(result_numpy, DEMO_DIR / "output_numpy.png")
    if has_opencv:
        save_image(result_cv, DEMO_DIR / "output_opencv.png")
    print()

    # --- Performance ---
    print("=== Performance ===")
    print(f"  {w}x{h} image, 50 runs, median time\n")

    t_numpy = benchmark(sobel_numpy, img)
    print(f"  NumPy          : {t_numpy:8.2f} ms")

    t_ea = benchmark(sobel_ea, img, so_path)
    print(f"  Eä (sobel.so)  : {t_ea:8.2f} ms")

    if has_opencv:
        t_cv = benchmark(sobel_opencv, img)
        print(f"  OpenCV         : {t_cv:8.2f} ms")

    print()
    print("=== Summary ===")
    print(f"  Eä vs NumPy  : {t_numpy / t_ea:.1f}x {'faster' if t_ea < t_numpy else 'slower'}")
    if has_opencv:
        print(f"  Eä vs OpenCV : {t_cv / t_ea:.1f}x {'faster' if t_ea < t_cv else 'slower'}")
        print(f"  Note: OpenCV uses optimized C++ with possible multithreading")

    print()
    print("Output images saved to demo/sobel/")


if __name__ == "__main__":
    main()
