#!/usr/bin/env python3
"""
scikit-image Pipeline Fusion Demo: Ea vs NumPy

Replaces a multi-stage image processing pipeline with one fused Ea kernel.
Compares three implementations:
  - NumPy baseline (4 separate array operations)
  - Ea unfused (4 separate kernel calls)
  - Ea fused (1 kernel: blur+sobel+threshold)
  + dilation (shared by all, separate kernel)

Usage:
    python run.py [image_path]

If no image is given, downloads a Kodak benchmark image (or generates synthetic).
"""

import sys
import time
import ctypes
import subprocess
import tracemalloc
import urllib.request
from pathlib import Path
import numpy as np

DEMO_DIR = Path(__file__).parent
EA_ROOT = DEMO_DIR / ".." / ".."

KODAK_URL = "https://r0k.us/graphics/kodak/kodak/kodim23.png"
THRESHOLD = 0.1

FLOAT_PTR = ctypes.POINTER(ctypes.c_float)


# ---------------------------------------------------------------------------
# Image loading (same pattern as demo/sobel/run.py)
# ---------------------------------------------------------------------------

def download_test_image():
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
    for y_coord in range(height):
        img[y_coord, :] = y_coord / height * 0.3
    img[100:300, 200:500] = 0.8
    img[150:250, 250:450] = 0.2
    img[400:700, 800:1200] = 0.7
    img[500:600, 900:1100] = 0.3
    yy, xx = np.ogrid[:height, :width]
    for cx, cy, r, val in [(960, 540, 200, 0.9), (960, 540, 100, 0.1),
                           (400, 800, 150, 0.6), (1500, 300, 120, 0.75)]:
        mask = (xx - cx)**2 + (yy - cy)**2 < r**2
        img[mask] = val
    return img


def save_image(data, path):
    """Save float32 array as grayscale PNG."""
    clamped = np.clip(data, 0, None)
    if clamped.max() > 0:
        clamped = clamped / clamped.max()
    uint8 = (clamped * 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(uint8, mode="L").save(str(path))
        return True
    except ImportError:
        pass
    try:
        import cv2
        cv2.imwrite(str(path), uint8)
        return True
    except ImportError:
        pass
    print(f"  Warning: could not save {path} (need Pillow or OpenCV)")
    return False


# ---------------------------------------------------------------------------
# NumPy baseline — manual implementations matching Ea kernels exactly
# ---------------------------------------------------------------------------

def gaussian_blur_numpy(img):
    """3x3 Gaussian blur: [1,2,1; 2,4,2; 1,2,1] / 16."""
    out = np.zeros_like(img)
    out[1:-1, 1:-1] = (
        img[:-2, :-2] + 2*img[:-2, 1:-1] + img[:-2, 2:]
        + 2*img[1:-1, :-2] + 4*img[1:-1, 1:-1] + 2*img[1:-1, 2:]
        + img[2:, :-2] + 2*img[2:, 1:-1] + img[2:, 2:]
    ) / 16.0
    return out


def sobel_numpy(img):
    """Sobel gradient magnitude: |Gx| + |Gy| (L1 norm)."""
    gx = (img[:-2, 2:] - img[:-2, :-2]
          + 2.0 * (img[1:-1, 2:] - img[1:-1, :-2])
          + img[2:, 2:] - img[2:, :-2])
    gy = (img[2:, :-2] - img[:-2, :-2]
          + 2.0 * (img[2:, 1:-1] - img[:-2, 1:-1])
          + img[2:, 2:] - img[:-2, 2:])
    mag = np.abs(gx) + np.abs(gy)
    out = np.zeros_like(img)
    out[1:-1, 1:-1] = mag
    return out


def threshold_numpy(img, thresh):
    """Binary threshold."""
    return (img > thresh).astype(np.float32)


def dilate_numpy(img):
    """3x3 binary dilation (max of neighborhood)."""
    out = np.zeros_like(img)
    out[1:-1, 1:-1] = np.maximum.reduce([
        img[:-2, :-2], img[:-2, 1:-1], img[:-2, 2:],
        img[1:-1, :-2], img[1:-1, 1:-1], img[1:-1, 2:],
        img[2:, :-2], img[2:, 1:-1], img[2:, 2:],
    ])
    return out


def pipeline_numpy(img):
    """Full NumPy pipeline: blur -> sobel -> threshold -> dilate."""
    blurred = gaussian_blur_numpy(img)
    edges = sobel_numpy(blurred)
    mask = threshold_numpy(edges, THRESHOLD)
    dilated = dilate_numpy(mask)
    return dilated


# ---------------------------------------------------------------------------
# Build Ea kernels
# ---------------------------------------------------------------------------

def build_ea_kernel(ea_name):
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
# Ea ctypes wrappers
# ---------------------------------------------------------------------------

def load_unfused_lib(so_path):
    lib = ctypes.CDLL(str(so_path))

    lib.gaussian_blur_3x3.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int32, ctypes.c_int32]
    lib.gaussian_blur_3x3.restype = None

    lib.sobel_magnitude.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int32, ctypes.c_int32]
    lib.sobel_magnitude.restype = None

    lib.threshold_f32x8.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int32, ctypes.c_float]
    lib.threshold_f32x8.restype = None

    return lib


def load_fused_lib(so_path):
    lib = ctypes.CDLL(str(so_path))

    lib.edge_detect_fused.argtypes = [
        FLOAT_PTR, FLOAT_PTR,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_float
    ]
    lib.edge_detect_fused.restype = None

    return lib


def load_dilation_lib(so_path):
    lib = ctypes.CDLL(str(so_path))

    lib.dilate_3x3.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int32, ctypes.c_int32]
    lib.dilate_3x3.restype = None

    return lib


def ea_dilate(img, lib_dil, width, height):
    """Run dilation kernel. Shared by both unfused and fused paths."""
    inp = np.ascontiguousarray(img, dtype=np.float32)
    result = np.zeros_like(inp)
    lib_dil.dilate_3x3(
        inp.ctypes.data_as(FLOAT_PTR),
        result.ctypes.data_as(FLOAT_PTR),
        ctypes.c_int32(width),
        ctypes.c_int32(height),
    )
    return result


def pipeline_ea_unfused(img, lib_unfused, lib_dil, width, height):
    """Ea unfused: 4 separate kernel calls."""
    inp = np.ascontiguousarray(img, dtype=np.float32)
    blurred = np.zeros_like(inp)
    edges = np.zeros_like(inp)
    mask = np.zeros_like(inp)
    n = width * height

    lib_unfused.gaussian_blur_3x3(
        inp.ctypes.data_as(FLOAT_PTR),
        blurred.ctypes.data_as(FLOAT_PTR),
        ctypes.c_int32(width), ctypes.c_int32(height),
    )
    lib_unfused.sobel_magnitude(
        blurred.ctypes.data_as(FLOAT_PTR),
        edges.ctypes.data_as(FLOAT_PTR),
        ctypes.c_int32(width), ctypes.c_int32(height),
    )
    lib_unfused.threshold_f32x8(
        edges.ctypes.data_as(FLOAT_PTR),
        mask.ctypes.data_as(FLOAT_PTR),
        ctypes.c_int32(n), ctypes.c_float(THRESHOLD),
    )
    dilated = ea_dilate(mask, lib_dil, width, height)
    return dilated


def pipeline_ea_fused(img, lib_fused, lib_dil, width, height):
    """Ea fused: 1 kernel call (blur+sobel+threshold) + dilation."""
    inp = np.ascontiguousarray(img, dtype=np.float32)
    mask = np.zeros_like(inp)

    lib_fused.edge_detect_fused(
        inp.ctypes.data_as(FLOAT_PTR),
        mask.ctypes.data_as(FLOAT_PTR),
        ctypes.c_int32(width), ctypes.c_int32(height),
        ctypes.c_float(THRESHOLD),
    )
    dilated = ea_dilate(mask, lib_dil, width, height)
    return dilated


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
# Visual output
# ---------------------------------------------------------------------------

def create_comparison_image(img_input, img_numpy, img_ea_fused, diff):
    """Create side-by-side comparison: input | numpy | ea fused."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("  Pillow not installed, skipping comparison image")
        return

    def to_uint8(arr):
        c = np.clip(arr, 0, None)
        if c.max() > 0:
            c = c / c.max()
        return (c * 255).astype(np.uint8)

    h, w = img_input.shape
    gap = 4
    canvas_w = w * 3 + gap * 2
    canvas_h = h + 30

    canvas = Image.new("L", (canvas_w, canvas_h), 0)

    canvas.paste(Image.fromarray(to_uint8(img_input), "L"), (0, 30))
    canvas.paste(Image.fromarray(to_uint8(img_numpy), "L"), (w + gap, 30))
    canvas.paste(Image.fromarray(to_uint8(img_ea_fused), "L"), (2 * (w + gap), 30))

    draw = ImageDraw.Draw(canvas)
    draw.text((w // 2 - 20, 5), "Input", fill=200)
    draw.text((w + gap + w // 2 - 20, 5), "NumPy", fill=200)
    draw.text((2 * (w + gap) + w // 2 - 30, 5), "Ea Fused", fill=200)

    canvas.save(str(DEMO_DIR / "compare.png"))
    print(f"  Saved: compare.png")

    diff_scaled = to_uint8(diff * 1000)
    Image.fromarray(diff_scaled, "L").save(str(DEMO_DIR / "diff_heatmap.png"))
    print(f"  Saved: diff_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
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

    so_unfused = build_ea_kernel("pipeline_unfused")
    so_fused = build_ea_kernel("pipeline_fused")
    so_dil = build_ea_kernel("dilation")

    lib_unfused = load_unfused_lib(so_unfused)
    lib_fused = load_fused_lib(so_fused)
    lib_dil = load_dilation_lib(so_dil)

    # --- Correctness ---
    print("=== Correctness ===")
    result_numpy = pipeline_numpy(img)
    result_unfused = pipeline_ea_unfused(img, lib_unfused, lib_dil, w, h)
    result_fused = pipeline_ea_fused(img, lib_fused, lib_dil, w, h)

    border = 3
    interior = np.s_[border:-border, border:-border]

    diff_unfused = np.abs(result_unfused[interior] - result_numpy[interior])
    diff_fused = np.abs(result_fused[interior] - result_numpy[interior])

    n_interior = diff_unfused.size
    pct_unfused = diff_unfused.sum() / n_interior * 100
    pct_fused = diff_fused.sum() / n_interior * 100

    print(f"  Ea unfused vs NumPy:  {pct_unfused:.3f}% pixels differ (binary threshold boundary)")
    print(f"  Ea fused vs NumPy:    {pct_fused:.3f}% pixels differ (binary threshold boundary)")

    if pct_unfused < 1.0 and pct_fused < 1.0:
        print("  Match: YES (< 1% pixel difference, FP rounding at threshold boundary)")
    else:
        print(f"  Match: APPROXIMATE (> 1% pixel difference)")

    # --- Memory ---
    print()
    print("=== Memory ===")

    tracemalloc.start()
    _ = pipeline_numpy(img)
    _, peak_numpy = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    _ = pipeline_ea_fused(img, lib_fused, lib_dil, w, h)
    _, peak_fused = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_numpy_mb = peak_numpy / (1024 * 1024)
    peak_fused_mb = peak_fused / (1024 * 1024)
    print(f"  NumPy peak:     {peak_numpy_mb:8.1f} MB  (4 intermediate arrays)")
    print(f"  Ea fused peak:  {peak_fused_mb:8.1f} MB  (output array only)")
    if peak_fused_mb > 0:
        print(f"  Reduction:      {peak_numpy_mb / peak_fused_mb:8.1f}x")

    # --- Save visual output ---
    print()
    print("=== Visual Output ===")
    save_image(img, DEMO_DIR / "input.png")
    save_image(result_numpy, DEMO_DIR / "output_numpy.png")
    save_image(result_unfused, DEMO_DIR / "output_ea_unfused.png")
    save_image(result_fused, DEMO_DIR / "output_ea_fused.png")

    full_diff = np.abs(result_fused.astype(np.float64) - result_numpy.astype(np.float64))
    create_comparison_image(img, result_numpy, result_fused, full_diff)

    # --- Performance ---
    print()
    print("=== Performance ===")
    print(f"  {w}x{h} image, 50 runs, median time\n")

    t_numpy, s_numpy = benchmark(pipeline_numpy, img)
    print(f"  NumPy (4 stages)     : {t_numpy:8.2f} ms  +/-{s_numpy:.2f}")

    t_unfused, s_unfused = benchmark(
        pipeline_ea_unfused, img, lib_unfused, lib_dil, w, h)
    print(f"  Ea unfused (4 calls) : {t_unfused:8.2f} ms  +/-{s_unfused:.2f}")

    t_fused, s_fused = benchmark(
        pipeline_ea_fused, img, lib_fused, lib_dil, w, h)
    print(f"  Ea fused (2 calls)   : {t_fused:8.2f} ms  +/-{s_fused:.2f}")

    # --- Summary ---
    print()
    print("=== Summary ===")
    print(f"  Ea unfused vs NumPy  : {t_numpy / t_unfused:.1f}x "
          f"{'faster' if t_unfused < t_numpy else 'slower'}")
    print(f"  Ea fused vs NumPy    : {t_numpy / t_fused:.1f}x "
          f"{'faster' if t_fused < t_numpy else 'slower'}")
    print(f"  Fusion speedup       : {t_unfused / t_fused:.1f}x "
          f"(4 calls -> 2 calls)")
    print(f"  Memory passes removed: 2 (blur + sobel intermediates eliminated)")

    print()
    print("Output images saved to demo/skimage_fusion/")

    # --- Scaling benchmark ---
    print()
    print("=== Scaling: Fusion speedup vs image size ===")
    print(f"  {'Size':>12}  {'Pixels':>10}  {'Unfused':>10}  {'Fused':>10}  {'Fusion':>8}")
    print(f"  {'':>12}  {'':>10}  {'(ms)':>10}  {'(ms)':>10}  {'speedup':>8}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    for scale_w, scale_h in [(768, 512), (1920, 1080), (3840, 2160), (4096, 4096)]:
        scale_img = np.random.rand(scale_h, scale_w).astype(np.float32)
        t_u, _ = benchmark(pipeline_ea_unfused, scale_img, lib_unfused, lib_dil,
                           scale_w, scale_h, warmup=3, runs=20)
        t_f, _ = benchmark(pipeline_ea_fused, scale_img, lib_fused, lib_dil,
                           scale_w, scale_h, warmup=3, runs=20)
        speedup = t_u / t_f
        print(f"  {scale_w:>5}x{scale_h:<5}  {scale_w*scale_h:>10,}  {t_u:>10.2f}  {t_f:>10.2f}  {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
