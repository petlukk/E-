#!/usr/bin/env python3
"""
Video Frame Anomaly Detection Demo: Ea vs NumPy vs OpenCV

Loads two grayscale frames, computes per-pixel absolute difference,
thresholds anomalous pixels, counts them. Compares correctness and performance.

Usage:
    python run.py [frame_a_path frame_b_path]

If no images given, downloads OpenCV sample video and extracts frames (or uses synthetic).
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
THRESHOLD = 0.1

OPENCV_VIDEO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"

FLOAT_PTR = ctypes.POINTER(ctypes.c_float)

# ---------------------------------------------------------------------------
# Video download and frame extraction
# ---------------------------------------------------------------------------

def download_test_video():
    """Download OpenCV sample video and extract two consecutive frames."""
    video_path = DEMO_DIR / "vtest.avi"
    frame_a_path = DEMO_DIR / "frame_a.png"
    frame_b_path = DEMO_DIR / "frame_b.png"

    # If extracted frames already exist, skip
    if frame_a_path.exists() and frame_b_path.exists():
        return frame_a_path, frame_b_path

    # Download video if needed
    if not video_path.exists():
        print(f"Downloading OpenCV sample video...")
        print(f"  Source: {OPENCV_VIDEO_URL}")
        try:
            urllib.request.urlretrieve(OPENCV_VIDEO_URL, str(video_path))
            print(f"  Saved: {video_path}")
        except Exception as e:
            print(f"  Download failed: {e}")
            return None, None

    # Extract two frames with visible differences (frames 0 and 30 for motion)
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("  Cannot open video")
            return None, None

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame_a_bgr = cap.read()
        if not ret:
            return None, None

        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ret, frame_b_bgr = cap.read()
        cap.release()
        if not ret:
            return None, None

        # Convert to grayscale and save
        frame_a_gray = cv2.cvtColor(frame_a_bgr, cv2.COLOR_BGR2GRAY)
        frame_b_gray = cv2.cvtColor(frame_b_bgr, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(frame_a_path), frame_a_gray)
        cv2.imwrite(str(frame_b_path), frame_b_gray)
        print(f"  Extracted frames 0 and 30 ({frame_a_gray.shape[1]}x{frame_a_gray.shape[0]})")
        return frame_a_path, frame_b_path
    except ImportError:
        # Without OpenCV, try Pillow approach — but AVI needs cv2
        print("  Need OpenCV to extract video frames: pip install opencv-python")
        return None, None


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


def generate_test_frames(width=1920, height=1080):
    """Generate two synthetic frames with known differences."""
    rng = np.random.default_rng(42)

    # Frame A: gradient with geometric shapes
    frame_a = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        frame_a[y, :] = y / height * 0.3

    frame_a[100:300, 200:500] = 0.7
    frame_a[400:700, 800:1200] = 0.5
    frame_a += rng.normal(0, 0.01, frame_a.shape).astype(np.float32)
    frame_a = np.clip(frame_a, 0.0, 1.0)

    # Frame B: same base but with anomalous regions
    frame_b = frame_a.copy()
    # Anomaly 1: bright rectangle
    frame_b[150:250, 600:900] = 0.9
    # Anomaly 2: dark patch
    frame_b[500:650, 300:500] = 0.05
    # Anomaly 3: scattered noise in a region
    frame_b[700:900, 1000:1500] += rng.uniform(0.15, 0.4,
        (200, 500)).astype(np.float32)
    frame_b = np.clip(frame_b, 0.0, 1.0)

    return frame_a, frame_b


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
# Build
# ---------------------------------------------------------------------------

def build_ea_kernel(ea_name="anomaly"):
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

    # Move from project root to demo dir
    built = EA_ROOT / f"{ea_name}.so"
    if built.exists():
        built.rename(so_path)

    print(f"  Built: {so_path}")
    return so_path


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

def anomaly_numpy(frame_a, frame_b, threshold):
    """Pure NumPy anomaly detection."""
    diff = np.abs(frame_a - frame_b)
    mask = (diff > threshold).astype(np.float32)
    count = float(mask.sum())
    return diff, mask, count


def anomaly_opencv(frame_a, frame_b, threshold):
    """OpenCV anomaly detection."""
    import cv2
    diff = cv2.absdiff(frame_a, frame_b)
    _, mask = cv2.threshold(diff, threshold, 1.0, cv2.THRESH_BINARY)
    count = float(cv2.countNonZero(mask))
    return diff, mask, count


def anomaly_ea(frame_a, frame_b, threshold, so_path):
    """Run Ea anomaly kernels via ctypes."""
    lib = ctypes.CDLL(str(so_path))

    lib.frame_diff_f32x8.argtypes = [FLOAT_PTR, FLOAT_PTR, FLOAT_PTR,
                                     ctypes.c_int32]
    lib.frame_diff_f32x8.restype = None

    lib.threshold_f32x8.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int32,
                                    ctypes.c_float]
    lib.threshold_f32x8.restype = None

    lib.sum_f32x8.argtypes = [FLOAT_PTR, ctypes.c_int32]
    lib.sum_f32x8.restype = ctypes.c_float

    h, w = frame_a.shape
    flat_a = np.ascontiguousarray(frame_a, dtype=np.float32).ravel()
    flat_b = np.ascontiguousarray(frame_b, dtype=np.float32).ravel()
    diff = np.zeros_like(flat_a)
    mask = np.zeros_like(flat_a)
    n = len(flat_a)

    lib.frame_diff_f32x8(
        flat_a.ctypes.data_as(FLOAT_PTR),
        flat_b.ctypes.data_as(FLOAT_PTR),
        diff.ctypes.data_as(FLOAT_PTR),
        n,
    )
    lib.threshold_f32x8(
        diff.ctypes.data_as(FLOAT_PTR),
        mask.ctypes.data_as(FLOAT_PTR),
        n,
        ctypes.c_float(threshold),
    )
    count = lib.sum_f32x8(mask.ctypes.data_as(FLOAT_PTR), n)

    return diff.reshape(h, w), mask.reshape(h, w), float(count)


def anomaly_ea_fused(frame_a, frame_b, threshold, so_path):
    """Run fused Ea kernel: diff+threshold+count in one pass. No intermediate arrays."""
    lib = ctypes.CDLL(str(so_path))

    lib.anomaly_count_fused.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int32,
                                        ctypes.c_float]
    lib.anomaly_count_fused.restype = ctypes.c_float

    flat_a = np.ascontiguousarray(frame_a, dtype=np.float32).ravel()
    flat_b = np.ascontiguousarray(frame_b, dtype=np.float32).ravel()
    n = len(flat_a)

    count = lib.anomaly_count_fused(
        flat_a.ctypes.data_as(FLOAT_PTR),
        flat_b.ctypes.data_as(FLOAT_PTR),
        n,
        ctypes.c_float(threshold),
    )
    return float(count)


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
    # Load or generate frames
    if len(sys.argv) > 2:
        path_a, path_b = sys.argv[1], sys.argv[2]
        print(f"Loading: {path_a}, {path_b}")
        frame_a = load_image(path_a)
        frame_b = load_image(path_b)
    else:
        # Try to download real video data
        path_a, path_b = download_test_video()
        if path_a is not None:
            print(f"Using OpenCV sample video frames")
            frame_a = load_image(path_a)
            frame_b = load_image(path_b)
        elif (DEMO_DIR / "frame_a.png").exists():
            print("Using included test frames")
            frame_a = load_image(DEMO_DIR / "frame_a.png")
            frame_b = load_image(DEMO_DIR / "frame_b.png")
        else:
            print("Generating synthetic test frames (install opencv-python for real video)")
            frame_a, frame_b = generate_test_frames()
            save_image(frame_a, DEMO_DIR / "frame_a.png")
            save_image(frame_b, DEMO_DIR / "frame_b.png")

    h, w = frame_a.shape
    print(f"Frames: {w}x{h} ({w*h:,} pixels)\n")

    # Build Ea kernels
    so_path = build_ea_kernel("anomaly")
    so_fused_path = build_ea_kernel("anomaly_fused")

    # --- Correctness ---
    print("=== Correctness ===")
    diff_np, mask_np, count_np = anomaly_numpy(frame_a, frame_b, THRESHOLD)
    diff_ea, mask_ea, count_ea = anomaly_ea(frame_a, frame_b, THRESHOLD,
                                            so_path)
    count_fused = anomaly_ea_fused(frame_a, frame_b, THRESHOLD, so_fused_path)

    max_diff_diff = np.abs(diff_ea - diff_np).max()
    max_diff_mask = np.abs(mask_ea - mask_np).max()
    print(f"  Ea vs NumPy diff image : max diff = {max_diff_diff:.6f}")
    print(f"  Ea vs NumPy threshold  : max diff = {max_diff_mask:.6f}")
    print(f"  Anomaly count NumPy    : {count_np:.0f}")
    print(f"  Anomaly count Ea       : {count_ea:.0f}")
    print(f"  Anomaly count Ea fused : {count_fused:.0f}")
    if max_diff_diff < 0.001 and max_diff_mask < 0.001:
        print("  Match: YES (within floating-point tolerance)")
    else:
        print(f"  Match: APPROXIMATE (diff image {max_diff_diff:.4f}, "
              f"mask {max_diff_mask:.4f})")

    has_opencv = False
    try:
        import cv2
        has_opencv = True
        diff_cv, mask_cv, count_cv = anomaly_opencv(frame_a, frame_b,
                                                    THRESHOLD)
        print(f"  Anomaly count OpenCV   : {count_cv:.0f}")
    except ImportError:
        print("  OpenCV not installed, skipping (pip install opencv-python)")

    # Save outputs
    save_image(diff_ea, DEMO_DIR / "output_diff.png")
    save_image(mask_ea, DEMO_DIR / "output_thresh.png")
    print()

    # --- Performance ---
    print("=== Performance ===")
    print(f"  {w}x{h} frames, 50 runs, median time\n")

    t_numpy = benchmark(anomaly_numpy, frame_a, frame_b, THRESHOLD)
    print(f"  NumPy              : {t_numpy:8.2f} ms")

    t_ea = benchmark(anomaly_ea, frame_a, frame_b, THRESHOLD, so_path)
    print(f"  Ea (3 kernels)     : {t_ea:8.2f} ms")

    t_fused = benchmark(anomaly_ea_fused, frame_a, frame_b, THRESHOLD,
                        so_fused_path)
    print(f"  Ea fused (1 kernel): {t_fused:8.2f} ms")

    if has_opencv:
        t_cv = benchmark(anomaly_opencv, frame_a, frame_b, THRESHOLD)
        print(f"  OpenCV             : {t_cv:8.2f} ms")

    print()
    print("=== Summary ===")
    print(f"  Ea 3-kernel vs NumPy  : {t_numpy / t_ea:.1f}x "
          f"{'faster' if t_ea < t_numpy else 'slower'}")
    print(f"  Ea fused vs NumPy     : {t_numpy / t_fused:.1f}x "
          f"{'faster' if t_fused < t_numpy else 'slower'}")
    if has_opencv:
        print(f"  Ea fused vs OpenCV    : {t_cv / t_fused:.1f}x "
              f"{'faster' if t_fused < t_cv else 'slower'}")
    print(f"  Fusion speedup        : {t_ea / t_fused:.1f}x "
          f"(3 kernels -> 1 kernel)")

    print()
    print("Output images saved to demo/video_anomaly/")


if __name__ == "__main__":
    main()
