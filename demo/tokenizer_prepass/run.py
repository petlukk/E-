#!/usr/bin/env python3
"""
Tokenizer Prepass Demo: Ea fused vs Ea unfused vs NumPy

SIMD structural text scanning — classify bytes, lowercase, detect token
boundaries. NOT a full tokenizer; a pre-tokenization acceleration layer.
Same strategy as simdjson: SIMD structural scan, then normal processing.

Usage:
    python run.py
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

U8_PTR = ctypes.POINTER(ctypes.c_uint8)

# Gutenberg text URL (plain text, public domain)
TEXT_URL = "https://www.gutenberg.org/files/1342/1342-0.txt"  # Pride and Prejudice


# ---------------------------------------------------------------------------
# Text input
# ---------------------------------------------------------------------------

def get_test_text():
    """Download real text or generate synthetic fallback."""
    text_path = DEMO_DIR / "test_text.txt"
    if text_path.exists():
        text = text_path.read_bytes()
        if len(text) > 0:
            print(f"Using cached text: {text_path.name} ({len(text):,} bytes)")
            return text

    print(f"Downloading test text from Project Gutenberg...")
    try:
        urllib.request.urlretrieve(TEXT_URL, str(text_path))
        text = text_path.read_bytes()
        print(f"  Downloaded: {len(text):,} bytes")
        return text
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Using synthetic text")

    # Synthetic fallback: mixed letters, digits, punctuation, whitespace
    base = b"Hello, World! 42 tokens here.\nAnother LINE with CAPS and 123 digits.\t"
    text = base * (1_000_000 // len(base) + 1)
    text_path.write_bytes(text)
    print(f"  Generated: {len(text):,} bytes")
    return text


# ---------------------------------------------------------------------------
# NumPy baseline
# ---------------------------------------------------------------------------

def classify_numpy(text_bytes):
    b = np.frombuffer(text_bytes, dtype=np.uint8)
    flags = np.zeros_like(b)
    # Whitespace
    flags |= np.isin(b, [0x20, 0x09, 0x0A, 0x0D]).astype(np.uint8) * 0x01
    # Letter
    upper = (b >= 0x41) & (b <= 0x5A)
    lower = (b >= 0x61) & (b <= 0x7A)
    flags |= (upper | lower).astype(np.uint8) * 0x02
    # Digit
    flags |= ((b >= 0x30) & (b <= 0x39)).astype(np.uint8) * 0x04
    # Non-ASCII
    flags |= (b > 0x7F).astype(np.uint8) * 0x10
    # Punctuation: printable non-classified
    printable = (b >= 0x21) & (b <= 0x7E)
    known = np.isin(b, [0x20, 0x09, 0x0A, 0x0D]) | upper | lower | ((b >= 0x30) & (b <= 0x39))
    flags |= (printable & ~known).astype(np.uint8) * 0x08
    return flags


def lowercase_numpy(text_bytes):
    b = np.frombuffer(text_bytes, dtype=np.uint8).copy()
    upper = (b >= 0x41) & (b <= 0x5A)
    b[upper] |= 0x20
    return b


def boundary_numpy(flags):
    boundaries = np.zeros_like(flags)
    boundaries[0] = 1
    boundaries[1:] = (flags[1:] != flags[:-1]).astype(np.uint8)
    return boundaries


# ---------------------------------------------------------------------------
# Build kernels
# ---------------------------------------------------------------------------

def build_kernels():
    build_script = DEMO_DIR / "build.sh"
    if not build_script.exists():
        print("ERROR: build.sh not found")
        sys.exit(1)
    print("Building Ea kernels...")
    result = subprocess.run(
        ["bash", str(build_script)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)
    print(result.stdout.strip())


def load_kernels():
    so_path = DEMO_DIR / "prepass.so"
    if not so_path.exists():
        build_kernels()

    lib = ctypes.CDLL(str(so_path))

    # Fused: text_prepass_fused(text, flags, lower, boundaries, len)
    lib.text_prepass_fused.argtypes = [U8_PTR, U8_PTR, U8_PTR, U8_PTR, ctypes.c_int32]
    lib.text_prepass_fused.restype = None

    # Unfused
    lib.classify_u8x16.argtypes = [U8_PTR, U8_PTR, ctypes.c_int32]
    lib.classify_u8x16.restype = None
    lib.lowercase_u8x16.argtypes = [U8_PTR, U8_PTR, ctypes.c_int32]
    lib.lowercase_u8x16.restype = None
    lib.boundary_detect.argtypes = [U8_PTR, U8_PTR, ctypes.c_int32]
    lib.boundary_detect.restype = None

    return lib, lib


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(func, n_warmup=5, n_runs=50):
    for _ in range(n_warmup):
        func()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sorted(times)[len(times) // 2]  # median


def main():
    text_bytes = get_test_text()
    n = len(text_bytes)
    text_arr = np.frombuffer(text_bytes, dtype=np.uint8).copy()
    text_ptr = text_arr.ctypes.data_as(U8_PTR)

    print(f"\nText size: {n:,} bytes")
    print("=" * 60)

    # --- NumPy baseline ---
    def run_numpy():
        f = classify_numpy(text_bytes)
        l = lowercase_numpy(text_bytes)
        b = boundary_numpy(f)
        return f, l, b

    np_flags, np_lower, np_boundaries = run_numpy()
    t_numpy = benchmark(lambda: run_numpy())
    print(f"NumPy (3 stages)     : {t_numpy*1000:8.2f} ms")

    # --- Ea unfused ---
    fused_lib, unfused_lib = load_kernels()

    flags_ea = np.zeros(n, dtype=np.uint8)
    lower_ea = np.zeros(n, dtype=np.uint8)
    bound_ea = np.zeros(n, dtype=np.uint8)
    flags_ptr = flags_ea.ctypes.data_as(U8_PTR)
    lower_ptr = lower_ea.ctypes.data_as(U8_PTR)
    bound_ptr = bound_ea.ctypes.data_as(U8_PTR)

    def run_ea_unfused():
        unfused_lib.classify_u8x16(text_ptr, flags_ptr, n)
        unfused_lib.lowercase_u8x16(text_ptr, lower_ptr, n)
        unfused_lib.boundary_detect(flags_ptr, bound_ptr, n)

    run_ea_unfused()
    t_unfused = benchmark(run_ea_unfused)
    print(f"Ea unfused (3 calls) : {t_unfused*1000:8.2f} ms")

    # --- Ea fused ---
    flags_f = np.zeros(n, dtype=np.uint8)
    lower_f = np.zeros(n, dtype=np.uint8)
    bound_f = np.zeros(n, dtype=np.uint8)
    flags_f_ptr = flags_f.ctypes.data_as(U8_PTR)
    lower_f_ptr = lower_f.ctypes.data_as(U8_PTR)
    bound_f_ptr = bound_f.ctypes.data_as(U8_PTR)

    def run_ea_fused():
        fused_lib.text_prepass_fused(text_ptr, flags_f_ptr, lower_f_ptr, bound_f_ptr, n)

    run_ea_fused()
    t_fused = benchmark(run_ea_fused)
    print(f"Ea fused (1 call)    : {t_fused*1000:8.2f} ms")

    # --- Speedups ---
    print()
    if t_unfused > 0:
        print(f"Ea unfused vs NumPy  :  {t_numpy/t_unfused:.1f}x")
    if t_fused > 0:
        print(f"Ea fused vs NumPy    :  {t_numpy/t_fused:.1f}x")
    if t_fused > 0 and t_unfused > 0:
        print(f"Fusion speedup       :  {t_unfused/t_fused:.2f}x")

    # --- Correctness ---
    print()
    print("Correctness check:")

    # Unfused vs NumPy
    flags_match = np.sum(flags_ea == np_flags) / n * 100
    lower_match = np.sum(lower_ea == np_lower) / n * 100
    bound_match = np.sum(bound_ea == np_boundaries) / n * 100
    print(f"  Unfused vs NumPy -- flags: {flags_match:.2f}%, lower: {lower_match:.2f}%, boundary: {bound_match:.2f}%")

    # Fused vs Unfused (must be exact)
    flags_exact = np.array_equal(flags_f, flags_ea)
    lower_exact = np.array_equal(lower_f, lower_ea)
    bound_exact = np.array_equal(bound_f, bound_ea)
    print(f"  Fused vs Unfused  -- flags: {'EXACT' if flags_exact else 'DIFFER'}, "
          f"lower: {'EXACT' if lower_exact else 'DIFFER'}, "
          f"boundary: {'EXACT' if bound_exact else 'DIFFER'}")

    if not (flags_exact and lower_exact and bound_exact):
        print("  WARNING: Fused and unfused produce different results!")

    # --- Visual output ---
    print()
    print("Visual output (first 80 bytes):")
    show_len = min(80, n)
    excerpt = text_arr[:show_len]
    lowered_excerpt = lower_f[:show_len]
    flags_excerpt = flags_f[:show_len]
    bound_excerpt = bound_f[:show_len]

    # Show original with boundaries
    tokens = []
    current_token = ""
    for j in range(show_len):
        ch = chr(excerpt[j]) if 32 <= excerpt[j] < 127 else '.'
        if bound_excerpt[j] == 1 and current_token:
            tokens.append(current_token)
            current_token = ""
        current_token += ch
    if current_token:
        tokens.append(current_token)
    print(f"  Tokens: |{'|'.join(tokens)}|")

    # Show lowercased
    lowered_str = ""
    for j in range(show_len):
        ch = chr(lowered_excerpt[j]) if 32 <= lowered_excerpt[j] < 127 else '.'
        lowered_str += ch
    print(f"  Lower:  {lowered_str}")

    # Show flag names
    FLAG_NAMES = {0x01: "WS", 0x02: "LT", 0x04: "DG", 0x08: "PN", 0x10: "NA"}
    flag_str = ""
    for j in range(min(40, show_len)):
        f = flags_excerpt[j]
        flag_str += FLAG_NAMES.get(f, f"?{f:02x}") + " "
    print(f"  Flags:  {flag_str.strip()}")

    # --- Memory ---
    print()
    numpy_mem = n * 8  # ~8 temp arrays
    ea_unfused_mem = n * 4  # flags(intermediate) + 3 outputs
    ea_fused_mem = n * 3  # 3 outputs, zero intermediates
    print(f"Memory estimate:")
    print(f"  NumPy    : {numpy_mem / 1024 / 1024:.1f} MB (temporary arrays)")
    print(f"  Unfused  : {ea_unfused_mem / 1024 / 1024:.1f} MB (1 intermediate)")
    print(f"  Fused    : {ea_fused_mem / 1024 / 1024:.1f} MB (zero intermediates)")


if __name__ == "__main__":
    main()
