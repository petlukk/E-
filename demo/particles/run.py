#!/usr/bin/env python3
"""
Particle Struct Demo: proves Ea struct layout matches C over FFI.

This is a correctness demo, not a performance demo.
It verifies that ctypes.Structure and Ea's struct Particle have
identical memory layout: field offsets, sizes, and alignment.

Usage:
    bash build.sh   # compile particle.ea -> particle.so
    python run.py   # run correctness tests + performance comparison
"""

import ctypes
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).parent
EA_ROOT = DEMO_DIR / ".." / ".."


# ---------------------------------------------------------------------------
# Struct definition (must match particle.ea exactly)
# ---------------------------------------------------------------------------

class Particle(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("vx", ctypes.c_float),
        ("vy", ctypes.c_float),
    ]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_ea_kernel():
    """Compile particle.ea to particle.so if needed."""
    so_path = DEMO_DIR / "particle.so"
    ea_path = DEMO_DIR / "particle.ea"

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

    built = EA_ROOT / "particle.so"
    if built.exists():
        built.rename(so_path)

    print(f"  Built: {so_path}")
    return so_path


def load_library(so_path):
    """Load particle.so and set up function signatures."""
    lib = ctypes.CDLL(str(so_path))

    lib.update_particles.argtypes = [
        ctypes.POINTER(Particle),
        ctypes.c_int32,
        ctypes.c_float,
    ]
    lib.update_particles.restype = None

    lib.kinetic_energy.argtypes = [
        ctypes.POINTER(Particle),
        ctypes.c_int32,
    ]
    lib.kinetic_energy.restype = ctypes.c_float

    return lib


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_update_particles(lib):
    """Test update_particles with 4 known particles."""
    print("=== Correctness: update_particles ===")

    particles = (Particle * 4)(
        Particle(x=0.0,  y=0.0,   vx=1.0,  vy=2.0),
        Particle(x=10.0, y=20.0,  vx=-0.5, vy=0.5),
        Particle(x=5.0,  y=5.0,   vx=3.0,  vy=-1.0),
        Particle(x=100.0, y=-50.0, vx=0.0,  vy=10.0),
    )

    dt = 0.1

    # Python reference: x_new = x + vx * dt, y_new = y + vy * dt
    expected = [
        (0.0 + 1.0 * dt,    0.0 + 2.0 * dt),
        (10.0 + (-0.5) * dt, 20.0 + 0.5 * dt),
        (5.0 + 3.0 * dt,    5.0 + (-1.0) * dt),
        (100.0 + 0.0 * dt,  -50.0 + 10.0 * dt),
    ]

    lib.update_particles(particles, ctypes.c_int32(4), ctypes.c_float(dt))

    all_pass = True
    for i in range(4):
        ex, ey = expected[i]
        px, py = particles[i].x, particles[i].y
        x_ok = math.isclose(px, ex, rel_tol=1e-6)
        y_ok = math.isclose(py, ey, rel_tol=1e-6)
        status = "PASS" if (x_ok and y_ok) else "FAIL"
        if not (x_ok and y_ok):
            all_pass = False
        print(f"  Particle {i}: x={px:.4f} (expected {ex:.4f}), "
              f"y={py:.4f} (expected {ey:.4f})  [{status}]")

    print()
    return all_pass


def test_kinetic_energy(lib):
    """Test kinetic_energy with known velocities."""
    print("=== Correctness: kinetic_energy ===")

    # Particle 0: vx=3, vy=4 -> 0.5 * (9 + 16) = 12.5
    # Particle 1: vx=0, vy=0 -> 0.5 * (0 + 0) = 0.0
    # Total KE = 12.5
    particles = (Particle * 2)(
        Particle(x=0.0, y=0.0, vx=3.0, vy=4.0),
        Particle(x=0.0, y=0.0, vx=0.0, vy=0.0),
    )

    result = lib.kinetic_energy(particles, ctypes.c_int32(2))
    expected = 12.5
    ok = math.isclose(result, expected, rel_tol=1e-6)
    status = "PASS" if ok else "FAIL"
    print(f"  KE = {result:.4f} (expected {expected:.4f})  [{status}]")
    print()
    return ok


# ---------------------------------------------------------------------------
# Performance comparison
# ---------------------------------------------------------------------------

def benchmark(func, warmup=5, runs=50):
    """Run function multiple times, return median time in ms."""
    for _ in range(warmup):
        func()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    median = times[len(times) // 2]
    return median


def perf_comparison(lib):
    """Compare Ea update_particles vs NumPy SoA equivalent."""
    print("=== Performance: 100,000 particles ===")

    n = 100_000
    dt = 0.016  # ~60 fps

    # --- Ea (AoS via struct) ---
    particles = (Particle * n)()
    rng = np.random.default_rng(42)
    positions = rng.uniform(-100, 100, size=(n, 2)).astype(np.float32)
    velocities = rng.uniform(-10, 10, size=(n, 2)).astype(np.float32)
    for i in range(n):
        particles[i].x = positions[i, 0]
        particles[i].y = positions[i, 1]
        particles[i].vx = velocities[i, 0]
        particles[i].vy = velocities[i, 1]

    def run_ea():
        lib.update_particles(particles, ctypes.c_int32(n), ctypes.c_float(dt))

    t_ea = benchmark(run_ea)

    # --- NumPy SoA ---
    x = positions[:, 0].copy()
    y = positions[:, 1].copy()
    vx = velocities[:, 0].copy()
    vy = velocities[:, 1].copy()

    def run_numpy():
        nonlocal x, y
        x += vx * dt
        y += vy * dt

    t_numpy = benchmark(run_numpy)

    print(f"  Ea (AoS struct)  : {t_ea:8.3f} ms  (median of 50 runs)")
    print(f"  NumPy (SoA)      : {t_numpy:8.3f} ms  (median of 50 runs)")
    ratio = t_ea / t_numpy if t_numpy > 0 else float("inf")
    if t_ea < t_numpy:
        print(f"  Ea is {t_numpy / t_ea:.1f}x faster")
    else:
        print(f"  NumPy is {ratio:.1f}x faster")
    print()
    print("  Note: NumPy uses SoA (Structure-of-Arrays) layout which has")
    print("  better cache behavior for this workload. Ea uses AoS (Array-of-")
    print("  Structs) which matches the C struct layout. This demo proves")
    print("  correctness of struct FFI, not performance superiority.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Particle Struct Demo")
    print("Proves Ea struct layout matches C exactly over FFI\n")

    so_path = build_ea_kernel()
    lib = load_library(so_path)

    pass1 = test_update_particles(lib)
    pass2 = test_kinetic_energy(lib)

    perf_comparison(lib)

    print("=== Summary ===")
    all_pass = pass1 and pass2
    if all_pass:
        print("  All correctness tests PASSED")
        print("  Struct layout is C-compatible over FFI")
    else:
        print("  Some tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
