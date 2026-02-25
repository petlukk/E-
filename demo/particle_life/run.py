#!/usr/bin/env python3
"""
Particle Life — Eä SIMD kernel demo.

N colored particles interact via asymmetric attraction/repulsion forces.
A random interaction matrix defines how each type affects every other type.
Emergent structures form from simple math running live from a single Eä kernel.

Controls:
    R       — rerandomize interaction matrix
    F       — toggle fused / unfused mode
    +/-     — increase / decrease particle count
    ESC     — quit

Usage:
    python run.py [--particles N]
"""

import argparse
import ctypes
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).parent
EA_ROOT = DEMO_DIR / ".." / ".."

NUM_TYPES = 6
R_MAX = 80.0
FRICTION = 0.5
DT = 0.02
SIZE = 800.0

COLORS = [
    (255, 70, 70),
    (70, 255, 70),
    (70, 130, 255),
    (255, 255, 70),
    (255, 70, 255),
    (70, 255, 255),
]


def build_kernel(ea_file):
    """Compile an .ea file to .so, return path to .so."""
    so_name = ea_file.stem + ".so"
    so_path = DEMO_DIR / so_name
    ea_path = DEMO_DIR / ea_file.name

    if so_path.exists() and so_path.stat().st_mtime > ea_path.stat().st_mtime:
        return so_path

    print(f"Building {ea_file.name}...")
    result = subprocess.run(
        ["cargo", "run", "--features=llvm", "--release", "--",
         str(ea_path), "--lib"],
        capture_output=True, text=True, cwd=str(EA_ROOT),
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)

    built = EA_ROOT / so_name
    if built.exists():
        built.rename(so_path)

    print(f"  Built: {so_path}")
    return so_path


def load_fused(so_path):
    """Load fused kernel .so and configure ctypes signatures."""
    lib = ctypes.CDLL(str(so_path))
    f32p = ctypes.POINTER(ctypes.c_float)
    i32p = ctypes.POINTER(ctypes.c_int32)
    lib.particle_life_step.argtypes = [
        f32p, f32p, f32p, f32p, i32p, f32p,
        ctypes.c_int32, ctypes.c_int32,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    lib.particle_life_step.restype = None
    return lib


def load_unfused(so_path):
    """Load unfused kernels .so and configure ctypes signatures."""
    lib = ctypes.CDLL(str(so_path))
    f32p = ctypes.POINTER(ctypes.c_float)
    i32p = ctypes.POINTER(ctypes.c_int32)
    lib.compute_forces.argtypes = [
        f32p, f32p, i32p, f32p, f32p, f32p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_float,
    ]
    lib.compute_forces.restype = None
    lib.update_velocities.argtypes = [
        f32p, f32p, f32p, f32p,
        ctypes.c_int32, ctypes.c_float, ctypes.c_float,
    ]
    lib.update_velocities.restype = None
    lib.update_positions.argtypes = [
        f32p, f32p, f32p, f32p,
        ctypes.c_int32, ctypes.c_float, ctypes.c_float,
    ]
    lib.update_positions.restype = None
    return lib


def make_state(n, rng):
    """Allocate and randomize particle state arrays."""
    px = rng.uniform(0, SIZE, n).astype(np.float32)
    py = rng.uniform(0, SIZE, n).astype(np.float32)
    vx = np.zeros(n, dtype=np.float32)
    vy = np.zeros(n, dtype=np.float32)
    types = rng.integers(0, NUM_TYPES, n).astype(np.int32)
    matrix = rng.uniform(-1, 1, NUM_TYPES * NUM_TYPES).astype(np.float32)
    fx = np.zeros(n, dtype=np.float32)
    fy = np.zeros(n, dtype=np.float32)
    return px, py, vx, vy, types, matrix, fx, fy


def ptr(arr):
    """Get ctypes pointer from numpy array."""
    if arr.dtype == np.float32:
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def main():
    parser = argparse.ArgumentParser(description="Particle Life — Eä demo")
    parser.add_argument("--particles", "-n", type=int, default=2000)
    args = parser.parse_args()

    fused_so = build_kernel(Path("particle_life.ea"))
    unfused_so = build_kernel(Path("particle_life_unfused.ea"))
    fused_lib = load_fused(fused_so)
    unfused_lib = load_unfused(unfused_so)

    rng = np.random.default_rng()
    n = args.particles
    px, py, vx, vy, types, matrix, fx, fy = make_state(n, rng)

    use_fused = True

    try:
        import pygame
    except ImportError:
        print("pygame not installed. Install with: pip install pygame")
        print("Running headless benchmark instead.\n")
        headless_benchmark(fused_lib, unfused_lib, n, px, py, vx, vy,
                           types, matrix, fx, fy)
        return

    pygame.init()
    screen = pygame.display.set_mode((int(SIZE), int(SIZE)))
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    matrix = rng.uniform(-1, 1,
                                         NUM_TYPES * NUM_TYPES).astype(np.float32)
                elif event.key == pygame.K_f:
                    use_fused = not use_fused
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    n = min(n + 500, 10000)
                    px, py, vx, vy, types, matrix, fx, fy = make_state(n, rng)
                elif event.key == pygame.K_MINUS:
                    n = max(n - 500, 100)
                    px, py, vx, vy, types, matrix, fx, fy = make_state(n, rng)

        t0 = time.perf_counter()

        if use_fused:
            fused_lib.particle_life_step(
                ptr(px), ptr(py), ptr(vx), ptr(vy), ptr(types), ptr(matrix),
                ctypes.c_int32(n), ctypes.c_int32(NUM_TYPES),
                ctypes.c_float(R_MAX), ctypes.c_float(DT),
                ctypes.c_float(FRICTION), ctypes.c_float(SIZE),
            )
        else:
            unfused_lib.compute_forces(
                ptr(px), ptr(py), ptr(types), ptr(matrix),
                ptr(fx), ptr(fy),
                ctypes.c_int32(n), ctypes.c_int32(NUM_TYPES),
                ctypes.c_float(R_MAX),
            )
            unfused_lib.update_velocities(
                ptr(vx), ptr(vy), ptr(fx), ptr(fy),
                ctypes.c_int32(n), ctypes.c_float(DT),
                ctypes.c_float(FRICTION),
            )
            unfused_lib.update_positions(
                ptr(px), ptr(py), ptr(vx), ptr(vy),
                ctypes.c_int32(n), ctypes.c_float(DT),
                ctypes.c_float(SIZE),
            )

        step_ms = (time.perf_counter() - t0) * 1000

        screen.fill((0, 0, 0))
        for k in range(n):
            color = COLORS[int(types[k]) % len(COLORS)]
            x = int(px[k]) % int(SIZE)
            y = int(py[k]) % int(SIZE)
            screen.set_at((x, y), color)

        mode = "FUSED" if use_fused else "UNFUSED"
        fps = clock.get_fps()
        pygame.display.set_caption(
            f"Particle Life [{mode}] N={n} | {step_ms:.1f}ms | {fps:.0f} FPS"
        )
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def headless_benchmark(fused_lib, unfused_lib, n, px, py, vx, vy,
                       types, matrix, fx, fy):
    """Run benchmark without pygame."""
    warmup = 5
    runs = 50

    for _ in range(warmup):
        fused_lib.particle_life_step(
            ptr(px), ptr(py), ptr(vx), ptr(vy), ptr(types), ptr(matrix),
            ctypes.c_int32(n), ctypes.c_int32(NUM_TYPES),
            ctypes.c_float(R_MAX), ctypes.c_float(DT),
            ctypes.c_float(FRICTION), ctypes.c_float(SIZE),
        )

    times_fused = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fused_lib.particle_life_step(
            ptr(px), ptr(py), ptr(vx), ptr(vy), ptr(types), ptr(matrix),
            ctypes.c_int32(n), ctypes.c_int32(NUM_TYPES),
            ctypes.c_float(R_MAX), ctypes.c_float(DT),
            ctypes.c_float(FRICTION), ctypes.c_float(SIZE),
        )
        times_fused.append((time.perf_counter() - t0) * 1000)

    for _ in range(warmup):
        unfused_lib.compute_forces(
            ptr(px), ptr(py), ptr(types), ptr(matrix), ptr(fx), ptr(fy),
            ctypes.c_int32(n), ctypes.c_int32(NUM_TYPES),
            ctypes.c_float(R_MAX),
        )
        unfused_lib.update_velocities(
            ptr(vx), ptr(vy), ptr(fx), ptr(fy),
            ctypes.c_int32(n), ctypes.c_float(DT), ctypes.c_float(FRICTION),
        )
        unfused_lib.update_positions(
            ptr(px), ptr(py), ptr(vx), ptr(vy),
            ctypes.c_int32(n), ctypes.c_float(DT), ctypes.c_float(SIZE),
        )

    times_unfused = []
    for _ in range(runs):
        t0 = time.perf_counter()
        unfused_lib.compute_forces(
            ptr(px), ptr(py), ptr(types), ptr(matrix), ptr(fx), ptr(fy),
            ctypes.c_int32(n), ctypes.c_int32(NUM_TYPES),
            ctypes.c_float(R_MAX),
        )
        unfused_lib.update_velocities(
            ptr(vx), ptr(vy), ptr(fx), ptr(fy),
            ctypes.c_int32(n), ctypes.c_float(DT), ctypes.c_float(FRICTION),
        )
        unfused_lib.update_positions(
            ptr(px), ptr(py), ptr(vx), ptr(vy),
            ctypes.c_int32(n), ctypes.c_float(DT), ctypes.c_float(SIZE),
        )
        times_unfused.append((time.perf_counter() - t0) * 1000)

    times_fused.sort()
    times_unfused.sort()
    med_fused = times_fused[len(times_fused) // 2]
    med_unfused = times_unfused[len(times_unfused) // 2]

    print(f"Particle Life Benchmark (N={n})")
    print(f"  Fused   (1 kernel) : {med_fused:8.3f} ms")
    print(f"  Unfused (3 kernels): {med_unfused:8.3f} ms")
    if med_fused > 0:
        ratio = med_unfused / med_fused
        print(f"  Fused is {ratio:.1f}x {'faster' if ratio > 1 else 'slower'}")


if __name__ == "__main__":
    main()
