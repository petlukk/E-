#!/usr/bin/env python3
"""
Particle Life — Eä SIMD kernel demo.

N colored particles interact via asymmetric attraction/repulsion forces.
A random interaction matrix defines how each type affects every other type.
Emergent structures form from simple math running live from a single Eä kernel.

Controls (keyboard or click panel buttons):
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
DT = 0.5
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


PANEL_H = 50
PANEL_BG = (30, 30, 30)
BTN_BG = (60, 60, 60)
BTN_HOVER = (80, 80, 80)
BTN_ACTIVE = (50, 120, 200)
BTN_TEXT = (220, 220, 220)
STATUS_TEXT = (180, 180, 180)


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

    def make_button(x, y, w, h, label):
        return {"rect": pygame.Rect(x, y, w, h), "label": label}

    def draw_button(btn, hover=False, active=False):
        bg = BTN_ACTIVE if active else (BTN_HOVER if hover else BTN_BG)
        pygame.draw.rect(screen, bg, btn["rect"], border_radius=4)
        pygame.draw.rect(screen, (100, 100, 100), btn["rect"], 1,
                         border_radius=4)
        text_surf = font.render(btn["label"], True, BTN_TEXT)
        text_rect = text_surf.get_rect(center=btn["rect"].center)
        screen.blit(text_surf, text_rect)

    pygame.init()
    win_w = int(SIZE)
    win_h = int(SIZE) + PANEL_H
    screen = pygame.display.set_mode((win_w, win_h), pygame.DOUBLEBUF)
    pygame.display.set_caption("Particle Life — Eä SIMD Demo")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    panel_y = int(SIZE)
    pad = 8
    btn_h = PANEL_H - 2 * pad
    btn_w = 90
    btn_mode = make_button(pad, panel_y + pad, btn_w, btn_h, "FUSED")
    btn_rand = make_button(pad + btn_w + pad, panel_y + pad, btn_w, btn_h,
                           "R Rerand")
    btn_more = make_button(pad + 2 * (btn_w + pad), panel_y + pad, 60, btn_h,
                           "+ More")
    btn_less = make_button(pad + 2 * (btn_w + pad) + 60 + pad, panel_y + pad,
                           60, btn_h, "- Less")
    buttons = [btn_mode, btn_rand, btn_more, btn_less]

    step_ms = 0.0
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

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
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_mode["rect"].collidepoint(event.pos):
                    use_fused = not use_fused
                elif btn_rand["rect"].collidepoint(event.pos):
                    matrix = rng.uniform(-1, 1,
                                         NUM_TYPES * NUM_TYPES).astype(np.float32)
                elif btn_more["rect"].collidepoint(event.pos):
                    n = min(n + 500, 10000)
                    px, py, vx, vy, types, matrix, fx, fy = make_state(n, rng)
                elif btn_less["rect"].collidepoint(event.pos):
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
        sz = int(SIZE)
        for k in range(n):
            color = COLORS[int(types[k]) % len(COLORS)]
            x = int(px[k]) % sz
            y = int(py[k]) % sz
            pygame.draw.circle(screen, color, (x, y), 2)

        # --- bottom panel ---
        pygame.draw.rect(screen, PANEL_BG, (0, panel_y, win_w, PANEL_H))

        btn_mode["label"] = "FUSED" if use_fused else "UNFUSED"
        for btn in buttons:
            hover = btn["rect"].collidepoint(mouse_pos)
            active = (btn is btn_mode)
            draw_button(btn, hover=hover, active=active)

        mode = "Fused" if use_fused else "Unfused"
        fps = clock.get_fps()
        status = f"{mode} | N={n} | {step_ms:.1f}ms | {fps:.0f} FPS"
        status_surf = font.render(status, True, STATUS_TEXT)
        screen.blit(status_surf, (win_w - status_surf.get_width() - pad,
                                  panel_y + PANEL_H // 2
                                  - status_surf.get_height() // 2))

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
