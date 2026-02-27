#!/usr/bin/env python3
"""Cornell Box ray tracer — Eä demo.

Renders a classic Cornell Box scene: 5 walls (red left, green right,
white floor/ceiling/back), 2 spheres (one diffuse, one mirror),
direct lighting with hard shadows, single-bounce mirror reflection.

The entire ray tracer runs as a single Eä kernel.

Usage:
    python run.py [width] [height]
    python run.py              # default 512x512
    python run.py 256 256      # quick render
"""

import ctypes
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).parent
EA_ROOT = DEMO_DIR / ".." / ".."


def compile_kernel():
    ea_path = DEMO_DIR / "cornell.ea"
    so_path = DEMO_DIR / "cornell.so"

    print("Compiling cornell.ea ...")
    result = subprocess.run(
        ["cargo", "run", "--features=llvm", "--release", "--",
         str(ea_path), "--lib"],
        capture_output=True, text=True, cwd=str(EA_ROOT),
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)

    built = EA_ROOT / "cornell.so"
    if built.exists():
        built.rename(so_path)

    print(f"  Built: {so_path}")
    return so_path


def main():
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    height = int(sys.argv[2]) if len(sys.argv) > 2 else 512

    so_path = compile_kernel()

    lib = ctypes.CDLL(str(so_path))
    lib.render.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.render.restype = None

    buf = np.zeros((height, width, 3), dtype=np.float32)
    ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Warmup
    print(f"\nRendering {width}x{height} ...")
    lib.render(ptr, width, height)

    # Benchmark
    n_runs = 10
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        lib.render(ptr, width, height)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    median = sorted(times)[n_runs // 2]

    # Save image
    img = np.clip(buf, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)

    out_path = DEMO_DIR / "output.png"
    try:
        from PIL import Image
        Image.fromarray(img).save(str(out_path))
    except ImportError:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(str(out_path), img)

    print(f"  Saved: {out_path}")

    print(f"\n=== Performance ===")
    print(f"  Resolution    : {width}x{height}")
    print(f"  Pixels        : {width * height:,}")
    print(f"  Eä render     : {median:.1f} ms (median of {n_runs} runs)")
    print(f"  Rays/sec      : {width * height / (median / 1000):,.0f}")
    print(f"  Features used : sqrt, rsqrt, to_f32, unary negation, struct return, recursion")

    print(f"\n=== Scene ===")
    print(f"  5 axis-aligned walls (red left, green right, white floor/ceiling/back)")
    print(f"  1 diffuse white sphere, 1 mirror sphere")
    print(f"  Point light with hard shadows")
    print(f"  Single-bounce mirror reflection")


if __name__ == "__main__":
    main()
