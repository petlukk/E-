#!/usr/bin/env python3
"""Quick FMA benchmark for iterating on optimization passes."""

import os
import subprocess
import time
import numpy as np
import ctypes
from pathlib import Path

ARRAY_SIZE = 1_000_000
NUM_RUNS = 100
WARMUP_RUNS = 20

def main():
    bench_dir = Path(__file__).parent
    ea_root = bench_dir.parent.parent
    os.chdir(bench_dir)

    # Compile Ea kernel
    r = subprocess.run(
        ["cargo", "run", "--features=llvm", "--", str(bench_dir / "kernel.ea"), "--lib"],
        capture_output=True, text=True, cwd=ea_root,
    )
    if r.returncode != 0:
        print(f"Ea compile failed:\n{r.stderr}")
        return
    # Move .so
    so_src = ea_root / "kernel.so"
    if so_src.exists():
        so_src.rename(bench_dir / "kernel.so")

    # Compile C reference
    subprocess.run(
        ["gcc", "-O3", "-march=native", "-ffast-math", "-shared", "-fPIC",
         "reference.c", "-o", "reference.so"],
        capture_output=True, cwd=bench_dir,
    )

    ea = ctypes.CDLL(str(bench_dir / "kernel.so"))
    c = ctypes.CDLL(str(bench_dir / "reference.so"))

    FP = ctypes.POINTER(ctypes.c_float)
    I32 = ctypes.c_int32
    for lib, name in [(ea, "fma_kernel_f32x4"), (c, "fma_kernel_f32x4_c"),
                       (c, "fma_kernel_f32x8_c"), (c, "fma_kernel_scalar_c")]:
        fn = getattr(lib, name)
        fn.argtypes = [FP, FP, FP, FP, I32]
        fn.restype = None

    np.random.seed(42)
    a = np.random.uniform(-1, 1, ARRAY_SIZE).astype(np.float32)
    b = np.random.uniform(-1, 1, ARRAY_SIZE).astype(np.float32)
    cc = np.random.uniform(-1, 1, ARRAY_SIZE).astype(np.float32)
    result = np.zeros(ARRAY_SIZE, dtype=np.float32)

    ap = a.ctypes.data_as(FP)
    bp = b.ctypes.data_as(FP)
    cp = cc.ctypes.data_as(FP)
    rp = result.ctypes.data_as(FP)

    def bench(func):
        for _ in range(WARMUP_RUNS):
            func(ap, bp, cp, rp, ARRAY_SIZE)
        times = []
        for _ in range(NUM_RUNS):
            s = time.perf_counter()
            func(ap, bp, cp, rp, ARRAY_SIZE)
            times.append(time.perf_counter() - s)
        return sum(times) / len(times), min(times)

    c8_avg, c8_min = bench(c.fma_kernel_f32x8_c)
    c4_avg, c4_min = bench(c.fma_kernel_f32x4_c)
    sc_avg, sc_min = bench(c.fma_kernel_scalar_c)
    ea4_avg, ea4_min = bench(ea.fma_kernel_f32x4)

    print(f"C f32x8:   avg {c8_avg*1e6:7.1f}us  min {c8_min*1e6:7.1f}us  (baseline)")
    print(f"C f32x4:   avg {c4_avg*1e6:7.1f}us  min {c4_min*1e6:7.1f}us  {c4_avg/c8_avg:.3f}x")
    print(f"C scalar:  avg {sc_avg*1e6:7.1f}us  min {sc_min*1e6:7.1f}us  {sc_avg/c8_avg:.3f}x")
    print(f"Ea f32x4:  avg {ea4_avg*1e6:7.1f}us  min {ea4_min*1e6:7.1f}us  {ea4_avg/c8_avg:.3f}x")
    print(f"\nEa f32x4 vs C f32x4: {ea4_avg/c4_avg:.3f}x")
    print(f"Ea f32x4 vs C f32x8: {ea4_avg/c8_avg:.3f}x")

if __name__ == "__main__":
    main()
