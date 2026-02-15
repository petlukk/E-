# Benchmark Methodology

## Reference Environment

Results reported in the README were measured on:

- **CPU**: AMD Ryzen 7 1700 Eight-Core Processor (Zen 1)
- **Features**: SSE4.2, AVX2, FMA
- **OS**: Ubuntu 22.04 on WSL2 (kernel 5.15.146.1)
- **GCC**: 11.4.0 (`gcc -O3 -march=native -ffast-math`)
- **LLVM**: 14.0.0 (via inkwell, no external `llc`)
- **Ea**: strict IEEE — no fast-math flags

## Parameters

| Parameter | FMA Benchmark | Reduction Benchmark |
|---|---|---|
| Array size | 1,000,000 f32 | 1,000,000 f32 |
| Runs | 100 | 200 |
| Warmup | 10 | 20 |
| Timing | `time.perf_counter()` | `time.perf_counter()` |
| Metric | Average wall time | Average wall time |
| Seed | `np.random.seed(42)` | `np.random.seed(42)` |

## Competitors

Each benchmark always runs Ea vs GCC (hand-written intrinsics). Additional
competitors are included automatically when their toolchains are detected:

| Competitor | Source | Flags | How to install |
|---|---|---|---|
| GCC (required) | `reference.c` | `-O3 -march=native -ffast-math` | System package |
| Clang | `reference.c` | `-O3 -march=native -ffast-math` | `apt install clang-14` (or any version) |
| ISPC | `.ispc` kernels | `--target=avx2-i32x8 -O2` | [ISPC releases](https://github.com/ispc/ispc/releases) |
| Rust std::simd | `rust_competitors/` | `-C target-cpu=native` (nightly) | `rustup toolchain install nightly` |

Missing tools are skipped gracefully — the core Ea-vs-GCC comparison always
runs. The baseline for ratio calculations is the fastest C SIMD implementation
(GCC or Clang), ensuring an honest comparison.

## What is Measured

Each benchmark compiles both an Ea kernel (`.so` via `--lib`) and a C reference
(`.so` via `gcc -O3 -march=native -ffast-math -shared -fPIC`), loads them via
`ctypes`, and calls them repeatedly on the same data.

The C reference uses explicit AVX2/SSE intrinsics — not auto-vectorized scalar
code. This is the strongest possible baseline.

## What is NOT Measured

- Memory allocation (arrays pre-allocated before timing loop)
- Library loading (loaded once before benchmarking)
- Compilation time

## Reproducing

```bash
cd benchmarks/fma_kernel && python3 bench.py
cd benchmarks/horizontal_reduction && python3 bench.py
```

Both scripts print CPU, compiler, and OS info at the top of the output.

## Key Design Decisions

- **No fast-math for Ea**: Ea uses strict IEEE floating point. The C reference
  uses `-ffast-math` because that is what a C developer would use in practice.
  Ea matching this baseline without fast-math is the stronger claim.
- **Explicit intrinsics in C**: The C reference is hand-written with
  `_mm256_fmadd_ps`, `_mm_max_ps`, etc. — not auto-vectorized. This ensures
  we compare generated SIMD against hand-written SIMD.
- **Host CPU targeting**: Both Ea and GCC target the host CPU (`-march=native`
  / `TargetMachine::get_host_cpu_features()`). Results will vary on different
  hardware.
