# Ea

**SIMD kernel language for C and Python.**

Write readable SIMD code. Compile to `.o` or `.so`. Call from C, Rust, Python via C ABI.
No runtime. No garbage collector. Explicit memory control. Just kernels.

## Example

```
export func fma_kernel(a: *f32, b: *f32, c: *f32, out: *mut f32, len: i32) {
    let mut i: i32 = 0
    while i + 4 <= len {
        let va: f32x4 = load(a, i)
        let vb: f32x4 = load(b, i)
        let vc: f32x4 = load(c, i)
        store(out, i, fma(va, vb, vc))
        i = i + 4
    }
}
```

SIMD width and tail handling are explicit by design -- you control the vector width, loop stepping, and remainder logic. No auto-vectorizer magic.

Compile and call from C:

```bash
ea kernel.ea --lib    # produces kernel.so
```

```c
extern void fma_kernel(const float*, const float*, const float*, float*, int);

fma_kernel(a, b, c, result, n);  // that's it
```

## Benchmark

Measured on AMD Ryzen 7 1700 (Zen 1, AVX2/FMA), GCC 11.4, LLVM 18, Linux (WSL2).
1M elements, 100-200 runs, averaged. Full methodology and scripts in `benchmarks/`.

**Ea uses strict IEEE floating point -- no fast-math flags.** The C reference was
compiled with `gcc -O3 -march=native -ffast-math`. Ea matching this baseline
without fast-math is the stronger claim.

### FMA Kernel (result[i] = a[i] \* b[i] + c[i])

| Implementation   | Avg (us) | vs Fastest C |
| ---------------- | -------- | ------------ |
| GCC f32x8 (AVX2) | 887      | 1.03x        |
| **Ea f32x4**     | **885**  | **1.03x**    |
| Clang-14 f32x8   | 859      | 1.00x        |
| ISPC             | 828      | 0.96x        |
| Rust std::simd   | 1001     | 1.17x        |

### Sum Reduction

| Implementation   | Avg (us) | vs Fastest C |
| ---------------- | -------- | ------------ |
| C f32x8 (AVX2)   | 110      | 1.00x        |
| **Ea f32x8**     | **105**  | **0.96x**    |
| Clang-14 f32x8   | 130      | 1.18x        |
| ISPC             | 105      | 0.95x        |
| Rust f32x8       | 111      | 1.01x        |

### Max Reduction (multi-accumulator)

| Implementation  | Avg (us) | vs Fastest C |
| --------------- | -------- | ------------ |
| C f32x4 (SSE)   | 100      | 1.00x        |
| **Ea f32x4**    | **78**   | **0.83x**    |
| Clang-14 f32x4  | 89       | 0.95x        |
| ISPC            | 71       | 0.76x        |
| Rust f32x4      | 180      | 1.93x        |

### Min Reduction (multi-accumulator)

| Implementation  | Avg (us) | vs Fastest C |
| --------------- | -------- | ------------ |
| C f32x4 (SSE)   | 80       | 1.00x        |
| **Ea f32x4**    | **78**   | **0.97x**    |
| Clang-14 f32x4  | 88       | 1.10x        |
| ISPC            | 73       | 0.91x        |
| Rust f32x4      | 220      | 2.77x        |

Ea's reduction kernels use explicit multi-accumulator patterns to break dependency
chains -- see `examples/reduction_multi_acc.ea`. This is faster than relying on
compiler auto-unrolling and stable across LLVM versions.

Competitors are optional -- benchmarks run with whatever toolchains are installed.
GCC is required; Clang, ISPC, and Rust nightly are detected and included automatically.

---

## Additional Benchmark Results (Intel i7-1260P, WSL2)

Measured on Intel i7-1260P (Alder Lake, AVX2/FMA), LLVM 18, Linux (WSL2).
1M elements, 100–200 runs, **minimum** time reported.

### Key Finding: `restrict` produces identical machine code

The `noalias` attribute is correctly emitted in LLVM IR, but it has no measurable impact on these benchmarks because:

- The generated assembly is byte-identical with and without `restrict` (confirmed via `objdump -d` + MD5 comparison)
- Ea's explicit SIMD intrinsics (`load` / `store` / `fma`) already use distinct base pointers, so LLVM's alias analysis does not require `noalias` hints
- Ea's explicit SIMD means the loop vectorizer and SLP passes have little to contribute beyond what is already expressed
- Reduction kernels have a single pointer parameter, making `noalias` vacuous

The implementation is correct and complete — it is simply not performance-relevant for these specific kernels. The feature is positioned for value when the optimizer pipeline grows (auto-tiling, software pipelining, prefetching) or when users write more complex aliasing patterns.

### FMA Kernel (1M f32, 100 runs, min time)

| Implementation       | Min (us) |
| -------------------- | -------- |
| GCC f32x8 (AVX2)     | ~395     |
| Clang-14 f32x8       | ~396     |
| **Ea f32x4**         | **~326** |
| Clang-14 f32x4       | ~376     |
| Rust std::simd f32x4 | ~365     |
| ISPC                 | ~656     |

### Sum Reduction (1M f32, 200 runs, min time)

| Implementation | Min (us) |
| -------------- | -------- |
| ISPC           | ~62      |
| **Ea f32x8**   | **~65**  |
| Rust f32x8     | ~66      |
| C f32x8 (AVX2) | ~68      |

### Notes

- ISPC compilation flag fixed (`--PIC` → `--pic`) in `bench_common.py`
- Benchmarks confirm that explicit SIMD already provides sufficient aliasing information to LLVM
- `restrict` becomes valuable when introducing auto-optimization features or more complex aliasing scenarios

## Performance Principle

LLVM optimizes instructions. Ea lets you optimize dependency structure.

A single-accumulator reduction creates a serial chain -- each iteration waits for
the previous one. On a superscalar CPU, this wastes execution units:

```
// Single accumulator: serial dependency, ~0.25 IPC on Zen 1
acc = max(acc, load(data, i))   // must wait for previous acc
```

Express the parallelism explicitly with multiple accumulators:

```
// Two accumulators: independent chains, ~1.0 IPC on Zen 1
acc0 = max(acc0, load(data, i))      // independent
acc1 = max(acc1, load(data, i + 4))  // independent
```

Result: 2x throughput from a source-level change, stable across LLVM versions,
no compiler flags or optimizer tuning required.

See `examples/reduction_single.ea` and `examples/reduction_multi_acc.ea`.

## Compute Model

Ea defines six kernel patterns that cover most compute workloads:

| Pattern | What it does | Example |
|---------|-------------|---------|
| Streaming | Element-wise transform | `fma.ea` |
| Reduction | Array → scalar with multi-acc ILP | `reduction.ea` |
| Branchless | Conditional logic via `select` | `threshold.ea` |
| Multi-pass | Reduction then streaming | `normalize.ea` |
| Stencil | Neighborhood access (convolution) | `conv2d.ea` |
| Pipeline | Multiple kernels composed | `sobel.ea` |

The full compute model — dependency structure, memory patterns, vector width
selection, and design principles — is documented in [`COMPUTE.md`](COMPUTE.md).

## Demos

| Demo | Domain | What it does |
|------|--------|-------------|
| [Sobel edge detection](demo/sobel/) | Image processing | 2.7x faster than OpenCV on 1080p |
| [Video anomaly detection](demo/video_anomaly/) | Video analysis | Frame differencing + threshold + counting |
| [Astronomy stacking](demo/astro_stack/) | Scientific computing | Mean-stack N noisy exposures, 6x faster, 16x less memory |

Each demo compiles an Ea kernel to `.so`, calls it from Python, and benchmarks against NumPy and OpenCV.

## Why not...

**C with intrinsics?**
Works, but `_mm256_fmadd_ps(_mm256_loadu_ps(&a[i]), ...)` is unreadable and error-prone.
Ea compiles `fma(load(a, i), load(b, i), load(c, i))` to the same instructions.

**Rust with `std::simd`?**
`std::simd` is nightly-only and Rust's type system adds friction for kernel code.
Ea is purpose-built: no lifetimes, no borrows, no generics. Just SIMD.

**ISPC?**
ISPC auto-vectorizes scalar code. Ea gives you explicit control over vector width and operations.
Different philosophy -- Ea is closer to "portable intrinsics" than an auto-vectorizer.

## Safety Model

Ea provides explicit pointer-based memory access similar to C.
There are no bounds checks or runtime safety guarantees -- correctness and
memory safety are the programmer's responsibility. This is intentional:
kernel code needs predictable performance without hidden checks.

## Features

- **SIMD**: `f32x4`, `f32x8`, `i32x4`, `i32x8` with `load`, `store`, `splat`, `fma`, `shuffle`, `select`
- **Reductions**: `reduce_add`, `reduce_max`, `reduce_min`
- **Structs**: C-compatible layout, pointer-to-struct, array-of-structs
- **Pointers**: `*T`, `*mut T`, pointer indexing (`arr[i]`)
- **Types**: `i8`-`i64`, `u8`-`u64`, `f32`, `f64`, `bool`
- **Output**: `.o` object files, `.so`/`.dll` shared libraries, linked executables
- **C ABI**: every `export func` is callable from any language

Currently tested on x86-64 with AVX2. Other architectures depend on LLVM backend support.

## Quick Start

```bash
# Requirements: LLVM 18, Rust
sudo apt install llvm-18-dev clang-18 libpolly-18-dev libzstd-dev

# Build
cargo build --features=llvm

# Compile a kernel to object file
ea kernel.ea              # -> kernel.o

# Compile to shared library
ea kernel.ea --lib        # -> kernel.so

# Compile standalone executable
ea app.ea -o app          # -> app

# Run tests (109 passing)
cargo test --features=llvm
```

## Call from Python

```python
import ctypes
import numpy as np

lib = ctypes.CDLL("./kernel.so")
lib.fma_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # a
    ctypes.POINTER(ctypes.c_float),  # b
    ctypes.POINTER(ctypes.c_float),  # c
    ctypes.POINTER(ctypes.c_float),  # out
    ctypes.c_int32,                  # len
]
lib.fma_kernel.restype = None

a = np.random.rand(1_000_000).astype(np.float32)
b = np.random.rand(1_000_000).astype(np.float32)
c = np.random.rand(1_000_000).astype(np.float32)
out = np.zeros(1_000_000, dtype=np.float32)

lib.fma_kernel(
    a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    len(a),
)
```

## Architecture

```
Source (.ea) -> Lexer -> Parser -> Type Check -> Codegen (LLVM 18) -> .o / .so
```

~4,600 lines of Rust. No file exceeds 500 lines. Every feature proven by end-to-end test.
109 tests covering C interop, SIMD operations, structs, and shared library output.

## License

Apache 2.0
