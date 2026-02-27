# Eä

**SIMD kernel language for C and Python.**

Write SIMD kernels in clean, minimal syntax. Compile to `.o` or `.so`. Call from C, Rust, Python via C ABI.
No runtime. No garbage collector. No hidden performance cliffs.

> **[Ea Showcase](https://github.com/petlukk/Ea_showcase)** — visual demo application showing Ea kernels running live.

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

## Design Principles

- **Explicit over implicit** — SIMD width, loop stepping, and memory access are programmer-controlled
- **Predictable performance over abstraction** — no hidden allocations, no auto-vectorizer surprises
- **Kernel isolation over language integration** — compute kernels are compiled separately, called via C ABI
- **Zero runtime cost** — no garbage collector, no runtime, no hidden checks

## Non-goals

- Not a general-purpose language — no strings, collections, or modules
- No safety guarantees — correctness is the programmer's responsibility
- No auto-vectorization in the default path — SIMD width is always explicit (`foreach` relies on LLVM, but explicit vector types are the primary path)
- Not intended to replace Rust, C++, or any host language

## Compute Model

Seven kernel patterns — streaming, reduction, stencil, streaming dataset, fused
pipeline, quantized inference, structural scan. See [`COMPUTE.md`](COMPUTE.md) for
the full model and [`COMPUTE_PATTERNS.md`](COMPUTE_PATTERNS.md) for measured analysis
of when each pattern wins and when it doesn't.

## v1.0 — error diagnostics, masked ops, scatter/gather

**`foreach`** — auto-vectorized element-wise loops with phi-node codegen:

```
export func scale(data: *f32, out: *mut f32, n: i32, factor: f32) {
    foreach (i in 0..n) {
        out[i] = data[i] * factor
    }
}
```

`foreach` generates a scalar loop with phi nodes. LLVM may auto-vectorize at `-O2+`.
For guaranteed SIMD width, use explicit `load`/`store` with `f32x4`/`f32x8`.

**`unroll(N)`** — hint to unroll the following loop:

```
unroll(4) foreach (i in 0..n) { out[i] = data[i] * factor }
unroll(4) while i < n { ... }
```

Relies on LLVM unrolling heuristics. Not a hard guarantee.

**`prefetch(ptr, offset)`** — software prefetch hint for large-array streaming:

```
prefetch(data, i + 16)
```

**`--header`** — generate a C header alongside the object file:

```bash
ea kernel.ea --header    # produces kernel.o + kernel.h
```

```c
// kernel.h (generated)
#ifndef KERNEL_H
#define KERNEL_H
#include <stdint.h>
void scale(const float* data, float* out, int32_t n, float factor);
#endif
```

**`--emit-asm`** — emit assembly for inspection:

```bash
ea kernel.ea --emit-asm  # produces kernel.s
```

## Demos

Real workloads. Real data. Verified against established tools.

| Demo                                           | Domain               | Patterns                                        | Result                                                                     |
| ---------------------------------------------- | -------------------- | ----------------------------------------------- | -------------------------------------------------------------------------- |
| [Sobel edge detection](demo/sobel/)            | Image processing     | Stencil, pipeline                               | 9.3x faster than NumPy, 2.7x faster than OpenCV                            |
| [Video anomaly detection](demo/video_anomaly/) | Video analysis       | Streaming, fused pipeline                       | 3 kernels: **0.98x (slower)**. Fused: **13x faster**                       |
| [Astronomy stacking](demo/astro_stack/)        | Scientific computing | Streaming dataset                               | 6.4x faster, 16x less memory than NumPy                                    |
| [MNIST preprocessing](demo/mnist_normalize/)   | ML preprocessing     | Streaming, fused pipeline                       | Single op: **0.9x (slower)**. Fused pipeline: **2.6x faster**              |
| [Pixel pipeline](demo/pixel_pipeline/)         | Image processing     | u8x16 threshold, u8→f32 widen                   | threshold: **22x**, normalize: **2.1x** vs NumPy                           |
| [Conv2d (dot/1d)](demo/conv2d/)                | Integer SIMD         | maddubs_i16, u8×i8                              | dot: **5.9x**, conv1d: **3.0x** vs NumPy                                   |
| [Conv2d 3×3 NHWC](demo/conv2d_3x3/)            | Quantized inference  | maddubs_i16 dual-acc / maddubs_i32 safe variant | **48x vs NumPy**, 38 GMACs/s on 56×56×64                                   |
| [Pipeline fusion](demo/skimage_fusion/)        | Image processing     | Stencil fusion, algebraic optimization          | 6.2x vs NumPy, **1.3x fusion at 4K**, 7x memory reduction                  |
| [Tokenizer prepass](demo/tokenizer_prepass/)   | Text/NLP             | Structural scan, bitwise ops                    | unfused: **78.7x**, fused: **58.1x** vs NumPy (fusion: 0.74x — see README) |
| [Particle update](demo/particles/)             | Struct FFI           | C-compatible structs over FFI                   | Correctness demo — proves struct layout matches C exactly                  |
| [Cornell Box ray tracer](demo/cornell_box/)    | Graphics             | Struct return, recursion, scalar math           | First non-SIMD demo: full ray tracer in 245 lines of Eä                    |
| [Particle life](demo/particle_life/)           | Simulation           | N-body scalar, fused vs unfused                 | Matches hand-written C at -O2. Interactive pygame UI                       |

Each demo compiles an Ea kernel to `.so`, calls it from Python via ctypes,
and benchmarks against NumPy and OpenCV. Run `python run.py` in any demo directory.

**Methodology:** all speedup numbers are warm-cache medians (50 runs after 5 warmup).
Where cold-cache numbers differ materially they are noted. See [`AUDIT_v0.3.0.md`](AUDIT_v0.3.0.md)
for the full integrity audit: assembly verification, cold-cache analysis, honest loss
accounting, i16 overflow constraint, and cross-machine results.

### Kernel fusion

**Streaming fusion** — the video anomaly demo ships both unfused (3-kernel) and
fused (1-kernel) implementations. Same language. Same compiler. Same data.

```
Ea (3 kernels)      :  1.12 ms   (0.98x — slightly slower due to FFI + memory overhead)
Ea fused (1 kernel) :  0.08 ms   (13x faster than NumPy, 12x faster than OpenCV)
```

The MNIST scaling experiment confirms this scales linearly with pipeline depth:

```
1 op  →   2.0x    Ea time: 39 ms (constant)
2 ops →   4.0x    NumPy time: scales linearly
4 ops →  12.0x    Each extra NumPy op = +125 ms (full RAM roundtrip)
8 ops →  25.2x    Each extra Ea op = ~0 ms (SIMD register instruction)
```

**Stencil fusion** — the pipeline fusion demo fuses Gaussian blur + Sobel +
threshold into a single 5x5 stencil. The first attempt was _slower_ than
unfused — naive composition computed 8 redundant Gaussian blurs per output
pixel. Algebraic reformulation (precomputing the combined convolution as a
separable 5x5 kernel) reduced ops from ~120 to ~50 and made fusion win:

```
  768x512   →  1.02x fusion speedup   (fits in L3 cache)
 3840x2160  →  1.33x fusion speedup   (intermediates spill to DRAM)
```

Same language. Same compiler. The compute formulation changed.

> **If data leaves registers, you probably ended a kernel too early.**

> **Fusion does not make bad kernels fast. Fusion amplifies good kernel design.**

See [`COMPUTE_PATTERNS.md`](COMPUTE_PATTERNS.md) for the full analysis of all
compute classes, including when Ea wins, when it doesn't, and when fusion hurts.

## Benchmarks

In tested kernels, Ea reaches performance comparable to hand-written C intrinsics
on FMA and reduction workloads, using strict IEEE floating point (no fast-math).
See [`BENCHMARKS.md`](BENCHMARKS.md) for full tables (AMD Ryzen 7, Intel i7),
restrict analysis, and ILP methodology.

## Relation to existing approaches

**C with intrinsics** —
Works, but `_mm256_fmadd_ps(_mm256_loadu_ps(&a[i]), ...)` is noisy and error-prone.
Ea compiles `fma(load(a, i), load(b, i), load(c, i))` to the same instructions — no casts, no prefixes, no headers.

**Rust with `std::simd`** —
`std::simd` is nightly-only and Rust's type system adds friction for kernel code.
Ea is purpose-built: no lifetimes, no borrows, no generics.

**ISPC** —
ISPC auto-vectorizes scalar code. Ea gives explicit control over vector width and operations.
Different philosophy — Ea is closer to portable intrinsics than an auto-vectorizer.

## Safety Model

Ea provides explicit pointer-based memory access similar to C.
There are no bounds checks or runtime safety guarantees -- correctness and
memory safety are the programmer's responsibility. This is intentional:
kernel code needs predictable performance without hidden checks.

## Features

- **SIMD**: `f32x4`, `f32x8`, `f32x16`, `i32x4`, `i32x8`, `i8x16`, `i8x32`, `u8x16`, `i16x8`, `i16x16` with `load`, `store`, `splat`, `fma`, `shuffle`, `select`
- **Vector bitwise**: `.&` (AND), `.|` (OR), `.^` (XOR) on integer vector types
- **Reductions**: `reduce_add`, `reduce_max`, `reduce_min`
- **Integer SIMD**: `maddubs_i16(u8x16, i8x16) -> i16x8` (SSSE3 pmaddubsw — 16 pairs/cycle, fast/wrapping); `maddubs_i32(u8x16, i8x16) -> i32x4` (pmaddubsw+pmaddwd — safe i32 accumulation)
- **Widening/narrowing**: `widen_u8_f32x4`, `widen_i8_f32x4`, `narrow_f32x4_i8`
- **Math**: `sqrt(x)`, `rsqrt(x)` for scalar and vector float types
- **Type conversions**: `to_f32(x)`, `to_f64(x)`, `to_i32(x)`, `to_i64(x)`
- **Unary negation**: `-x` on numeric types and vectors
- **Structs**: C-compatible layout, pointer-to-struct, array-of-structs
- **Pointers**: `*T`, `*mut T`, pointer indexing (`arr[i]`)
- **Literals**: decimal (`255`), hex (`0xFF`), binary (`0b11110000`)
- **Control flow**: `if`/`else if`/`else`, `while`, short-circuit `&&`/`||`
- **Types**: `i8`, `u8`, `i16`, `u16`, `i32`, `i64`, `u32`, `u64`, `f32`, `f64`, `bool`
- **foreach**: `foreach (i in 0..n) { ... }` — element-wise loops (LLVM may auto-vectorize at O2+)
- **unroll(N)**: loop unrolling hint for `while` and `foreach`
- **prefetch**: `prefetch(ptr, offset)` — software prefetch for large-array streaming
- **Output**: `.o` object files, `.so`/`.dll` shared libraries, linked executables
- **C ABI**: every `export func` is callable from any language
- **Tooling**: `--header` (C header generation), `--emit-asm` (assembly output), `--emit-llvm` (IR output)
- **Masked memory**: `load_masked`, `store_masked` for safe SIMD tail handling
- **Scatter/Gather**: `gather(ptr, indices)`, `scatter(ptr, indices, values)` (scatter requires `--avx512`)
- **Restrict pointers**: `*restrict T`, `*mut restrict T` for alias-free optimization
- **AVX-512**: `f32x16` via `--avx512` flag

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

# Run tests (247 passing)
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

~7,300 lines of Rust. No file exceeds 500 lines. Every feature proven by end-to-end test.
247 tests covering C interop, SIMD operations, structs, integer types, shared library output, foreach loops, short-circuit evaluation, error diagnostics, masked operations, scatter/gather, and compiler flags.

## License

Apache 2.0
