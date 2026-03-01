# Eä

**Write compute accelerators. Call them from any language.**

Write a kernel once in clean, explicit syntax. `ea bind` generates native bindings for Python, Rust, C++, PyTorch, and CMake. No runtime, no garbage collector, no glue code by hand.

> **[Ea Showcase](https://github.com/petlukk/Ea_showcase)** — visual demo application showing Ea kernels running live.

## Example

```
export kernel vscale(data: *f32, out result: *mut f32 [cap: n], factor: f32)
    over i in n step 8
    tail scalar { result[i] = data[i] * factor }
{
    let v: f32x8 = load(data, i)
    store(result, i, v .* splat(factor))
}
```

Compile, bind, use:

```bash
ea kernel.ea --lib                          # -> kernel.so + kernel.ea.json
ea bind kernel.ea --python --rust --cpp     # -> kernel.py, kernel.rs, kernel.hpp
```

```python
import numpy as np, kernel
data = np.random.rand(1_000_000).astype(np.float32)
result = kernel.vscale(data, 2.0)  # output auto-allocated, len auto-filled, dtype checked
```

```rust
// kernel.rs (generated)
let result = kernel::vscale(&data, 2.0); // Vec<f32> returned, length from slice
```

```cpp
// kernel.hpp (generated)
auto result = ea::vscale(data_span, 2.0f);  // std::vector returned
```

The kernel is the same. The language boundary disappears.

## `ea bind`

One kernel, five targets:

```bash
ea bind kernel.ea --python    # -> kernel.py         (NumPy + ctypes)
ea bind kernel.ea --rust      # -> kernel.rs         (FFI + safe wrappers)
ea bind kernel.ea --pytorch   # -> kernel_torch.py   (autograd.Function)
ea bind kernel.ea --cpp       # -> kernel.hpp        (std::span + extern "C")
ea bind kernel.ea --cmake     # -> CMakeLists.txt + EaCompiler.cmake
```

Multiple flags in one invocation: `ea bind kernel.ea --python --rust --cpp`

Each generator reads `kernel.ea.json` (emitted by `--lib`) and produces idiomatic glue for the target ecosystem. Pointer args become slices/arrays/tensors. Length params are collapsed automatically. Types are checked at the boundary.

## Quick Start

```bash
# Requirements: LLVM 18, Rust
sudo apt install llvm-18-dev clang-18 libpolly-18-dev libzstd-dev

# Build
cargo build --features=llvm

# Compile a kernel to shared library (+ JSON metadata)
ea kernel.ea --lib        # -> kernel.so + kernel.ea.json

# Generate bindings for your language
ea bind kernel.ea --python              # -> kernel.py
ea bind kernel.ea --rust                # -> kernel.rs
ea bind kernel.ea --cpp                 # -> kernel.hpp
ea bind kernel.ea --pytorch             # -> kernel_torch.py
ea bind kernel.ea --cmake               # -> CMakeLists.txt + EaCompiler.cmake
ea bind kernel.ea --python --rust --cpp # -> all three at once

# Or compile to object file / executable
ea kernel.ea              # -> kernel.o
ea app.ea -o app          # -> app

# Run tests (356 passing)
cargo test --features=llvm
```

## Design Principles

- **Explicit over implicit** — SIMD width, loop stepping, and memory access are programmer-controlled
- **Predictable performance over abstraction** — no hidden allocations, no auto-vectorizer surprises
- **Write once, bind everywhere** — one kernel source, native bindings for each host language
- **Zero runtime cost** — no garbage collector, no runtime, no hidden checks

## Non-goals

- Not a general-purpose language — no strings, collections, or modules
- No safety guarantees — correctness is the programmer's responsibility
- No auto-vectorization in the default path — SIMD width is always explicit (`foreach` relies on LLVM, but explicit vector types are the primary path)
- Not intended to replace Rust, C++, or any host language — intended to accelerate them

## Compute Model

Seven kernel patterns — streaming, reduction, stencil, streaming dataset, fused
pipeline, quantized inference, structural scan. See [`COMPUTE.md`](COMPUTE.md) for
the full model and [`COMPUTE_PATTERNS.md`](COMPUTE_PATTERNS.md) for measured analysis
of when each pattern wins and when it doesn't.

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
- **Kernels**: `export kernel name(...) over i in n step N { ... }` — declarative data-parallel loops with automatic range bound parameter
- **Tail strategies**: `tail scalar { ... }`, `tail mask { ... }`, `tail pad` — handle SIMD remainder elements
- **Output annotations**: `out name: *mut T [cap: expr]` — mark output params for auto-allocation in bindings
- **Compile-time constants**: `const NAME: TYPE = LITERAL` — inlined at every use site
- **Static assertions**: `static_assert(condition, "message")` — compile-time validation of constants
- **Multi-kernel files**: multiple exports, shared structs, shared constants in one `.ea` file
- **foreach**: `foreach (i in 0..n) { ... }` — element-wise loops (LLVM may auto-vectorize at O2+)
- **unroll(N)**: loop unrolling hint for `while` and `foreach`
- **prefetch**: `prefetch(ptr, offset)` — software prefetch for large-array streaming
- **Output**: `.o` object files, `.so`/`.dll` shared libraries, linked executables
- **C ABI**: every `export func` is callable from any language
- **Tooling**: `--header` (C header generation), `--emit-asm` (assembly output), `--emit-llvm` (IR output), `ea inspect` (post-optimization analysis)
- **`ea bind`**: auto-generated bindings for Python/NumPy, Rust, C++/std::span, PyTorch/autograd, CMake
- **Masked memory**: `load_masked`, `store_masked` for safe SIMD tail handling
- **Scatter/Gather**: `gather(ptr, indices)`, `scatter(ptr, indices, values)` (scatter requires `--avx512`)
- **Restrict pointers**: `*restrict T`, `*mut restrict T` for alias-free optimization
- **AVX-512**: `f32x16` via `--avx512` flag
- **ARM/NEON**: Cross-compile to AArch64 with `--target=aarch64` (128-bit NEON SIMD: f32x4, i32x4, u8x16, i8x16, i16x8)

Tested on x86-64 (AVX2) and AArch64 (NEON). CI runs on both architectures plus Windows. See [`CHANGELOG.md`](CHANGELOG.md) for version history.

## Demos

Real workloads. Real data. Verified against established tools.

| Demo                                           | Domain               | Patterns                                        | Result                                                                     |
| ---------------------------------------------- | -------------------- | ----------------------------------------------- | -------------------------------------------------------------------------- |
| [Sobel edge detection](demo/sobel/)            | Image processing     | Stencil, pipeline                               | 12.0x faster than NumPy, 4.2x faster than OpenCV                           |
| [Video anomaly detection](demo/video_anomaly/) | Video analysis       | Streaming, fused pipeline                       | 3 kernels: **0.96x (slower)**. Fused: **11.5x faster**                     |
| [Astronomy stacking](demo/astro_stack/)        | Scientific computing | Streaming dataset                               | 2.6x faster, 16x less memory than NumPy                                    |
| [MNIST preprocessing](demo/mnist_normalize/)   | ML preprocessing     | Streaming, fused pipeline                       | Single op: **1.0x (memory-bound)**. Fused pipeline: **4.0x faster**         |
| [Pixel pipeline](demo/pixel_pipeline/)         | Image processing     | u8x16 threshold, u8→f32 widen                   | threshold: **20.7x**, normalize: **2.1x** vs NumPy                         |
| [Conv2d (dot/1d)](demo/conv2d/)                | Integer SIMD         | maddubs_i16, u8×i8                              | dot: **5.6x**, conv1d: **3.3x** vs NumPy                                   |
| [Conv2d 3×3 NHWC](demo/conv2d_3x3/)            | Quantized inference  | maddubs_i16 dual-acc / maddubs_i32 safe variant | **57.3x vs NumPy**, 44.6 GMACs/s on 56×56×64                               |
| [Pipeline fusion](demo/skimage_fusion/)        | Image processing     | Stencil fusion, algebraic optimization          | 5.4x vs NumPy, **1.28x fusion at 4K**, 7x memory reduction                 |
| [Tokenizer prepass](demo/tokenizer_prepass/)   | Text/NLP             | Structural scan, bitwise ops                    | unfused: **65.5x**, fused: **56.1x** vs NumPy (fusion: 0.74x — see README) |
| [Particle update](demo/particles/)             | Struct FFI           | C-compatible structs over FFI                   | Correctness demo — proves struct layout matches C exactly                  |
| [Cornell Box ray tracer](demo/cornell_box/)    | Graphics             | Struct return, recursion, scalar math           | First non-SIMD demo: full ray tracer in 245 lines of Eä                    |
| [Particle life](demo/particle_life/)           | Simulation           | N-body scalar, fused vs unfused                 | Matches hand-written C at clang-18 -O2. Interactive pygame UI              |
| [Eastat](demo/eastat/)                         | CSV analytics        | Structural scan, SIMD reduction                 | 0.5x (slower than pandas) — honest phase breakdown, zero manual ctypes via `ea bind` |

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
Ea (3 kernels)      :  1.58 ms   (0.96x — slightly slower due to FFI + memory overhead)
Ea fused (1 kernel) :  0.13 ms   (11.5x faster than NumPy, 9.5x faster than OpenCV)
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
  768x512   →  0.97x fusion speedup   (fits in L3 cache)
 3840x2160  →  1.28x fusion speedup   (intermediates spill to DRAM)
```

Same language. Same compiler. The compute formulation changed.

> **If data leaves registers, you probably ended a kernel too early.**

> **Fusion does not make bad kernels fast. Fusion amplifies good kernel design.**

See [`COMPUTE_PATTERNS.md`](COMPUTE_PATTERNS.md) for the full analysis of all
compute classes, including when Ea wins, when it doesn't, and when fusion hurts.

## Integrations

For Python, Rust, C++, PyTorch, and CMake — `ea bind` generates the glue. See [`ea bind`](#ea-bind) above.

For ecosystems that need manual integration (custom build systems, embedding into larger C projects), [`integrations/`](integrations/) has a reference example:

| Example | Ecosystem | What it shows |
|---------|-----------|---------------|
| [FFmpeg filter](integrations/ffmpeg-filter/) | Video/C | libav* decode + Ea kernel per scanline — realistic C embed pattern |

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

## Architecture

```
Source (.ea) -> Lexer -> Parser -> Desugar (kernel→func) -> Type Check -> Codegen (LLVM 18) -> .o / .so
                                                                                             -> .ea.json -> ea bind -> .py / .rs / .hpp / _torch.py / CMakeLists.txt
```

~10,300 lines of Rust. Every feature proven by end-to-end test.
375 tests covering C interop, SIMD operations, structs, integer types, shared library output, foreach loops, short-circuit evaluation, error diagnostics, masked operations, scatter/gather, ARM target validation, compiler flags, kernel construct, tail strategies, compile-time constants, output annotations, and binding generation for all five targets.

## License

Apache 2.0
