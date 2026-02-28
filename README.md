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

## v1.5 — Multi-kernel files, `static_assert`, `ea inspect`

**Multi-kernel files** — multiple structs, constants, helper functions, and exported kernels in a single `.ea` file. The full pipeline (parser, desugarer, type checker, codegen, metadata, header, all five binding generators) handles everything seamlessly. No special syntax needed — just write multiple exports.

```
struct Vec2 { x: f32, y: f32 }
const SCALE: f32 = 2.0

export kernel add(a: *f32, b: *f32, out: *mut f32) over i in n step 8 { ... }
export kernel mul(a: *f32, b: *f32, out: *mut f32) over i in n step 8 { ... }
export func dot(a: *f32, b: *f32, n: i32) -> f32 { ... }
```

**`static_assert`** — compile-time assertions evaluated during type checking. No code emitted.

```
const STEP: i32 = 8
static_assert(STEP % 4 == 0, "STEP must be SIMD-aligned")
static_assert(STEP > 0 && STEP <= 16, "STEP must be in range 1..16")
```

Supports arithmetic (`+`, `-`, `*`, `/`, `%`), comparisons (`==`, `!=`, `<`, `>`, `<=`, `>=`), and boolean logic (`&&`, `||`, `!`) on compile-time constants. Non-constant references produce clear errors.

**`ea inspect`** — analyze post-optimization instruction mix, loops, vector width, and register usage.

```bash
ea inspect kernel.ea                  # all exports, native target
ea inspect kernel.ea --avx512         # with AVX-512
ea inspect kernel.ea --target=skylake # specific CPU
```

```
=== vscale (exported) ===
  vector instructions:  12
  scalar instructions:   4
  vector width:         256-bit (f32x8)
  loops:                2 (1 main, 1 tail)
  vector registers:     ymm0, ymm1, ymm2, ymm3 (4 used)
```

## v1.4 — Output annotations

Mark `*mut` pointer parameters as outputs with buffer sizing hints. Binding generators auto-allocate and return buffers, eliminating the allocate-call-unpack pattern from host code.

```
export func transform(data: *f32, out result: *mut f32 [cap: n], n: i32) {
    let mut i: i32 = 0
    while i < n {
        result[i] = data[i] * 2.0
        i = i + 1
    }
}
```

Three forms:

| Syntax | Behavior |
|--------|----------|
| `out result: *mut f32 [cap: n]` | Auto-allocated by binding, returned to caller |
| `out result: *mut f32 [cap: n, count: actual]` | Auto-allocated with separate actual-length path |
| `out result: *mut f32` | Caller provides buffer (stays in signature) |

Generated bindings handle allocation per target:

| Target | Auto-allocation | Return type |
|--------|----------------|-------------|
| Python | `np.empty(n, dtype=np.float32)` | `np.ndarray` |
| Rust | `vec![Default::default(); n]` | `Vec<f32>` |
| C++ | `std::vector<float>(n)` | `std::vector<float>` |
| PyTorch | `torch.empty(n, dtype=torch.float32)` | `Tensor` |

Type checker validates: `out` requires `*mut` pointer type, cap identifiers must reference preceding input params or constants. Metadata JSON emits `direction`, `cap`, and `count` fields per arg. Backward-compatible: old JSON without these fields works unchanged.

## v1.3 — Kernel construct, compile-time constants, tail strategies

**`kernel`** — declarative loop construct for data-parallel operations:

```
export kernel double_it(data: *i32, out: *mut i32)
    over i in n step 1
{
    out[i] = data[i] * 2
}
```

Desugars to a function with a generated loop. The range bound (`n`) becomes the last parameter automatically. SIMD kernels use `step 4`/`step 8` for explicit vectorization.

**`tail`** — handle remainder elements when array length isn't a multiple of step:

```
export kernel vscale(data: *f32, out: *mut f32, factor: f32)
    over i in n step 8
    tail scalar { out[i] = data[i] * factor }
{
    store(out, i, load(data, i) .* splat(factor))
}
```

Three strategies: `tail scalar { ... }` (element-wise loop), `tail mask { ... }` (single masked iteration), `tail pad` (skip remainder).

**`const`** — compile-time constants inlined at every use site:

```
const BLOCK_SIZE: i32 = 64
const PI: f64 = 3.14159265358979
```

Supports integer and float types. Constants are validated at type-check time and referenced in kernel bodies, cap expressions, and function parameters.

## v1.2 — `ea bind` multi-language bindings

**`ea bind`** now generates native bindings for five targets from a single kernel:

| Flag | Output | What you get |
|------|--------|--------------|
| `--python` | `kernel.py` | NumPy ctypes module with dtype checks, length collapsing |
| `--rust` | `kernel.rs` | `extern "C"` FFI + safe wrappers with `&[T]`/`&mut [T]` |
| `--pytorch` | `kernel_torch.py` | `torch.autograd.Function` per export, tensor contiguity/device checks |
| `--cpp` | `kernel.hpp` | `namespace ea`, `extern "C"` declarations, `std::span` overloads |
| `--cmake` | `CMakeLists.txt` + `EaCompiler.cmake` | Ready-to-build CMake project skeleton |

All generators share a common JSON parser (`bind_common.rs`) and the same length-collapsing heuristic: parameters named `n`/`len`/`length`/`count`/`size`/`num` after a pointer arg are auto-filled from the slice/array/tensor size.

## v1.1 — ARM/NEON support, integration examples, CI

**ARM/AArch64 cross-compilation** — compile kernels for ARM targets with NEON (128-bit) SIMD:

```bash
ea kernel.ea --lib --target=aarch64   # produces kernel.so for ARM
```

The compiler validates vector widths at the type-check level: 128-bit types (`f32x4`, `i32x4`, `u8x16`, `i16x8`) work on ARM; 256-bit+ types (`f32x8`, `i32x8`) and x86-specific intrinsics (`maddubs`, `gather`, `scatter`) produce clear error messages with alternatives.

**Integration examples** — manual integration patterns for embedding Eä kernels into host projects. Most are now superseded by `ea bind`; see [FFmpeg filter](integrations/ffmpeg-filter/) for the remaining manual example.

**CI** — build and test on Linux x86_64, Linux ARM (aarch64), and Windows on every push.

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
| [Eastat](demo/eastat/)                         | CSV analytics        | Structural scan, SIMD reduction                 | ~2x faster than pandas (comparable work) — honest phase breakdown, zero manual ctypes via `ea bind` |

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

Tested on x86-64 (AVX2) and AArch64 (NEON). CI runs on both architectures plus Windows.

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

## Call from your language

### Python (auto-generated)

```bash
ea kernel.ea --lib && ea bind kernel.ea --python
```

```python
import numpy as np, kernel

a = np.random.rand(1_000_000).astype(np.float32)
b = np.random.rand(1_000_000).astype(np.float32)
c = np.random.rand(1_000_000).astype(np.float32)

# With output annotations: result auto-allocated and returned
result = kernel.fma_kernel(a, b, c)  # len auto-filled, dtype checked, output returned

# Without output annotations: caller provides buffer
out = np.zeros(1_000_000, dtype=np.float32)
kernel.fma_kernel(a, b, c, out)      # len auto-filled from array size
```

### PyTorch (auto-generated)

```bash
ea kernel.ea --lib && ea bind kernel.ea --pytorch
```

```python
import torch, kernel_torch

data = torch.randn(1_000_000)
result = kernel_torch.scale(data, 2.0)  # autograd-compatible, CPU tensors
```

### Rust (auto-generated)

```bash
ea kernel.ea --lib && ea bind kernel.ea --rust
```

```rust
// include the generated kernel.rs
let mut data = vec![1.0f32; 1_000_000];
kernel::scale(&mut data, 2.0);  // safe wrapper, length from slice
```

### C++ (auto-generated)

```bash
ea kernel.ea --lib && ea bind kernel.ea --cpp
```

```cpp
#include "kernel.hpp"
std::vector<float> data(1'000'000, 1.0f);
ea::scale(data, 2.0f);  // std::span overload, length from .size()
```

### Manual ctypes (for custom control)

```python
import ctypes, numpy as np

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
Source (.ea) -> Lexer -> Parser -> Desugar (kernel→func) -> Type Check -> Codegen (LLVM 18) -> .o / .so
                                                                                             -> .ea.json -> ea bind -> .py / .rs / .hpp / _torch.py / CMakeLists.txt
```

~9,700 lines of Rust. Every feature proven by end-to-end test.
356 tests covering C interop, SIMD operations, structs, integer types, shared library output, foreach loops, short-circuit evaluation, error diagnostics, masked operations, scatter/gather, ARM target validation, compiler flags, kernel construct, tail strategies, compile-time constants, output annotations, and binding generation for all five targets.

## License

Apache 2.0
