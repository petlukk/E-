# Changelog

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

## v0.4.0

### Breaking Changes
- `maddubs(u8x16, i8x16) -> i16x8` renamed to `maddubs_i16` — update all kernels

### New Intrinsics
- `maddubs_i32(u8x16, i8x16) -> i32x4` — safe accumulation via pmaddubsw+pmaddwd chain.
  Programmer explicitly chooses the overflow model by choosing the instruction.
  No silent widening.

### Demos
- `demo/conv2d_3x3/conv_safe.ea` — i32x4 accumulator variant, immune to accumulator overflow
