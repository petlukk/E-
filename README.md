# Ea

**SIMD kernel language for C and Python.**

Write readable SIMD code. Compile to `.o` or `.so`. Call from C, Rust, Python via C ABI.
No runtime. No garbage collector. No unsafe blocks. Just kernels.

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

Compile and call from C:

```bash
ea kernel.ea --lib    # produces kernel.so
```

```c
extern void fma_kernel(const float*, const float*, const float*, float*, int);

fma_kernel(a, b, c, result, n);  // that's it
```

## Benchmark

Measured against hand-optimized C with AVX2 intrinsics (`gcc -O3 -march=native -ffast-math`).
Ea uses strict IEEE floating point -- no fast-math flags.

| Kernel | Ea vs C | Notes |
|---|---|---|
| FMA (f32x4) | ~1.0x | At parity |
| Sum reduction (f32x8) | ~1.0x | At parity |
| Sum reduction (f32x4) | ~0.9x | Ea faster |
| Max reduction (f32x4) | ~0.95x | Ea faster |
| Min reduction (f32x4) | ~1.0x | At parity |

1M elements, 100-200 runs, averaged. AMD Ryzen 7 1700, GCC 11.4, LLVM 14, Linux (WSL2).
Full methodology and benchmark code in `benchmarks/`.

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

## Features

- **SIMD**: `f32x4`, `f32x8`, `i32x4`, `i32x8` with `load`, `store`, `splat`, `fma`, `shuffle`, `select`
- **Reductions**: `reduce_add`, `reduce_max`, `reduce_min`
- **Structs**: C-compatible layout, pointer-to-struct, array-of-structs
- **Pointers**: `*T`, `*mut T`, pointer indexing (`arr[i]`)
- **Types**: `i8`-`i64`, `u8`-`u64`, `f32`, `f64`, `bool`
- **Output**: `.o` object files, `.so`/`.dll` shared libraries, linked executables
- **C ABI**: every `export func` is callable from any language

## Quick Start

```bash
# Requirements: LLVM 14, Rust
sudo apt install llvm-14-dev clang-14

# Build
cargo build --features=llvm

# Compile a kernel to object file
ea kernel.ea              # -> kernel.o

# Compile to shared library
ea kernel.ea --lib        # -> kernel.so

# Compile standalone executable
ea app.ea -o app          # -> app

# Run tests (95 passing)
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
Source (.ea) -> Lexer -> Parser -> Type Check -> Codegen (LLVM 14) -> .o / .so
```

~4,500 lines of Rust. No file exceeds 500 lines. Every feature proven by end-to-end test.
95 tests covering C interop, SIMD operations, structs, and shared library output.

## License

Apache 2.0
