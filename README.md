# Eä: SIMD Kernel Compiler

Compile readable SIMD kernels to object files callable from any language.

## What It Does

Eä transforms SIMD kernels written in a purpose-built language into native object files (.o) and shared libraries (.so) that integrate seamlessly with C, Rust, Python, and any language supporting the C ABI. Write once, call from anywhere.

**Currently**: Phase 5 — Explicit SIMD support with vector types and operations
**Focus**: Correctness and interop, not compilation speed

## Why This Matters

Performance-critical code often requires explicit SIMD. Hand-written intrinsics are error-prone and verbose. General-purpose languages struggle to expose vectorization clearly. Eä offers a middle path: a readable, type-safe syntax that compiles directly to optimized SIMD instructions without the overhead of a runtime.

## Use Cases

- Real-time signal processing (audio, video, sensor data)
- Machine learning inference kernels
- Numerical computing libraries
- Game engine physics and rendering
- Computer vision primitives
- Graphics shaders compiled to CPU code

## The Language

A strict subset focused on computational kernels:

**Types**: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`, `bool`, `*T`, `*mut T`, `f32x4`, `i32x4`

**Features** (Phase 5):
- Export functions with C calling convention
- Variables and arithmetic
- Control flow (if/else, while)
- Functions and pointer indexing
- **SIMD vectors**: load, store, splat, element access, masked operations

**Not included**: strings, generics, modules, JIT, garbage collection

## Getting Started

```bash
# Install LLVM 14
sudo apt install llvm-14-dev clang-14

# Build
cargo build --features=llvm

# Compile a kernel
cargo run --features=llvm -- kernel.ea  # Produces kernel.o
cargo run --features=llvm -- kernel.ea --lib  # Produces kernel.so

# Run tests
cargo test --features=llvm
```

## Test Status

All 51 tests passing:

- **Phase 2** (End-to-end): 22 tests — basic types, arithmetic, functions, C interop
- **Phase 3** (Control flow): 17 tests — if/else, while loops, boolean logic
- **Phase 4** (Pointers): 10 tests — pointer arithmetic, array access, recursive functions
- **Phase 5** (SIMD): 12 tests — vector literals, element access, SIMD arithmetic, reduction

```
running 51 tests
test result: ok. 51 passed; 0 failed; 0 ignored
```

## Example

```rust
export func dot_product(a: *f32, b: *f32, n: i32) -> f32 {
    let mut sum: f32 = 0.0;
    let mut i: i32 = 0;
    while i < n {
        let av = load(a + i);
        let bv = load(b + i);
        sum = sum + reduce_add(av .* bv);
        i = i + 4;
    }
    return sum;
}
```

Call from C:

```c
extern float dot_product(float *a, float *b, int n);
float result = dot_product(arr1, arr2, 1000);
```

## Architecture

```
Source → Lexer → Parser → Type Check → Codegen (LLVM) → Object File
```

No JIT, no runtime, no intermediate representation in the standard path.

## Development

Each file stays under 500 lines. Every feature is proven by end-to-end tests. No dead code, no placeholders.

See `CLAUDE.md` for design principles and build commands.
See `EA_V2_SPECIFICATION.md` for the full language spec.

## Roadmap

- Phase 6: Reduction operations (add, max, min) and element shuffles
- Phase 7: Structs and shared library packaging

## License

See LICENSE file.