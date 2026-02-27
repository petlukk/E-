# Rust build.rs — Eä kernel in a Cargo project

This example shows how to call an Eä SIMD kernel from Rust using
`build.rs` to compile and link the kernel at build time.

## How it works

`build.rs`:
1. Runs `ea kernel.ea` to produce `kernel.o`
2. Wraps it in `libkernel.a` via `ar rcs`
3. Sets `cargo:rustc-link-search` and `cargo:rustc-link-lib=static=kernel`
4. Uses `cargo:rerun-if-changed=kernel.ea` so editing the kernel triggers recompilation

`src/main.rs` declares the kernel with `extern "C"` and calls it through
Rust's FFI:

```rust
extern "C" {
    fn dot_product(a: *const f32, b: *const f32, n: i32) -> f32;
}
```

## Prerequisites

- `ea` compiler on PATH
- Rust toolchain (cargo)
- `ar` (standard on Linux/macOS)

## Usage

```bash
cargo run
```

Output:
```
dot product: 72
verified: matches Rust computation
```

## Files

| File | Purpose |
|------|---------|
| `kernel.ea` | Dot product kernel (f32x4, reduce_add) |
| `build.rs` | Compiles .ea and creates static lib |
| `src/main.rs` | Rust program with extern "C" FFI |
| `Cargo.toml` | Crate manifest |
