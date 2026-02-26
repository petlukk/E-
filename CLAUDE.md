# CLAUDE.md — Eä v2: SIMD Kernel Compiler

## What This Is

**Eä compiles SIMD kernels to object files callable from any language.**

Write readable SIMD code → compile to `.o` or `.so` → call from C, Rust, Python via C ABI. Not a general-purpose language. Not competing with Rust. Competing with ISPC and hand-written intrinsics.

**v1 reference** (read-only, never modify): `/mnt/c/Users/Peter.lukka/Desktop/DEV/EA`
**v2 specification**: `EA_V2_SPECIFICATION.md` in this repo

## Hard Rules

1. **No file exceeds 500 lines.** Split before you hit it.
2. **Every feature proven by end-to-end test** — source in, binary out, correct output.
3. **No fake functions.** If it can't do what its name says, don't write it.
4. **No premature features.** If the current phase doesn't need it, don't build it.
5. **Delete, don't comment.** No `// TODO`, no `// placeholder`, no dead code.
6. **C interop is the product.** Every `export func` must be callable from C correctly.

## Build Commands

```bash
cargo build --features=llvm                    # Build compiler
cargo test --features=llvm                     # End-to-end tests (198 passing)
cargo run --features=llvm -- kernel.ea         # → kernel.o
cargo run --features=llvm -- kernel.ea --lib   # → kernel.so
cargo run --features=llvm -- app.ea -o app     # → linked executable
cargo run --features=llvm -- kernel.ea --emit-asm # → kernel.s
cargo run --features=llvm -- kernel.ea --header   # → kernel.h (alongside .o)
cargo fmt && cargo clippy --all-targets --all-features -- -D warnings
```

## Pipeline

```
Source (.ea) → Lexer → Parser → Type Check → Codegen (inkwell) → .o file (TargetMachine, no llc)
```

Default output is an object file. No JIT. No `.ll` in the default path. Use `--emit-llvm` for IR inspection.

## Architecture

```
src/
├── main.rs                    # CLI
├── lib.rs                     # Pipeline orchestration
├── error.rs                   # Error types
├── target.rs                  # LLVM target machine setup
├── header.rs                  # C header generation (--header)
├── lexer/
│   ├── mod.rs                 # Logos tokenizer
│   └── tokens.rs              # Token utilities
├── ast/
│   └── mod.rs                 # Core AST (Expr, Stmt, types)
├── parser/
│   ├── mod.rs                 # Parser core
│   ├── expressions.rs         # Expression parsing
│   └── statements.rs          # Statement parsing
├── typeck/
│   ├── mod.rs                 # Type checker entry point
│   ├── types.rs               # Type enum + utilities
│   ├── check.rs               # Statement-level type checking
│   ├── expr_check.rs          # Expression type checking
│   ├── intrinsics.rs          # Scalar intrinsic signatures
│   └── intrinsics_simd.rs     # SIMD intrinsic signatures
└── codegen/
    ├── mod.rs                 # Core + module + object emission
    ├── expressions.rs         # Expression compilation
    ├── builtins.rs            # println compilation
    ├── statements.rs          # Statement + function compilation
    ├── structs.rs             # Struct codegen + field access
    ├── simd.rs                # SIMD vector literals + intrinsic dispatch
    ├── simd_arithmetic.rs     # SIMD arithmetic ops (.+, .*, etc.)
    ├── simd_math.rs           # SIMD math (fma, sqrt, reduce, widen)
    └── simd_memory.rs         # SIMD load/store/splat
```

## Phases (strict order)

1. **Hello World + Export** — `export func`, C ABI, object file output, println for standalone
2. **Variables + Arithmetic** — let bindings, i32/f32/f64, operators
3. **Control Flow** — if/else, while
4. **Functions + Pointers** — recursion, `*T` / `*mut T`, pointer indexing, C array interop
5. **Explicit SIMD** — f32x4, `.+` `.* `, load/store/splat, element access
6. **Deep SIMD** — fma, reduce_add/max/min, shuffle, select, f32x8
7. **Structs + Shared Libs** — C-compatible structs, `--lib` → `.so`/`.dll`
8. **Extended Types + Demos** — i8/u8/i16/u16/u32/u64, byte vectors, widening/narrowing, bitwise, sqrt/rsqrt, type conversions, Cornell Box ray tracer, particle life
9. **Compiler Maturity** — fix `--opt-level` pass pipeline, `--emit-asm`, `unroll(N)`, `prefetch(ptr, offset)`, `--header` C header generation
10. **foreach** — `foreach (i in start..end) { ... }` auto-vectorized loops with phi-node codegen, `unroll` + `foreach` composition

## Testing

Two test types, both end-to-end:

**Standalone** (has `main()`): compile → link with libc → execute → check stdout
**C interop** (export only): compile → link with C harness → execute → check stdout

```rust
#[test]
fn test_add_export() {
    let result = compile_and_link_with_c(
        "export func add(a: i32, b: i32) -> i32 { return a + b; }",
        r#"
            #include <stdio.h>
            extern int add(int, int);
            int main() { printf("%d\n", add(3, 4)); return 0; }
        "#,
    );
    assert_eq!(result.stdout.trim(), "7");
}
```

## LLVM

- **LLVM 18**
- `inkwell` with `features = ["llvm18-0"]`
- System: `sudo apt install llvm-18-dev clang-18 libpolly-18-dev libzstd-dev`
- Object files via `TargetMachine::write_to_file()` — no `llc` subprocess

## The Language (kernel subset)

**Types**: `i8`, `u8`, `i16`, `u16`, `i32`, `i64`, `u32`, `u64`, `f32`, `f64`, `bool`, `*T`, `*mut T`, `f32x4`, `f32x8`, `f32x16`, `i32x4`, `i32x8`, `i8x16`, `i8x32`, `u8x16`, `i16x8`, `i16x16`
**No**: strings, collections, references, generics, traits, modules, imports

## What NOT to Build (ever, for a kernel language)

- Package manager, standard library, LSP, JIT
- Compile-time execution engine
- Incremental/parallel/streaming compilation
- Memory profiler, string interner, parser optimizer
- VS Code extension
- Benchmarks against other languages (until Phase 6+)
