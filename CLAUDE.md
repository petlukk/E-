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
cargo test --features=llvm                     # End-to-end tests
cargo run --features=llvm -- kernel.ea         # → kernel.o
cargo run --features=llvm -- kernel.ea --lib   # → kernel.so
cargo run --features=llvm -- app.ea -o app     # → linked executable
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
├── main.rs                 # CLI
├── lib.rs                  # Pipeline orchestration
├── error.rs                # Error types
├── target.rs               # LLVM target machine setup
├── lexer/
│   ├── mod.rs              # Logos tokenizer
│   └── tokens.rs           # Token utilities
├── ast/
│   ├── mod.rs              # Core AST (Expr, Stmt, types)
│   └── simd.rs             # SIMD AST nodes (Phase 5+)
├── parser/
│   ├── mod.rs              # Parser core + expressions
│   └── statements.rs       # Statement parsing
├── typeck/
│   └── mod.rs              # Type checker
└── codegen/
    ├── mod.rs              # Core + module + object emission
    ├── expressions.rs      # Expression compilation
    ├── statements.rs       # Statement compilation
    ├── functions.rs        # Function compilation + C ABI
    └── simd.rs             # SIMD codegen (Phase 5+)
```

Files get created as needed per phase. Not all exist from day one.

## Phases (strict order)

1. **Hello World + Export** — `export func`, C ABI, object file output, println for standalone
2. **Variables + Arithmetic** — let bindings, i32/f32/f64, operators
3. **Control Flow** — if/else, while
4. **Functions + Pointers** — recursion, `*T` / `*mut T`, pointer indexing, C array interop
5. **Explicit SIMD** — f32x4, `.+` `.* `, load/store/splat, element access
6. **Deep SIMD** — fma, reduce_add/max/min, shuffle, select, f32x8
7. **Structs + Shared Libs** — C-compatible structs, `--lib` → `.so`/`.dll`

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

- **LLVM 14 exactly** (not 15+)
- `inkwell` with `features = ["llvm14-0"]`
- System: `sudo apt install llvm-14-dev clang-14`
- Object files via `TargetMachine::write_to_file()` — no `llc` subprocess

## The Language (kernel subset)

**Types**: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`, `bool`, `*T`, `*mut T`, `f32x4`, `i32x4`
**No**: strings, collections, references, generics, traits, modules, imports

## What NOT to Build (ever, for a kernel language)

- Package manager, standard library, LSP, JIT
- Compile-time execution engine
- Incremental/parallel/streaming compilation
- Memory profiler, string interner, parser optimizer
- VS Code extension
- Benchmarks against other languages (until Phase 6+)
