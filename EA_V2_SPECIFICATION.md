# Eä v2 — SIMD Kernel Language Specification

## What Eä Is

**Eä is the easiest way to write fast SIMD code that works with your existing project.**

Write SIMD kernels in clean syntax. Compile them to `.o` or `.so` files. Call them from C, Rust, Python, or any language with C FFI. No need to learn a new ecosystem — Eä drops into your existing build.

```ea
export func dot_product(a: &[f32], b: &[f32], len: i32) -> f32 {
    let sum = splat(0.0)f32x4;
    let i: i32 = 0;
    while i + 4 <= len {
        let va = load(a, i)f32x4;
        let vb = load(b, i)f32x4;
        sum = sum .+ (va .* vb);
        i = i + 4;
    }
    return reduce_add(sum);
}
```

```c
// caller.c — links against the compiled .o
extern float dot_product(const float* a, const float* b, int len);

int main() {
    float a[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[] = {8, 7, 6, 5, 4, 3, 2, 1};
    printf("dot = %f\n", dot_product(a, b, 8));
}
```

**The pitch**: ISPC-style power, modern readable syntax, zero adoption cost.

---

## Part 1: Post-Mortem — What Went Wrong in v1

### Root Causes of Project Halt

1. **No real end-to-end pipeline**: We emitted LLVM IR but compared it against compiled Rust/C binaries as if we had a complete compiler. We didn't.

2. **God files**: `codegen/mod.rs` reached **12,544 lines**. `type_system/mod.rs` hit **4,555 lines**. `parser/mod.rs` hit **3,975 lines**.

3. **Premature features**: Package manager (2,354 lines), LSP server (1,015 lines), compile-time execution engine (1,754 lines), advanced SIMD (2,276 lines) — all before the core pipeline worked.

4. **Fake code**: `comptime/mod.rs` had 14 "algorithm implementations" that were just structs with hardcoded values. `package/mod.rs` had a dependency resolver for a language with no imports.

5. **Confused stdlib**: Rust-side stdlib files were Rust code that EA programs couldn't call. The actual stdlib was C runtime files. Two layers doing different things.

6. **Test inflation**: 158 "passing tests" mostly tested internal struct creation, not end-to-end compilation.

7. **Wrong scope**: Trying to be a general-purpose systems language competing with Rust/Zig — a fight that can't be won by a small project.

### What Was Actually Good

- **Lexer**: Clean, logos-based. Reusable structure.
- **AST design**: Solid type hierarchy. Good reference.
- **Error types**: 96 lines. Copy as-is.
- **SIMD syntax**: `.+`, `.-`, `.*` operators and `[1.0, 2.0, 3.0, 4.0]f32x4` — genuinely good design.
- **C runtime files**: Real working C implementations.
- **LLVM codegen core**: SSA generation, basic blocks, function setup — buried in 12K lines but extractable.
- **build.rs**: C runtime compilation setup works.

---

## Part 2: Why a Kernel Language

### The Market Reality

| Approach | Competition | Chance of Adoption |
|----------|------------|-------------------|
| Full systems language | Rust, Zig, C, Odin, Jai + hundreds of dead projects | ~1% |
| SIMD kernel language | ISPC (ugly C syntax, Intel-focused) | Real niche, proven demand |

### Who Uses This

| Person | Current Pain | What Eä Solves |
|--------|-------------|---------------|
| Game dev (particle systems, physics) | Writing `_mm256_fmadd_ps` intrinsics | `a .* b .+ c` → same instruction |
| Audio/DSP engineer | Manually unrolling SIMD loops | Clean loops that vectorize |
| Scientist with Python pipeline | NumPy can't optimize their hot loop | Write kernel in Eä, call via ctypes |
| Rust dev needing explicit SIMD | `std::simd` still unstable | Write SIMD in Eä, link the .o |
| C++ dev tired of intrinsics | SSE/AVX intrinsic soup | Readable math that compiles to intrinsics |

### Why This Works

1. **Zero adoption cost** — produces `.o`/`.so` files, callable from any language via C ABI
2. **No ecosystem needed** — the calling language handles I/O, strings, collections, networking
3. **Small scope** — a kernel language doesn't need a package manager, LSP, or standard library
4. **Proven model** — ISPC is used at Pixar, Intel, Epic Games. The niche exists.
5. **Achievable** — one developer can build a correct, complete kernel compiler

---

## Part 3: v2 Architecture

### Core Principle: Nothing Exists Until It Compiles and Runs

Every feature must be proven by an end-to-end test:
```
EA source → lexer → parser → type check → codegen → inkwell Module → .o file → link with test harness → execute → verify output
```

If the binary output doesn't match, the feature doesn't exist.

### File Size Rule

**No file exceeds 500 lines.** Split before you hit it. No exceptions.

### Compilation Pipeline

```
Source (.ea)
    ↓
  Lexer (logos) → Vec<Token>
    ↓
  Parser (recursive descent) → Vec<Stmt>
    ↓
  Type Checker → Vec<Stmt> (validated)
    ↓
  Codegen (inkwell) → LLVM Module
    ↓
  TargetMachine::write_to_file() → .o file  (no llc step — inkwell does this in-process)
    ↓
  cc → link with caller's code → native binary
```

**Key difference from v1**: No `.ll` file in the default pipeline. No `llc` subprocess. Inkwell's `TargetMachine` writes object files directly. Emit `.ll` only with `--emit-llvm` for debugging.

### Output Modes

```bash
ea kernel.ea                    # → kernel.o (default: object file)
ea kernel.ea --lib              # → kernel.so / kernel.dll (shared library)
ea kernel.ea --emit-llvm        # → kernel.ll (LLVM IR for inspection)
ea kernel.ea -o main            # → main (linked executable, for testing with main())
```

### What Does NOT Exist in v2

**Never build these** (the calling language handles them):
- ❌ Package manager
- ❌ Standard library (Vec, HashMap, String, File I/O)
- ❌ LSP server
- ❌ JIT execution
- ❌ Compile-time execution engine
- ❌ Incremental/parallel/streaming compilation
- ❌ Memory profiler / string interner / parser optimizer
- ❌ VS Code extension
- ❌ Benchmarks against other languages (until Phase 6+)

---

## Part 4: Build Phases

### Phase 1: Hello World + Export (Target: ~1,500 lines)

**Goal**: A function compiles to an object file, links with a C test harness, and runs.

```ea
// hello.ea
export func add(a: i32, b: i32) -> i32 {
    return a + b;
}

func main() {
    println(add(3, 4));
}
```

Two output modes from day one:
1. `ea hello.ea -o hello && ./hello` → prints "7" (standalone executable)
2. `ea hello.ea` → `hello.o` (object file with `add` symbol, callable from C)

The C test harness:
```c
// test_harness.c
#include <stdio.h>
extern int add(int a, int b);
int main() {
    printf("%d\n", add(3, 4));
    return 0;
}
// cc test_harness.c hello.o -o test && ./test → prints "7"
```

**What gets built**:
- Lexer: `func`, `export`, `main`, `return`, `println`, `(`, `)`, `{`, `}`, `,`, `:`, `->`, string/integer literals, identifiers, type keywords (`i32`)
- Parser: function declarations (with `export`), function calls, return statements, parameters, type annotations
- AST: `Stmt::Function { export: bool }`, `Stmt::Return`, `Expr::Call`, `Expr::Literal`, `Expr::Variable`
- Type checker: function signatures, parameter types, return type validation
- Codegen: LLVM module, function generation with C ABI, `printf` integration, `TargetMachine` object file output
- CLI: argument parsing for output modes

**Files created**:
```
src/
├── main.rs          (~150 lines)  CLI entry point
├── lib.rs           (~100 lines)  Pipeline orchestration
├── error.rs         (~60 lines)   Error types
├── lexer/
│   ├── mod.rs       (~250 lines)  Tokenizer
│   └── tokens.rs    (~100 lines)  Token utilities
├── ast/
│   └── mod.rs       (~120 lines)  Core AST nodes
├── parser/
│   └── mod.rs       (~300 lines)  Parser
├── typeck/
│   └── mod.rs       (~150 lines)  Type checking
└── codegen/
    └── mod.rs       (~350 lines)  LLVM codegen + object file emission
```

**Success criteria**:
- [ ] `ea hello.ea -o hello && ./hello` prints correct output
- [ ] `ea hello.ea` produces `hello.o` with correct C ABI symbols
- [ ] C test harness links against `hello.o` and calls `add()` correctly
- [ ] `ea` with no args prints usage
- [ ] `ea nonexistent.ea` prints clear error
- [ ] 5+ end-to-end tests pass

---

### Phase 2: Variables + Arithmetic (Target: +700 lines, ~2,200 total)

**Goal**: Let bindings, arithmetic, multiple numeric types.

```ea
export func scale_and_offset(value: f32, scale: f32, offset: f32) -> f32 {
    let result: f32 = value * scale + offset;
    return result;
}

func main() {
    println(scale_and_offset(10.0, 2.5, 3.0));
}
// Output: 28
```

**What gets added**:
- Lexer: `let`, `=`, `+`, `-`, `*`, `/`, `%`, float literals, types (`i64`, `f32`, `f64`, `bool`)
- Parser: variable declarations, binary expressions, type annotations
- AST: `Stmt::Let`, `Expr::Binary`, `Expr::Variable`, `TypeAnnotation`
- Type checker: variable scoping, arithmetic type rules, numeric type promotion
- Codegen: `alloca`/`store`/`load`, integer + float arithmetic, printf formatting per type

**Success criteria**:
- [ ] Integer arithmetic (+, -, *, /, %) correct
- [ ] Float arithmetic correct
- [ ] Type annotations enforced (wrong type = compile error)
- [ ] Exported functions with numeric params callable from C
- [ ] 15+ end-to-end tests pass

---

### Phase 3: Control Flow (Target: +600 lines, ~2,800 total)

**Goal**: Branches and loops.

```ea
export func sum_range(start: i32, end: i32) -> i32 {
    let sum: i32 = 0;
    let i: i32 = start;
    while i <= end {
        sum = sum + i;
        i = i + 1;
    }
    return sum;
}
```

**What gets added**:
- Lexer: `if`, `else`, `while`, comparisons (`<=`, `>=`, `==`, `!=`), logical (`&&`, `||`)
- Parser: if/else, while, comparison/logical expressions
- AST: `Stmt::If`, `Stmt::While`
- Type checker: boolean validation, control flow return analysis
- Codegen: conditional branches, loop basic blocks, phi nodes

**Split codegen**:
```
codegen/
├── mod.rs              Core + module setup + object emission
├── expressions.rs      Expression compilation
└── statements.rs       Statement compilation (if/while/for)
```

**Success criteria**:
- [ ] if/else branches correctly
- [ ] while loops iterate correctly
- [ ] Nested control flow (if inside while) works
- [ ] Exported looping functions callable from C produce correct results
- [ ] 25+ end-to-end tests pass

---

### Phase 4: Functions + Pointers (Target: +600 lines, ~3,400 total)

**Goal**: Multiple functions, recursion, and pointer/slice arguments for C interop.

```ea
func fibonacci(n: i32) -> i32 {
    if n <= 1 {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

export func fill_fibonacci(out: *mut i32, count: i32) {
    let i: i32 = 0;
    while i < count {
        out[i] = fibonacci(i);
        i = i + 1;
    }
}
```

```c
// C caller:
extern void fill_fibonacci(int* out, int count);
int buf[10];
fill_fibonacci(buf, 10);
// buf = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

**What gets added**:
- Lexer: `*`, `mut` (in pointer context)
- Parser: pointer type annotations, pointer indexing (`ptr[i]`), pointer dereference
- AST: `TypeAnnotation::Pointer { mutable, inner }`, `Expr::Index`
- Type checker: pointer type rules, mutability checking
- Codegen: pointer parameters (C ABI), GEP instructions, pointer load/store

**Add file**:
```
codegen/functions.rs    Function compilation + C ABI
```

**This is the phase that makes Eä useful.** After Phase 4, someone can write a function in Eä that operates on a C array and call it from their project.

**Success criteria**:
- [ ] Recursive functions work (fibonacci)
- [ ] Functions can call each other
- [ ] `*i32` and `*mut i32` pointer parameters work with C ABI
- [ ] Pointer indexing (`ptr[i]`) reads/writes correctly
- [ ] C test harness passes arrays to exported Eä functions
- [ ] 35+ end-to-end tests pass

---

### Phase 5: Explicit SIMD (Target: +700 lines, ~4,100 total)

**Goal**: SIMD vector types with explicit element-wise operations.

```ea
export func vector_add(a: *f32, b: *f32, out: *mut f32, len: i32) {
    let i: i32 = 0;
    while i + 4 <= len {
        let va = load(a, i)f32x4;
        let vb = load(b, i)f32x4;
        let result = va .+ vb;
        store(out, i, result);
        i = i + 4;
    }
    // scalar remainder
    while i < len {
        out[i] = a[i] + b[i];
        i = i + 1;
    }
}
```

**What gets added**:
- Lexer: SIMD type tokens (`f32x4`, `i32x4`), `.+`, `.-`, `.*`, `./` operators, `load`, `store`, `splat`
- Parser: SIMD type annotations, element-wise operations, SIMD built-in calls, vector literals
- AST: `SIMDExpr`, `SIMDVectorType`, `SIMDOperator`
- Type checker: vector type validation, element-wise operator compatibility
- Codegen: LLVM vector types, vector load/store, vector arithmetic, extract/insert element

**Add files**:
```
ast/simd.rs             SIMD AST nodes
codegen/simd.rs         SIMD code generation
```

**This is the differentiator.** After Phase 5, Eä can do something no other easily-accessible tool does: let you write readable SIMD code that compiles to native vector instructions and links into any project.

**Success criteria**:
- [ ] `f32x4` and `i32x4` types compile to LLVM vector types
- [ ] `.+`, `.-`, `.*`, `./` produce correct results
- [ ] `load()` and `store()` move data between pointers and vectors
- [ ] `splat()` broadcasts scalar to all lanes
- [ ] Element extraction (`vec[0]`) works
- [ ] C test harness passes float arrays, Eä processes them with SIMD, results correct
- [ ] 45+ end-to-end tests pass

---

### Phase 6: Deep SIMD (Target: +500 lines, ~4,600 total)

**Goal**: The operations that make SIMD actually useful beyond basic arithmetic.

```ea
export func fma_kernel(a: *f32, b: *f32, c: *f32, out: *mut f32, len: i32) {
    let i: i32 = 0;
    while i + 4 <= len {
        let va = load(a, i)f32x4;
        let vb = load(b, i)f32x4;
        let vc = load(c, i)f32x4;
        let result = fma(va, vb, vc);  // a*b+c in one instruction
        store(out, i, result);
        i = i + 4;
    }
}

export func horizontal_sum(data: *f32, len: i32) -> f32 {
    let acc = splat(0.0)f32x4;
    let i: i32 = 0;
    while i + 4 <= len {
        let v = load(data, i)f32x4;
        acc = acc .+ v;
        i = i + 4;
    }
    return reduce_add(acc);
}
```

**What gets added**:
- `fma(a, b, c)` — fused multiply-add (single instruction on modern CPUs)
- `reduce_add(v)` — horizontal sum of all lanes → scalar
- `reduce_max(v)` / `reduce_min(v)` — horizontal max/min
- `shuffle(v, [3,1,2,0])` — lane reordering
- `select(mask, a, b)` — per-lane conditional select
- `f32x8` / `i32x8` — 256-bit vectors (AVX)

**Success criteria**:
- [ ] FMA produces correct results and maps to FMA instruction on supporting hardware
- [ ] Horizontal reductions produce correct scalar results
- [ ] Shuffle reorders lanes correctly
- [ ] Select picks correct lanes based on mask
- [ ] 256-bit vectors work on AVX-capable hardware
- [ ] 55+ end-to-end tests pass

---

### Phase 7: Structs + Shared Library Output (Target: +500 lines, ~5,100 total)

**Goal**: Structs for passing structured data across FFI, and `.so`/`.dll` output.

```ea
struct Particle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
}

export func update_particles(particles: *mut Particle, count: i32, dt: f32) {
    let i: i32 = 0;
    while i < count {
        particles[i].x = particles[i].x + particles[i].vx * dt;
        particles[i].y = particles[i].y + particles[i].vy * dt;
        i = i + 1;
    }
}
```

```python
# Python caller via ctypes:
import ctypes
lib = ctypes.CDLL('./particles.so')
# ... call update_particles with array of structs
```

**What gets added**:
- Parser: struct definitions, field access, struct construction
- AST: `Stmt::Struct`, `Expr::FieldAccess`, `Expr::StructInit`
- Type checker: struct type registration, field type validation
- Codegen: LLVM struct types (C-compatible layout), GEP for field access
- CLI: `--lib` flag for shared library output

**Success criteria**:
- [ ] Struct definitions compile with C-compatible layout
- [ ] Field access reads correct values
- [ ] Pointer-to-struct indexing works (`particles[i].x`)
- [ ] `--lib` flag produces loadable `.so`
- [ ] Python ctypes can load and call functions from the `.so`
- [ ] 65+ end-to-end tests pass

---

## Part 5: What to Copy From v1

**v1 location**: `/mnt/c/Users/Peter.lukka/Desktop/DEV/EA`

### Copy and Modify

| v1 File | What to Take | Changes |
|---------|-------------|---------|
| `ea-compiler/src/error.rs` (96 lines) | Everything | Copy as-is |
| `ea-compiler/Cargo.toml` | Dependency patterns | Strip to minimal |
| `ea-compiler/src/lexer/mod.rs` | Logos pattern, Position struct | Rewrite smaller, remove string interner / memory profiler deps |
| `ea-compiler/src/ast.rs` | BinaryOp, UnaryOp, Literal enums | Cherry-pick enums only, rebuild Expr/Stmt minimal |
| `ea-compiler/src/codegen/mod.rs` lines 86-100 | CodeGenerator struct pattern | Extract Context/Module/Builder setup only |

### Do NOT Copy

Everything else. Specifically:
- `comptime/` — fake implementations
- `simd_advanced/` — 37 unused instruction sets
- `package/` — package manager with no imports
- `lsp/` — premature
- `stdlib/*.rs` — Rust code EA can't call
- `type_system/simd_validator.rs` — over-engineered
- All optimization modules (`incremental_compilation`, `parallel_compilation`, `streaming_compiler`, `memory_profiler`, `parser_optimization`, `resource_manager`, `string_interner`, `llvm_optimization`, `llvm_context_pool`)
- JIT modules (`jit_execution`, `jit_cache`, `jit_cached`, `jit_batch_symbols`, `execution_mode`)
- `runtime/*.c` — **not needed**. A kernel language doesn't provide its own collections. The caller handles that.

### Note on C Runtime Files

v1's C runtime files (`vec_runtime.c`, `hashmap_runtime.c`, etc.) are **not needed** in v2. A kernel language doesn't need Vec, HashMap, String, or File I/O. The calling language provides all of that. Eä just processes data through pointers.

---

## Part 6: The Language

### Core Types

```
i8, i16, i32, i64          Signed integers
u8, u16, u32, u64          Unsigned integers
f32, f64                   Floating point
bool                       Boolean
*T                         Immutable pointer to T
*mut T                     Mutable pointer to T
f32x4, f32x8               SIMD float vectors
i32x4, i32x8               SIMD integer vectors
```

No strings. No collections. No references. Kernel code works with numbers and pointers.

### Functions

```ea
// Internal function (not visible from C)
func helper(x: i32) -> i32 {
    return x * 2;
}

// Exported function (C ABI, visible from linker)
export func public_api(x: i32) -> i32 {
    return helper(x) + 1;
}

// Entry point (only for standalone executable mode)
func main() {
    println(public_api(21));
}
```

### Variables

```ea
let x: i32 = 42;           // immutable
let y: f32 = 3.14;
let mut counter: i32 = 0;  // mutable (needed for loops)
counter = counter + 1;
```

### Control Flow

```ea
if x > 0 {
    // ...
} else {
    // ...
}

while i < len {
    // ...
    i = i + 1;
}
```

### Pointers (C Interop)

```ea
export func process(data: *f32, output: *mut f32, len: i32) {
    let i: i32 = 0;
    while i < len {
        output[i] = data[i] * 2.0;
        i = i + 1;
    }
}
```

### SIMD

```ea
// Vector literals
let v = [1.0, 2.0, 3.0, 4.0]f32x4;

// Broadcast
let ones = splat(1.0)f32x4;

// Load from pointer (4 consecutive floats starting at ptr + offset)
let v = load(ptr, offset)f32x4;

// Store to pointer
store(ptr, offset, v);

// Element-wise arithmetic
let sum = a .+ b;
let diff = a .- b;
let prod = a .* b;
let quot = a ./ b;

// Element access
let first: f32 = v[0];

// Fused multiply-add: a*b + c
let result = fma(a, b, c);

// Horizontal reductions
let total: f32 = reduce_add(v);
let maximum: f32 = reduce_max(v);
let minimum: f32 = reduce_min(v);

// Shuffle lanes
let reversed = shuffle(v, [3, 2, 1, 0]);

// Per-lane select
let picked = select(mask, a, b);
```

### Structs (C-Compatible)

```ea
struct Vec2 {
    x: f32,
    y: f32,
}

export func magnitude(v: *Vec2) -> f32 {
    return sqrt(v.x * v.x + v.y * v.y);
}
```

Structs use C-compatible memory layout. A `struct Vec2` in Eä has the same layout as:
```c
struct Vec2 { float x; float y; };
```

---

## Part 7: Directory Structure

```
ea2/
├── Cargo.toml
├── build.rs                    # Target setup (no C runtime needed)
├── CLAUDE.md                   # Dev guidance
├── EA_V2_SPECIFICATION.md      # This file
├── src/
│   ├── main.rs                 # CLI (~150 lines)
│   ├── lib.rs                  # Pipeline API (~100 lines)
│   ├── error.rs                # Error types (~60 lines)
│   ├── lexer/
│   │   ├── mod.rs              # Tokenizer (~300 lines max)
│   │   └── tokens.rs           # Token utilities (~150 lines max)
│   ├── ast/
│   │   ├── mod.rs              # Core AST (~250 lines max)
│   │   └── simd.rs             # SIMD AST (Phase 5+, ~200 lines max)
│   ├── parser/
│   │   ├── mod.rs              # Parser core + expressions (~400 lines max)
│   │   └── statements.rs       # Statement parsing (~300 lines max)
│   ├── typeck/
│   │   └── mod.rs              # Type checker (~400 lines max)
│   ├── codegen/
│   │   ├── mod.rs              # Codegen core + object emission (~400 lines max)
│   │   ├── expressions.rs      # Expression compilation (~350 lines max)
│   │   ├── statements.rs       # Statement compilation (~350 lines max)
│   │   ├── functions.rs        # Function compilation + C ABI (~300 lines max)
│   │   └── simd.rs             # SIMD codegen (Phase 5+, ~400 lines max)
│   └── target.rs               # Target machine setup (~100 lines max)
├── tests/
│   ├── end_to_end.rs           # Compile-and-run tests
│   └── c_interop.rs            # C harness linking tests
└── examples/
    ├── hello.ea
    ├── fibonacci.ea
    ├── vector_add.ea
    └── dot_product.ea
```

Total budget: ~4,500 lines of Rust for a working SIMD kernel compiler.

No `runtime/` directory. No `stdlib/`. No `benches/`. No `demo/`.

---

## Part 8: Technical Decisions

### Cargo.toml

```toml
[package]
name = "ea"
version = "0.1.0"
edition = "2021"
description = "SIMD kernel compiler — write readable SIMD, call from any language"

[dependencies]
logos = { version = "0.13", features = ["export_derive", "std"] }
anyhow = "1.0"
thiserror = "1.0"
inkwell = { version = "0.4", features = ["llvm14-0"], optional = true }
tempfile = "3.8"

[build-dependencies]
# No cc crate — no C runtime to compile

[dev-dependencies]
rstest = "0.18"

[features]
default = ["llvm"]
llvm = ["inkwell"]
```

### Object File Emission (No llc)

```rust
// In codegen — write .o directly from inkwell
let target_triple = TargetMachine::get_default_triple();
let target = Target::from_triple(&target_triple).unwrap();
let machine = target.create_target_machine(
    &target_triple,
    "native",  // use host CPU features
    "",
    OptimizationLevel::Default,
    RelocMode::PIC,    // position-independent for .so output
    CodeModel::Default,
).unwrap();

machine.write_to_file(&module, FileType::Object, Path::new("output.o")).unwrap();
```

### C ABI Compliance

All `export` functions use the C calling convention:
```rust
let fn_type = context.i32_type().fn_type(&[context.i32_type().into()], false);
let function = module.add_function("my_func", fn_type, Some(Linkage::External));
// Linkage::External makes the symbol visible to the linker
```

Internal `func` uses `Linkage::Private` — invisible from outside.

### Testing Strategy

**Two types of end-to-end tests**:

1. **Standalone tests** (has `main()`):
```rust
fn compile_and_run(source: &str) -> TestOutput {
    // 1. Write source to temp .ea file
    // 2. Invoke our compiler → produces .o
    // 3. Link: cc temp.o -o temp_bin (with printf from libc)
    // 4. Execute temp_bin
    // 5. Capture stdout + exit code
}
```

2. **C interop tests** (export functions, no `main()`):
```rust
fn compile_and_link_with_c(ea_source: &str, c_source: &str) -> TestOutput {
    // 1. Write .ea source, compile → .o
    // 2. Write .c harness
    // 3. Link: cc harness.c kernel.o -o test_bin
    // 4. Execute test_bin
    // 5. Capture stdout + exit code
}
```

Both verify actual execution output. No unit tests for internal types.

---

## Part 9: Rules for Development

### The Golden Rules

1. **If it doesn't compile and run, it doesn't exist.**
2. **500-line file limit.** Split before you hit it.
3. **No premature features.** If the current phase doesn't need it, don't build it.
4. **No fake implementations.** If a function can't do what its name says, don't write it.
5. **End-to-end tests first.** Write the test before the feature.
6. **One phase at a time.** Don't start Phase N+1 until Phase N's tests all pass.
7. **Delete, don't comment.** No `// TODO`, no dead code.
8. **C interop is the product.** Every exported function must be callable from C correctly.

### Anti-Patterns to Avoid

| Anti-Pattern | v1 Example | v2 Rule |
|-------------|-----------|---------|
| God files | codegen at 12,544 lines | 500-line limit |
| Aspirational modules | Package manager with no imports | Don't build what users don't need |
| Fake benchmarks | Comparing IR gen vs Rust compile | Only benchmark when the pipeline is complete |
| Fake implementations | ComptimeEngine with hardcoded values | Functions do what they claim |
| Premature optimization | String interner, context pools | Optimize after correctness |
| Kitchen sink | 37 SIMD instruction sets | Start with f32x4, add more when tested |
| Wrong scope | Trying to compete with Rust | Compete with ISPC: SIMD kernels, not general purpose |

---

## Part 10: Known Risks

### Risk 1: LLVM Gravity

LLVM is a code generator, not a playground. It will constantly tempt you into:
- "I should build a custom optimizer"
- "I should support 12 target features"
- "I should write a custom pass manager"

**Rule**: Do not touch LLVM optimization, target features, or pass management before Phase 7 is complete. Use `OptimizationLevel::Default` and `"native"` target. That's it.

### Risk 2: SIMD Scope Creep

Phases 5-6 are the danger zone. It's easy to think:
- "SSE2 too", "ARM NEON too", "mask types", "gather/scatter", "auto-unrolling", "alignment control"

**Rule**: Phase 5 supports exactly: `f32x4`, `i32x4`, `load`, `store`, `splat`, `.+`, `.-`, `.*`, `./`, element access. Nothing more. Phase 6 adds exactly: `fma`, `reduce_add`, `reduce_max`, `reduce_min`, `shuffle`, `select`, `f32x8`/`i32x8`. Nothing more.

### Risk 3: Syntax Expansion

Every new syntax rule multiplies: parser complexity, type checker logic, edge cases, codegen paths. The question is always: "Is this necessary for a SIMD kernel, or is it nice-to-have?"

Example: `for i in 0..len` is nice. But `while` does the same thing. Cut it. (Cut.)

**Rule**: Before adding any syntax, ask: "Does a kernel need this, or do I just want it?" If the answer is want, the answer is no.

### The Real Success Metric

The real success point is not Phase 7 completion. It's:

> One person writes: "I replaced intrinsics with Eä and it became cleaner."

Phase 4 + Phase 5 stable + one external user in a real project. That's the win.

---

## Part 11: After Phase 7 — What Comes Next

Only after all 7 phases are complete and solid:

1. **More SIMD types** — f64x2, f64x4, i16x8, u8x16 (one at a time, tested)
2. **Auto-vectorization hints** — let the compiler widen scalar loops to SIMD
3. **Optimization levels** — `-O0` through `-O3` via LLVM passes
4. **Header generation** — `ea kernel.ea --header` produces `kernel.h` for C callers
5. **Python bindings generator** — `ea kernel.ea --python` produces ctypes wrapper
6. **Benchmarks** — compare Eä kernels against hand-written intrinsics and ISPC

Each as a focused addition with its own end-to-end tests.

---

## Appendix: Key Technical References From v1

### LLVM Codegen Patterns

**Module + function setup**:
```rust
let context = Context::create();
let module = context.create_module("ea_module");
let builder = context.create_builder();

let fn_type = context.i32_type().fn_type(&[context.i32_type().into()], false);
let function = module.add_function("add", fn_type, Some(Linkage::External));
let entry = context.append_basic_block(function, "entry");
builder.position_at_end(entry);
```

**Printf for standalone mode**:
```rust
let printf_type = context.i32_type().fn_type(
    &[context.ptr_type(AddressSpace::default()).into()],
    true
);
module.add_function("printf", printf_type, Some(Linkage::External));
```

**SIMD vector operations**:
```rust
let f32x4_type = context.f32_type().vec_type(4);
let result = builder.build_float_add(left_vec, right_vec, "vadd").unwrap();
```

**Object file emission**:
```rust
let triple = TargetMachine::get_default_triple();
let target = Target::from_triple(&triple).unwrap();
let machine = target.create_target_machine(&triple, "native", "", OptLevel::Default, RelocMode::PIC, CodeModel::Default).unwrap();
machine.write_to_file(&module, FileType::Object, Path::new("out.o")).unwrap();
```

### Logos Lexer Pattern
```rust
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r\n]+")]
pub enum TokenKind {
    #[token("func")]
    Func,
    #[token("export")]
    Export,
    #[token("return")]
    Return,
    // ...
}
```
