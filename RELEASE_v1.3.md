# Eä v1.3 — Kernel Language

v1.2 shipped `ea bind` and eliminated hundreds of lines of ctypes boilerplate.
v1.3 attacks the two remaining boilerplate sources: the .ea kernel side (loop
scaffolding, tail handling) and the host-language side (buffer allocation,
result unpacking). Four features, each independently useful, together
transformative.

**Design constraint:** every feature is explicit. The programmer chooses width,
step, tail strategy, and buffer capacity. The compiler generates scaffolding
from those choices — no heuristics, no pattern-matching on user code, no magic.

---

## Feature 1: Compile-Time Constants

### What

Top-level `const` declarations. Integer and float only. Inlined as immediates
at every use site. No runtime storage.

```
const STEP: i32 = 8
const RADIUS: i32 = 1
const PI: f32 = 3.14159265

export func scale(data: *f32, out: *mut f32, len: i32) {
    let factor: f32x8 = splat(PI)
    let mut i: i32 = 0
    while i + STEP <= len {
        store(out, i, load(data, i) .* factor)
        i = i + STEP
    }
}
```

### Why

Every stencil kernel hardcodes `1` where it means RADIUS. Every unrolled loop
hardcodes `2` where it means NUM_ACC. Constants document intent, enable
specialization (same source, different constants), and improve constant folding.

### Semantics

- `const NAME: TYPE = LITERAL` at top level only (not inside functions).
- TYPE must be `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `f32`,
  or `f64`. No pointers, no vectors, no structs, no booleans.
- VALUE must be a literal (integer, float, hex, binary). No expressions, no
  references to other constants. This keeps evaluation trivial — the compiler
  does not need a const-eval interpreter.
- Constants are substituted during type checking. By codegen, they are
  indistinguishable from literals. No LLVM globals, no runtime loads.
- Scope: visible to all functions in the same file, from point of declaration
  forward (like structs).
- Name collisions with variables are a compile error:
  `"'STEP' is a constant, cannot be used as variable name"`.
- Constants do NOT appear in metadata JSON or generated bindings. They are
  internal to the kernel source.
- Redefinition of the same name is a compile error.

### Implementation

**AST:** New variant `Stmt::Const { name, ty, value, span }`. `value` is
restricted to `Expr::Literal`.

**Parser:** In `declaration()`, match `const` keyword. Parse
`const NAME: TYPE = LITERAL`. Reject non-literal values at parse time with
a clear error: `"const value must be a literal"`.

**Type checker:** Register constants in a `HashMap<String, (TypeAnnotation,
Literal)>` on the checker context. When resolving a variable name, check
constants first. Verify the literal type matches the declared type. Error if
a `let` binding shadows a constant name.

**Codegen:** Constants are resolved during type checking and never reach
codegen as a distinct node. The type checker replaces `Expr::Variable` refs
to constants with `Expr::Literal` carrying the constant's value and the
appropriate span.

**Metadata:** Skip `Stmt::Const` in `generate_json`. Constants are not
exported.

**Tests:**
- Basic: `const X: i32 = 42` used in arithmetic.
- Float: `const PI: f32 = 3.14` used in FMA kernel.
- Hex/binary: `const MASK: i32 = 0xFF`, `const FLAGS: i32 = 0b1010`.
- Error: `const X: i32 = 5` then `let X: i32 = 10` → compile error.
- Error: `const X: i32 = y + 1` → error (not a literal).
- Error: duplicate `const X` → error.
- Error: `const V: f32x4 = ...` → error (unsupported type).

**Estimate:** ~150 lines. Parser (~30), type checker (~60), AST variant (~10),
tests (~50+).

---

## Feature 2: Kernel Construct

### What

`export kernel` is a new top-level declaration for streaming iteration.
It declares the loop structure, iteration variable, and step size in the
signature. The body is per-step — the compiler generates the outer loop.

```
export kernel fma(a: *f32, b: *f32, c: *f32, out: *mut f32)
    over i in len step 4
{
    let va: f32x4 = load(a, i)
    let vb: f32x4 = load(b, i)
    let vc: f32x4 = load(c, i)
    store(out, i, fma(va, vb, vc))
}
```

This compiles to exactly the same code as:

```
export func fma(a: *f32, b: *f32, c: *f32, out: *mut f32, len: i32) {
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

No more, no less.

### Why

Every streaming kernel in the codebase has the same structure:
`let mut i = 0 / while i + N <= len / body / i = i + N`. This is ~4 lines of
scaffolding per kernel that carry no information beyond what `over i in len
step 4` says. The `kernel` construct makes the iteration pattern structural —
the compiler knows this is a streaming kernel, which is prerequisite for
tail handling (Feature 3) and future fusion.

### Syntax

```
[export] kernel NAME(PARAMS) over VAR in RANGE step STEP [tail STRATEGY] {
    BODY
}
```

- `NAME`: function name (same rules as `func`).
- `PARAMS`: parameter list. Same as `func` except `RANGE` is not listed —
  it becomes an implicit parameter. See below.
- `VAR`: the iteration variable name. Declared explicitly — no implicit `i`.
  Type is always `i32`. Immutable within the body.
- `RANGE`: the name of the range bound. Becomes an `i32` parameter appended
  to the C ABI signature after all declared params. The kernel iterates
  `VAR` from `0` to `RANGE` (exclusive), advancing by `STEP`.
- `STEP`: a positive integer literal or const reference. Determines the loop
  stride.
- `tail STRATEGY`: optional, see Feature 3.

**The `RANGE` parameter rule:** `over i in len step 4` adds `len: i32` to the
end of the generated C function signature. So the kernel:

```
export kernel fma(a: *f32, b: *f32, c: *f32, out: *mut f32)
    over i in len step 4
{ ... }
```

produces the C signature: `void fma(float* a, float* b, float* c, float* out, int32_t len)`.

This is identical to what the user would write with `export func`. The `len`
parameter appears in metadata JSON and generated bindings. From the caller's
perspective, `kernel` and `func` produce the same ABI.

**Why not put `len` in the parameter list?** Because `len` is not a parameter
the kernel uses freely — it is the range bound. Putting it in `PARAMS` and
also in `over ... in len` would repeat the name twice. The `over` clause is
the single source of truth for the iteration structure.

### Semantics

- The generated loop is: `let mut VAR: i32 = 0; while VAR + STEP <= RANGE { BODY; VAR = VAR + STEP; }`
- `VAR` is immutable inside `BODY`. Assigning to `VAR` is a compile error:
  `"cannot assign to loop variable 'i' — it is advanced by the kernel's step"`.
- `BODY` may not contain `return` with a value. Streaming kernels return void.
  `return` (bare, for early exit) is allowed.
- `BODY` may reference `RANGE` as a read-only i32.
- `BODY` may reference all PARAMS normally.
- `BODY` may call helper `func`s, declare locals, use `if`/`while`/`foreach`.
- `STEP` must be a positive integer literal or a const name resolving to a
  positive integer. Zero or negative step is a compile error.
- Without a `tail` clause, the kernel does NOT handle remainders. Elements
  where `i + STEP > len` are not processed. This is the same as today's
  `while i + N <= len` — no silent scalar fallback.
- A `kernel` without `export` is valid (internal helper kernel).

### Implementation

**AST:** New variant:
```rust
Stmt::Kernel {
    name: String,
    params: Vec<Param>,
    range_var: String,        // "i"
    range_bound: String,      // "len"
    step: u32,                // 4
    tail: Option<TailStrategy>,
    body: Vec<Stmt>,
    export: bool,
    span: Span,
}
```

New enum:
```rust
enum TailStrategy {
    Scalar,
    Mask,
    None,   // "pad" — caller guarantees no remainder
}
```

**Parser:** In `declaration()`, after matching `export` or at top level, check
for `kernel` keyword. Parse the parameter list, then `over VAR in RANGE step
STEP`, then optional `tail STRATEGY`, then block.

**Type checker:** Desugar `Stmt::Kernel` into `Stmt::Function` with the
generated loop. This keeps the rest of the pipeline (codegen, metadata, header
generation) unchanged. The desugaring:

1. Create a `len` parameter: `Param { name: range_bound, ty: i32 }` appended
   to params.
2. Build the loop body: `let mut VAR: i32 = 0; while VAR + STEP <= RANGE { BODY; VAR = VAR + STEP; }`.
3. If tail strategy is set, append the tail (see Feature 3).
4. Wrap as `Stmt::Function { export, name, params: params + [range_param], body: [loop], ... }`.

Type-check the desugared function normally. The iteration variable and range
bound are introduced as typed bindings during desugaring, so the type checker
handles them as regular variables.

**Why desugar at type-check rather than codegen?** Because the desugared form
is a `Stmt::Function`, and all downstream phases (codegen, metadata, header,
bindings) already handle functions. No changes needed in codegen, metadata,
header generation, or any binding generator.

**Metadata:** The desugared function appears in metadata as a normal export
with all parameters including the range bound. Bindings work unchanged —
`ea bind --python` collapses `len` as before.

**Tests:**
- Basic streaming: `kernel fma(...) over i in n step 4` produces same output
  as equivalent `func`.
- Const step: `const STEP: i32 = 8; kernel k(...) over i in n step STEP`.
- Error: assignment to loop variable → clear error.
- Error: `return 42` inside kernel → error.
- Error: `step 0` → error.
- Error: `step -1` → error (not positive integer literal).
- Non-export kernel: `kernel helper(...)` without `export`.
- Verify C signature: `len` parameter appears in header output.
- Verify metadata: `len` parameter appears in JSON.
- Verify bindings: `ea bind --python` collapses `len` parameter.

**Estimate:** ~350 lines. AST variant (~20), parser (~80), desugaring in type
checker (~100), tests (~150+).

---

## Feature 3: Tail Clause

### What

The `tail` clause on a `kernel` declaration specifies how to handle remaining
elements when `len` is not a multiple of `step`.

```
export kernel scale(data: *f32, out: *mut f32, factor: f32)
    over i in len step 8
    tail scalar
{
    let vf: f32x8 = splat(factor)
    store(out, i, load(data, i) .* vf)
}
```

Three strategies:

| Strategy | Generated code | When to use |
|----------|---------------|-------------|
| `scalar` | Scalar loop `while i < len { out[i] = ...; i = i + 1 }` | Default safe choice. Works everywhere. |
| `mask` | `load_masked` / `store_masked` with `len - i` as count | SIMD tail. Faster. Requires `load_masked`/`store_masked` support. |
| `pad` | No tail. Kernel stops at last full step. | Caller guarantees `len % step == 0`, or doesn't care about remainder. |

No `tail` clause = no tail handling (same as `pad`). This maintains backward
compatibility and the principle that the compiler does nothing the programmer
didn't ask for.

### Why

The tail loop is the single most repeated pattern in every demo kernel. It
carries no algorithmic information — the strategy (scalar vs masked vs none)
is the only decision. Once that decision is declared, the implementation is
mechanical.

### Semantics

**`tail scalar`:** The compiler generates a scalar fallback loop after the
main SIMD loop. The scalar body is NOT auto-derived from the SIMD body. The
programmer provides it via a `tail { ... }` block:

```
export kernel scale(data: *f32, out: *mut f32, factor: f32)
    over i in len step 8
    tail scalar {
        out[i] = data[i] * factor
    }
{
    let vf: f32x8 = splat(factor)
    store(out, i, load(data, i) .* vf)
}
```

The tail block is a per-element body. The compiler generates:
`while i < len { TAIL_BODY; i = i + 1 }`.

The iteration variable `i` is shared between the main body and the tail body.
When the main loop exits, `i` is at the last unprocessed position. The tail
continues from there.

**Why not auto-derive the scalar body?** Because the scalar version of a SIMD
body is not always obvious. `load(data, i)` becomes `data[i]`, but
`fma(va, vb, vc)` becomes `a[i] * b[i] + c[i]`, and `select(mask, a, b)` has
no scalar equivalent without a branch. Auto-deriving this is an optimization
compiler problem. Eä is not an optimization compiler. The programmer writes
both — the SIMD body (which they already write) and the scalar body (2-3 lines
they currently copy-paste).

**`tail mask`:** The compiler generates a masked tail using `load_masked` and
`store_masked`. The mask body uses the same variable names as the main body,
but loads/stores are replaced with masked variants. Like `scalar`, the
programmer provides the masked body explicitly:

```
export kernel scale(data: *f32, out: *mut f32, factor: f32)
    over i in len step 8
    tail mask {
        let rem: i32 = len - i
        let vf: f32x8 = splat(factor)
        let v: f32x8 = load_masked(data, i, rem)
        store_masked(out, i, v .* vf, rem)
    }
{
    let vf: f32x8 = splat(factor)
    store(out, i, load(data, i) .* vf)
}
```

The compiler wraps this in: `let rem: i32 = RANGE - VAR; if rem > 0 { MASK_BODY }`.

Wait — that adds `rem` implicitly. To stay explicit:

The compiler generates: `if i < len { TAIL_BODY }`. The programmer computes
`len - i` themselves inside the block. This is exactly what they write today,
minus the `if` wrapper.

**`tail pad`:** No code generated. Explicit declaration that no tail handling
is needed. Equivalent to omitting the `tail` clause entirely, but documents
intent. Useful for kernels where the caller guarantees alignment.

**Syntax summary:**

```
// No tail (or explicit pad):
kernel k(...) over i in n step 4 { ... }
kernel k(...) over i in n step 4 tail pad { ... }

// Scalar tail:
kernel k(...) over i in n step 4 tail scalar { SCALAR_BODY } { SIMD_BODY }

// Masked tail:
kernel k(...) over i in n step 4 tail mask { MASK_BODY } { SIMD_BODY }
```

### Implementation

**AST:** The `TailStrategy` enum and body are part of `Stmt::Kernel`:

```rust
Stmt::Kernel {
    ...
    tail: Option<TailStrategy>,
    tail_body: Option<Vec<Stmt>>,  // body for scalar/mask strategies
    ...
}
```

**Parser:** After parsing `step N`, check for `tail` keyword. If present,
parse strategy name (`scalar`, `mask`, `pad`). If strategy is `scalar` or
`mask`, parse a block (the tail body), then parse the main body block.
If strategy is `pad` or absent, parse only the main body block.

**Type checker (desugaring):** Append the tail after the main loop:

- `scalar`: `while VAR < RANGE { TAIL_BODY; VAR = VAR + 1; }`
- `mask`: `if VAR < RANGE { TAIL_BODY }`
- `pad` / none: nothing.

The desugared function is type-checked normally. The tail body has access to
the same parameters, the iteration variable, and the range bound.

**Tests:**
- `tail scalar`: verify remainder elements are processed correctly.
- `tail mask`: verify masked load/store handles all tail lengths (1 through
  step-1).
- `tail pad`: verify no tail code generated (check LLVM IR or assembly).
- No tail: same behavior as `tail pad`.
- Error: `tail scalar` without tail body → parse error.
- Error: `tail mask` without tail body → parse error.
- Verify: kernel with `tail scalar` produces same results as hand-written
  `func` with scalar tail loop across multiple input lengths (including lengths
  that are exact multiples of step).

**Estimate:** ~200 lines. Parser extension (~50), desugaring extension (~40),
tests (~110+).

---

## Feature 4: Output Annotations

### What

Parameters annotated with `out` in the kernel source are marked as outputs in
metadata JSON. Optional `cap` and `count` annotations provide buffer sizing
and trimming hints. Binding generators use these to allocate and return buffers
automatically.

```
export func extract_positions(
    text: *u8,
    len: i32,
    delim: u8,
    out delim_pos: *mut i32 [cap: len],
    out lf_pos: *mut i32 [cap: len / 4],
    out counts: *mut i32 [cap: 3]
) {
    ...
}
```

### Why

The biggest remaining friction in host-language code is the
allocate-call-unpack pattern:

```python
# Before (current): 6 lines per kernel call
delim_pos = np.empty(n // 4 + 256, dtype=np.int32)
lf_pos = np.empty(n // 20 + 256, dtype=np.int32)
counts = np.zeros(3, dtype=np.int32)
csv_parse.extract_positions_quoted(text, delim_byte, delim_pos, lf_pos, counts)
n_delims = int(counts[0])
delim_pos = delim_pos[:n_delims]

# After (with output annotations): 1 line
delim_pos, lf_pos, counts = csv_parse.extract_positions_quoted(text, delim_byte)
```

The kernel author knows the buffer capacity bounds. Today that knowledge lives
in comments or in the caller's head. Output annotations put it in the source,
where the binding generator can use it.

### Syntax

```
out NAME: *mut TYPE [cap: EXPR]
out NAME: *mut TYPE [cap: EXPR, count: PATH]
```

- `out` keyword before the parameter name. Signals that this parameter is
  written to (output direction).
- `[cap: EXPR]` — capacity expression. Must reference only preceding input
  parameters and constants. Supports `+`, `-`, `*`, `/` on integer values.
  The binding generator evaluates this to allocate the output buffer.
- `[count: PATH]` — optional. Specifies which output scalar holds the actual
  count of valid elements. `PATH` is `param_name` for a single-element output,
  or `param_name[INDEX]` for an element of an array output. The binding
  generator uses this to trim the returned buffer.

**Rules:**
- `out` is valid only on `*mut` pointer parameters.
- `out` on a non-pointer or immutable pointer is a compile error.
- `cap` expression may reference: named parameters that appear before this
  parameter in the list, and `const` values. No references to other `out`
  parameters.
- `cap` and `count` are metadata-only annotations. They do not affect
  compilation, type checking, or codegen. The kernel still receives a pointer
  and writes to it. The annotations tell the binding generator how to allocate
  and trim.
- Parameters without `out` are inputs. This is the default — backward
  compatible with all existing code.
- `out` parameters without `[cap: ...]` are still valid — they appear as
  outputs in metadata, but the binding generator cannot auto-allocate. The
  caller must provide the buffer (same as today). This allows incremental
  adoption.

### Metadata

Output annotations appear in the JSON metadata:

```json
{
  "name": "extract_positions",
  "args": [
    {"name": "text", "type": "*u8", "direction": "in"},
    {"name": "len", "type": "i32", "direction": "in"},
    {"name": "delim", "type": "u8", "direction": "in"},
    {"name": "delim_pos", "type": "*mut i32", "direction": "out",
     "cap": "len", "count": null},
    {"name": "lf_pos", "type": "*mut i32", "direction": "out",
     "cap": "len / 4", "count": null},
    {"name": "counts", "type": "*mut i32", "direction": "out",
     "cap": "3", "count": null}
  ],
  "return_type": null
}
```

New fields on arg objects:
- `"direction"`: `"in"` (default) or `"out"`.
- `"cap"`: string representation of capacity expression, or `null`.
- `"count"`: string path to count output, or `null`.

For backward compatibility, if `"direction"` is missing, treat as `"in"`.

### Binding Generator Changes

**Python (`ea bind --python`):**

For functions with `out` parameters that have `cap` annotations, the generator
produces a function that:
1. Accepts only input parameters.
2. Evaluates `cap` expressions from input values.
3. Allocates NumPy arrays of the appropriate dtype and capacity.
4. Calls the C function with all parameters (inputs + allocated outputs).
5. If `count` is specified, trims the output to actual count.
6. Returns output arrays (single array unwrapped, multiple as tuple).

```python
# Generated:
def extract_positions(text: _np.ndarray, delim: int):
    _len = text.size
    _delim_pos = _np.empty(_len, dtype=_np.int32)
    _lf_pos = _np.empty(_len // 4, dtype=_np.int32)
    _counts = _np.empty(3, dtype=_np.int32)
    _lib.extract_positions(
        text.ctypes.data_as(...), _ct.c_int32(_len), _ct.c_uint8(delim),
        _delim_pos.ctypes.data_as(...), _lf_pos.ctypes.data_as(...),
        _counts.ctypes.data_as(...)
    )
    return _delim_pos, _lf_pos, _counts
```

For `out` parameters without `cap`, the parameter remains in the Python
signature — the caller provides the buffer. This preserves backward
compatibility and allows advanced users to manage their own buffers.

**Rust, C++ (`ea bind --rust`, `--cpp`):**

Similar pattern. Rust returns `Vec<T>` for out params with cap. C++ returns
`std::vector<T>`. Both trim to count if specified.

**PyTorch (`ea bind --pytorch`):**

Out params with cap become returned tensors allocated with `torch.empty(cap)`.

### Implementation

**AST:** Extend `Param`:

```rust
pub struct Param {
    pub name: String,
    pub ty: TypeAnnotation,
    pub span: Span,
    pub output: bool,          // new: true if `out` keyword present
    pub cap: Option<String>,   // new: capacity expression as string
    pub count: Option<String>, // new: count path as string
}
```

**Parser:** In `parse_params()`, before parsing the parameter name, check for
`out` keyword. If present, set `output: true`. After parsing the type, check
for `[`. If present, parse `cap: EXPR` and optionally `count: PATH`, then
expect `]`.

The `cap` expression is parsed as a string (not evaluated) — it is stored in
the AST and emitted to metadata verbatim. The binding generator evaluates it
in the target language. This avoids building an expression evaluator in the
compiler for metadata-only annotations.

**Type checker:** Verify `out` parameters are `*mut` pointers. Error otherwise:
`"'out' annotation requires a mutable pointer (*mut T), got *T"`.
Verify `cap` references only preceding input parameters and constants.

**Metadata:** Extend `generate_json` to emit `direction`, `cap`, and `count`
fields on arg objects.

**Binding generators:** Extend `bind_common.rs` to parse new metadata fields.
Extend each generator to handle output parameters:
- `bind_python.rs`: generate allocation + return pattern.
- `bind_rust.rs`: generate Vec allocation + return.
- `bind_cpp.rs`: generate std::vector allocation + return.
- `bind_pytorch.rs`: generate tensor allocation + return.
- `bind_cmake.rs`: no change (C level, caller manages buffers).

**Backward compatibility:** Functions without `out` annotations produce the
same metadata as before (no `direction`/`cap`/`count` fields). All existing
bindings continue to work unchanged. The binding generators treat missing
`direction` as `"in"`.

**Tests:**
- Basic: `out buf: *mut f32 [cap: n]` → metadata has direction/cap fields.
- Cap expression: `[cap: n / 4]`, `[cap: n * 2 + 16]`, `[cap: 3]` (literal).
- Cap with const: `const BUF_SIZE: i32 = 256; out buf: *mut i32 [cap: BUF_SIZE]`.
- Count: `out buf: *mut i32 [cap: n, count: out_count]`.
- Error: `out` on immutable pointer → compile error.
- Error: `out` on non-pointer → compile error.
- Error: `cap` referencing a later parameter → compile error.
- Backward compat: function without `out` → same metadata as v1.2.
- Python binding: verify generated code allocates and returns correctly.
- Rust binding: verify generated code returns Vec.
- Integration: rewrite eastat's `extract_positions_quoted` with `out`
  annotations, verify the generated Python binding works with simplified
  calling code.

**Estimate:** ~400 lines. AST extension (~20), parser (~60), type checker
(~40), metadata (~30), bind_common (~40), bind_python (~60), bind_rust (~50),
bind_cpp (~50), bind_pytorch (~30), tests (~100+).

---

## What v1.3 Does NOT Include

These were discussed and explicitly deferred:

- **`accum` / reduction kernels** — deferred to v1.4. Reduction has hard
  design questions (merge operation semantics, multi-accumulator ILP
  expression) that need a release cycle of `kernel` usage to inform.
- **`pipeline` / fusion** — deferred to v1.5. Depends on `kernel` being
  mature. Fusion correctness verification is complex and must not ship
  prematurely.
- **`dispatch` / multi-width** — deferred. cpuid dispatch is runtime logic;
  Eä is AOT. The current approach (separate .so per target, host-language
  dispatch) works. Revisit after v1.4 when kernels are shorter and the cost
  of writing two variants is lower.
- **Auto-derived scalar tails** — rejected. Eä does not transform SIMD code
  into scalar code. The programmer writes both.
- **Cap expression evaluation in the compiler** — rejected. Cap expressions
  are strings passed to binding generators. The compiler validates references
  but does not evaluate.

---

## Implementation Order

Features have dependencies:

1. **Compile-time constants** — no dependencies. Ship first because `kernel`
   step can reference constants (e.g., `step STEP`).
2. **Kernel construct** (without tail) — depends on constants (for const
   step references). The `over ... step` desugaring is the foundation.
3. **Tail clause** — extends kernel construct. Cannot exist without `kernel`.
4. **Output annotations** — independent of 1-3. Can be implemented in parallel.

Critical path: Constants → Kernel → Tail (sequential).
Output annotations: parallel with any of the above.

---

## Updated Examples After v1.3

### fma.ea (before)

```
export func fma_f32x8(a: *restrict f32, b: *restrict f32, c: *restrict f32,
                      out: *mut f32, len: i32) {
    let mut i: i32 = 0
    while i + 8 <= len {
        let va: f32x8 = load(a, i)
        let vb: f32x8 = load(b, i)
        let vc: f32x8 = load(c, i)
        store(out, i, fma(va, vb, vc))
        i = i + 8
    }
    while i < len {
        out[i] = a[i] * b[i] + c[i]
        i = i + 1
    }
}
```

### fma.ea (after)

```
const STEP: i32 = 8

export kernel fma(a: *restrict f32, b: *restrict f32, c: *restrict f32,
                  out: *mut f32)
    over i in len step STEP
    tail scalar {
        out[i] = a[i] * b[i] + c[i]
    }
{
    let va: f32x8 = load(a, i)
    let vb: f32x8 = load(b, i)
    let vc: f32x8 = load(c, i)
    store(out, i, fma(va, vb, vc))
}
```

### reduction.ea (no change in v1.3)

Reduction kernels keep `export func` syntax. The `accum` pattern is deferred
to v1.4. This is intentional — `kernel` is for streaming, `func` is for
everything else.

### eastat calling code (before — 6 lines per call)

```python
delim_pos = np.empty(n // 4 + 256, dtype=np.int32)
lf_pos = np.empty(n // 20 + 256, dtype=np.int32)
counts = np.zeros(3, dtype=np.int32)
csv_parse.extract_positions_quoted(text, delim_byte, delim_pos, lf_pos, counts)
n_delims = int(counts[0])
delim_pos = delim_pos[:n_delims]
```

### eastat calling code (after — 1 line)

```python
delim_pos, lf_pos, counts = csv_parse.extract_positions_quoted(text, delim_byte)
```

---

## Verification Criteria for v1.3

All of these must be true before tagging v1.3:

1. `const` declarations work for all integer and float types.
2. `const` values inline correctly (verify via `--emit-llvm`).
3. `export kernel ... over ... step` produces identical LLVM IR to equivalent
   `export func` with hand-written loop.
4. `kernel` with `tail scalar` produces correct results for all input lengths
   (0 through step*10+step-1).
5. `kernel` with `tail mask` produces correct results for all tail lengths.
6. `out` annotations appear correctly in metadata JSON.
7. `ea bind --python` generates auto-allocating functions for `out` params
   with `cap`.
8. All existing tests pass unchanged (no regressions).
9. All existing demos compile unchanged (backward compatibility).
10. No file exceeds 500 lines.
11. No TODOs, no stubs, clippy clean, fmt clean.
12. Specification.md updated to reflect v1.3 features.
13. CHANGELOG.md updated.

---

## Estimated Total

| Feature | New/changed lines | Files touched |
|---------|------------------|---------------|
| Constants | ~150 | ast, parser, typeck, tests |
| Kernel construct | ~350 | ast, parser, typeck, tests |
| Tail clause | ~200 | ast, parser, typeck, tests |
| Output annotations | ~400 | ast, parser, typeck, metadata, bind_*, tests |
| **Total** | **~1,100** | |

Consistent with Eä's pace: v1.0 was ~7,300 lines, v1.2 added ~1,400 for
bind. v1.3 adds ~1,100 for kernel + output annotations. No file should
approach 500 lines if work is distributed properly.
