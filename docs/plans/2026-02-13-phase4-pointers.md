# Phase 4: Functions + Pointers — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add pointer types (`*T`, `*mut T`), pointer indexing (`ptr[i]`), index assignment (`ptr[i] = val`), and verify recursion — making Eä useful for real C interop where functions operate on arrays.

**Architecture:** Changes span all compiler layers. Lexer adds `[`/`]`. AST gains `TypeAnnotation::Pointer`, `Expr::Index`, `Stmt::IndexAssign`. Type checker validates pointer operations and mutability. Codegen maps pointers to LLVM typed pointers and uses GEP for indexing. Recursion already works (two-pass declaration) — just needs tests.

**Tech Stack:** Rust, inkwell (LLVM 14), logos lexer, GEP instructions for pointer arithmetic.

---

## Current State

- **39 passing tests** (22 Phase 1+2, 17 Phase 3)
- Recursion: Already works (two-pass function declaration). Needs test coverage.
- `*` token: Already exists as `Star`. `mut` token: Already exists as `Mut`.
- Missing: `[`/`]` tokens, pointer type annotations, index expressions, index assignment.

**Files to modify** (11 files):

| File | Current Lines | Changes |
|------|--------------|---------|
| `src/lexer/mod.rs` | 217 | +2 tokens (`[`, `]`) |
| `src/lexer/tokens.rs` | 49 | +2 display entries |
| `src/ast/mod.rs` | 195 | +Pointer type, Index expr, IndexAssign stmt |
| `src/parser/mod.rs` | 186 | +pointer type parsing |
| `src/parser/expressions.rs` | 191 | +postfix `[expr]` indexing |
| `src/parser/statements.rs` | 101 | +index assignment `name[i] = val` |
| `src/typeck/types.rs` | 87 | +Pointer variant, helpers, resolve |
| `src/typeck/check.rs` | 246 | +Index check, IndexAssign check |
| `src/codegen/mod.rs` | 167 | +pointer in llvm_type, resolve_annotation |
| `src/codegen/expressions.rs` | 314 | +Index expr codegen |
| `src/codegen/statements.rs` | 210 | +IndexAssign codegen |
| `tests/phase4.rs` | NEW | ~10 new tests |

All files stay well under 500 lines.

---

### Task 1: Lexer + AST foundation

**Files:**
- Modify: `src/lexer/mod.rs:148-156`
- Modify: `src/lexer/tokens.rs`
- Modify: `src/ast/mod.rs`

**Step 1: Add bracket tokens to lexer**

In `src/lexer/mod.rs`, after the `Bang` token (line 149), add:

```rust
#[token("[")]
LeftBracket,
#[token("]")]
RightBracket,
```

**Step 2: Add bracket display to tokens.rs**

In `src/lexer/tokens.rs`, add entries for the new tokens in the `token_name` function's match (follow existing pattern):

```rust
TokenKind::LeftBracket => "[",
TokenKind::RightBracket => "]",
```

**Step 3: Extend TypeAnnotation in AST**

In `src/ast/mod.rs`, replace the `TypeAnnotation` enum (lines 89-92):

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum TypeAnnotation {
    Named(String),
    Pointer {
        mutable: bool,
        inner: Box<TypeAnnotation>,
    },
}
```

Update its Display impl (lines 94-100):

```rust
impl fmt::Display for TypeAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeAnnotation::Named(name) => write!(f, "{name}"),
            TypeAnnotation::Pointer { mutable, inner } => {
                if *mutable {
                    write!(f, "*mut {inner}")
                } else {
                    write!(f, "*{inner}")
                }
            }
        }
    }
}
```

**Step 4: Add Index expression to AST**

In `src/ast/mod.rs`, add to the `Expr` enum (after `Not`):

```rust
Index {
    object: Box<Expr>,
    index: Box<Expr>,
},
```

Add its Display arm:

```rust
Expr::Index { object, index } => write!(f, "{object}[{index}]"),
```

**Step 5: Add IndexAssign statement to AST**

In `src/ast/mod.rs`, add to the `Stmt` enum (after `Assign`):

```rust
IndexAssign {
    object: String,
    index: Expr,
    value: Expr,
},
```

Add its Display arm:

```rust
Stmt::IndexAssign { object, index, .. } => write!(f, "{object}[{index}] = ..."),
```

---

### Task 2: Parser — pointer types + indexing

**Files:**
- Modify: `src/parser/mod.rs:92-114`
- Modify: `src/parser/expressions.rs:167-177`
- Modify: `src/parser/statements.rs:29-40`

**Step 1: Add pointer type parsing**

In `src/parser/mod.rs`, replace the `parse_type` method (lines 92-114) with:

```rust
pub(super) fn parse_type(&mut self) -> crate::error::Result<TypeAnnotation> {
    // Pointer types: *T or *mut T
    if self.check(TokenKind::Star) {
        self.advance(); // consume *
        let mutable = if self.check(TokenKind::Mut) {
            self.advance();
            true
        } else {
            false
        };
        let inner = self.parse_type()?;
        return Ok(TypeAnnotation::Pointer {
            mutable,
            inner: Box::new(inner),
        });
    }

    let type_tokens = [
        TokenKind::I32,
        TokenKind::I64,
        TokenKind::F32,
        TokenKind::F64,
        TokenKind::Bool,
    ];
    for tk in &type_tokens {
        if self.check(tk.clone()) {
            let token = self.advance().clone();
            return Ok(TypeAnnotation::Named(token.lexeme.clone()));
        }
    }
    if self.check(TokenKind::Identifier) {
        let token = self.advance().clone();
        return Ok(TypeAnnotation::Named(token.lexeme.clone()));
    }
    Err(CompileError::parse_error(
        format!("expected type, found {:?}", self.peek_kind()),
        self.current_position(),
    ))
}
```

**Step 2: Add postfix indexing to expression parser**

In `src/parser/expressions.rs`, in the `primary` method, replace the identifier handling (lines 167-177) with:

```rust
if self.check(TokenKind::Identifier) {
    let token = self.advance().clone();
    let name = token.lexeme.clone();
    if self.check(TokenKind::LeftParen) {
        self.advance();
        let args = self.parse_args()?;
        self.expect_kind(TokenKind::RightParen, "expected ')' after arguments")?;
        return Ok(Expr::Call { name, args });
    }
    let mut expr = Expr::Variable(name);
    // Postfix indexing: name[expr]
    while self.check(TokenKind::LeftBracket) {
        self.advance(); // consume [
        let index = self.expression()?;
        self.expect_kind(TokenKind::RightBracket, "expected ']' after index")?;
        expr = Expr::Index {
            object: Box::new(expr),
            index: Box::new(index),
        };
    }
    return Ok(expr);
}
```

**Step 3: Add index assignment to statement parser**

In `src/parser/statements.rs`, replace the assignment handling (lines 29-37) with a version that also handles index assignment. Replace from line 29 to line 40 (the `else` expression parsing):

```rust
// Assignment: name = value  OR  index assign via expression fallthrough
if self.check(TokenKind::Identifier) && self.peek_next_kind() == Some(&TokenKind::Equals) {
    let name = self.advance().lexeme.clone();
    self.advance(); // consume '='
    let value = self.expression()?;
    return Ok(Stmt::Assign {
        target: name,
        value,
    });
}

// Parse expression (could be index expr, function call, etc.)
let expr = self.expression()?;

// Check for index assignment: expr[i] = value
if self.check(TokenKind::Equals) {
    self.advance(); // consume '='
    let value = self.expression()?;
    if let Expr::Index { object, index } = expr {
        if let Expr::Variable(name) = *object {
            return Ok(Stmt::IndexAssign {
                object: name,
                index: *index,
                value,
            });
        }
    }
    return Err(crate::error::CompileError::parse_error(
        "invalid assignment target",
        self.current_position(),
    ));
}

Ok(Stmt::ExprStmt(expr))
```

Note: This replaces the final `let expr = self.expression()?; Ok(Stmt::ExprStmt(expr))` at the end of the method too.

---

### Task 3: Type system — pointer types

**Files:**
- Modify: `src/typeck/types.rs`
- Modify: `src/typeck/check.rs`

**Step 1: Add Pointer variant to Type enum**

In `src/typeck/types.rs`, add to the `Type` enum (after `Void`):

```rust
Pointer {
    mutable: bool,
    inner: Box<Type>,
},
```

**Step 2: Add pointer helpers to Type impl**

In `src/typeck/types.rs`, add to the `impl Type` block:

```rust
pub fn is_pointer(&self) -> bool {
    matches!(self, Type::Pointer { .. })
}

pub fn pointee(&self) -> Option<&Type> {
    match self {
        Type::Pointer { inner, .. } => Some(inner),
        _ => None,
    }
}
```

**Step 3: Add Pointer to resolve_type**

In `src/typeck/types.rs`, in the `resolve_type` function, add a new match arm for `TypeAnnotation::Pointer`:

```rust
pub fn resolve_type(ty: &TypeAnnotation) -> crate::error::Result<Type> {
    match ty {
        TypeAnnotation::Named(name) => match name.as_str() {
            "i32" => Ok(Type::I32),
            "i64" => Ok(Type::I64),
            "f32" => Ok(Type::F32),
            "f64" => Ok(Type::F64),
            "bool" => Ok(Type::Bool),
            other => Err(CompileError::type_error(
                format!("unknown type '{other}'"),
                Position::default(),
            )),
        },
        TypeAnnotation::Pointer { mutable, inner } => {
            let inner_type = resolve_type(inner)?;
            Ok(Type::Pointer {
                mutable: *mutable,
                inner: Box::new(inner_type),
            })
        }
    }
}
```

**Step 4: Add Index expression type checking**

In `src/typeck/check.rs`, in the `check_expr` method, add a new match arm before `Expr::Binary` (after `Expr::Not`):

```rust
Expr::Index { object, index } => {
    let obj_type = self.check_expr(object, locals)?;
    let idx_type = self.check_expr(index, locals)?;
    if !idx_type.is_integer() {
        return Err(CompileError::type_error(
            format!("index must be integer, got {idx_type:?}"),
            Position::default(),
        ));
    }
    match obj_type.pointee() {
        Some(inner) => Ok(inner.clone()),
        None => Err(CompileError::type_error(
            format!("cannot index non-pointer type {obj_type:?}"),
            Position::default(),
        )),
    }
}
```

**Step 5: Add IndexAssign statement type checking**

In `src/typeck/check.rs`, in the `check_body` method, add a new match arm before `Stmt::Function`:

```rust
Stmt::IndexAssign { object, index, value } => {
    let (var_type, _) = locals.get(object).cloned().ok_or_else(|| {
        CompileError::type_error(
            format!("undefined variable '{object}'"),
            Position::default(),
        )
    })?;
    match &var_type {
        Type::Pointer { mutable, inner } => {
            if !mutable {
                return Err(CompileError::type_error(
                    format!("cannot write through immutable pointer '{object}'"),
                    Position::default(),
                ));
            }
            let idx_type = self.check_expr(index, locals)?;
            if !idx_type.is_integer() {
                return Err(CompileError::type_error(
                    format!("index must be integer, got {idx_type:?}"),
                    Position::default(),
                ));
            }
            let val_type = self.check_expr(value, locals)?;
            if !types::types_compatible(&val_type, inner) {
                return Err(CompileError::type_error(
                    format!(
                        "cannot assign {val_type:?} to element of {var_type:?}"
                    ),
                    Position::default(),
                ));
            }
        }
        _ => {
            return Err(CompileError::type_error(
                format!("cannot index-assign to non-pointer '{object}'"),
                Position::default(),
            ));
        }
    }
}
```

---

### Task 4: Codegen — pointer types + indexing

**Files:**
- Modify: `src/codegen/mod.rs:92-114`
- Modify: `src/codegen/expressions.rs:72-81`
- Modify: `src/codegen/statements.rs:87-95`

**Step 1: Add pointer to `llvm_type`**

In `src/codegen/mod.rs`, replace the `llvm_type` method (lines 92-101):

```rust
pub(crate) fn llvm_type(&self, ty: &Type) -> BasicTypeEnum<'ctx> {
    match ty {
        Type::I32 | Type::IntLiteral => BasicTypeEnum::IntType(self.context.i32_type()),
        Type::I64 => BasicTypeEnum::IntType(self.context.i64_type()),
        Type::F32 => BasicTypeEnum::FloatType(self.context.f32_type()),
        Type::F64 | Type::FloatLiteral => BasicTypeEnum::FloatType(self.context.f64_type()),
        Type::Bool => BasicTypeEnum::IntType(self.context.bool_type()),
        Type::Pointer { inner, .. } => {
            let inner_ty = self.llvm_type(inner);
            BasicTypeEnum::PointerType(inner_ty.ptr_type(AddressSpace::default()))
        }
        _ => BasicTypeEnum::IntType(self.context.i32_type()),
    }
}
```

Note: `BasicTypeEnum` has a method `ptr_type(AddressSpace)` that works on any variant.

**Step 2: Add pointer to `resolve_annotation`**

In `src/codegen/mod.rs`, replace the `resolve_annotation` method (lines 103-114):

```rust
pub(crate) fn resolve_annotation(&self, ann: &TypeAnnotation) -> Type {
    match ann {
        TypeAnnotation::Named(name) => match name.as_str() {
            "i32" => Type::I32,
            "i64" => Type::I64,
            "f32" => Type::F32,
            "f64" => Type::F64,
            "bool" => Type::Bool,
            _ => Type::I32,
        },
        TypeAnnotation::Pointer { mutable, inner } => {
            let inner_type = self.resolve_annotation(inner);
            Type::Pointer {
                mutable: *mutable,
                inner: Box::new(inner_type),
            }
        }
    }
}
```

**Step 3: Add Index expression codegen**

In `src/codegen/expressions.rs`, in the `compile_expr_typed` match, add a new arm before `Expr::Binary` (after `Expr::Variable`):

```rust
Expr::Index { object, index } => {
    let ptr_val = self.compile_expr(object, function)?;
    let ptr = ptr_val.into_pointer_value();
    let idx = self.compile_expr(index, function)?.into_int_value();
    let elem_ptr = unsafe { self.builder.build_gep(ptr, &[idx], "elemptr") }
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
    let val = self
        .builder
        .build_load(elem_ptr, "elem")
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
    Ok(val)
}
```

**Step 4: Add IndexAssign statement codegen**

In `src/codegen/statements.rs`, in the `compile_stmt` match, add a new arm before `Stmt::If` (after `Stmt::ExprStmt`):

```rust
Stmt::IndexAssign {
    object,
    index,
    value,
} => {
    let (ptr_alloca, var_type) =
        self.variables.get(object).cloned().ok_or_else(|| {
            CompileError::codegen_error(format!("undefined variable '{object}'"))
        })?;
    let ptr = self
        .builder
        .build_load(ptr_alloca, object)
        .map_err(|e| CompileError::codegen_error(e.to_string()))?
        .into_pointer_value();
    let idx = self.compile_expr(index, function)?.into_int_value();
    let elem_ptr = unsafe { self.builder.build_gep(ptr, &[idx], "elemptr") }
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
    let inner_type = match &var_type {
        Type::Pointer { inner, .. } => inner.as_ref().clone(),
        _ => {
            return Err(CompileError::codegen_error(
                "index-assign on non-pointer type",
            ))
        }
    };
    let val = self.compile_expr_typed(value, Some(&inner_type), function)?;
    self.builder
        .build_store(elem_ptr, val)
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
    Ok(false)
}
```

**Step 5: Add Stmt::IndexAssign import**

In `src/codegen/statements.rs`, the existing import `use crate::ast::{Stmt, TypeAnnotation};` also needs `Type` for the pointer match. Add:

```rust
use crate::typeck::Type;
```

---

### Task 5: Verify compilation

**Step 1: Build**

Run: `cargo build --features=llvm`
Expected: 0 errors.

**Step 2: Existing tests**

Run: `cargo test --features=llvm`
Expected: All 39 existing tests pass.

**Step 3: Clippy**

Run: `cargo clippy --all-targets --all-features -- -D warnings`
Expected: Clean (may have warnings about `unsafe` blocks — address with `#[allow(unsafe_code)]` or similar if needed).

---

### Task 6: Add Phase 4 tests

**Files:**
- Create: `tests/phase4.rs`

Create `tests/phase4.rs` with the following content:

```rust
#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === Phase 4: Functions + Pointers tests ===

    #[test]
    fn test_recursive_fibonacci() {
        assert_output_lines(
            r#"
            func fibonacci(n: i32) -> i32 {
                if n <= 1 { return n }
                return fibonacci(n - 1) + fibonacci(n - 2)
            }
            func main() {
                println(fibonacci(0))
                println(fibonacci(1))
                println(fibonacci(5))
                println(fibonacci(10))
            }
        "#,
            &["0", "1", "5", "55"],
        );
    }

    #[test]
    fn test_mutual_function_calls() {
        assert_output(
            r#"
            func double(x: i32) -> i32 { return x * 2 }
            func quad(x: i32) -> i32 { return double(double(x)) }
            func main() { println(quad(3)) }
        "#,
            "12",
        );
    }

    #[test]
    fn test_pointer_read_c_interop() {
        assert_c_interop(
            r#"
            export func sum_array(data: *i32, len: i32) -> i32 {
                let mut total: i32 = 0
                let mut i: i32 = 0
                while i < len {
                    total = total + data[i]
                    i = i + 1
                }
                return total
            }
        "#,
            r#"#include <stdio.h>
            extern int sum_array(const int*, int);
            int main() {
                int arr[] = {10, 20, 30, 40, 50};
                printf("%d\n", sum_array(arr, 5));
                return 0;
            }"#,
            "150",
        );
    }

    #[test]
    fn test_pointer_write_c_interop() {
        assert_c_interop(
            r#"
            export func fill_zeros(out: *mut i32, len: i32) {
                let mut i: i32 = 0
                while i < len {
                    out[i] = 0
                    i = i + 1
                }
            }
        "#,
            r#"#include <stdio.h>
            extern void fill_zeros(int*, int);
            int main() {
                int arr[] = {1, 2, 3, 4};
                fill_zeros(arr, 4);
                printf("%d %d %d %d\n", arr[0], arr[1], arr[2], arr[3]);
                return 0;
            }"#,
            "0 0 0 0",
        );
    }

    #[test]
    fn test_fill_fibonacci_c_interop() {
        assert_c_interop(
            r#"
            func fibonacci(n: i32) -> i32 {
                if n <= 1 { return n }
                return fibonacci(n - 1) + fibonacci(n - 2)
            }
            export func fill_fibonacci(out: *mut i32, count: i32) {
                let mut i: i32 = 0
                while i < count {
                    out[i] = fibonacci(i)
                    i = i + 1
                }
            }
        "#,
            r#"#include <stdio.h>
            extern void fill_fibonacci(int*, int);
            int main() {
                int buf[10];
                fill_fibonacci(buf, 10);
                for (int i = 0; i < 10; i++) printf("%d ", buf[i]);
                printf("\n");
                return 0;
            }"#,
            "0 1 1 2 3 5 8 13 21 34",
        );
    }

    #[test]
    fn test_float_pointer_c_interop() {
        assert_c_interop(
            r#"
            export func scale_array(data: *mut f32, len: i32, factor: f32) {
                let mut i: i32 = 0
                while i < len {
                    data[i] = data[i] * factor
                    i = i + 1
                }
            }
        "#,
            r#"#include <stdio.h>
            extern void scale_array(float*, int, float);
            int main() {
                float arr[] = {1.0f, 2.0f, 3.0f, 4.0f};
                scale_array(arr, 4, 2.5f);
                printf("%g %g %g %g\n", arr[0], arr[1], arr[2], arr[3]);
                return 0;
            }"#,
            "2.5 5 7.5 10",
        );
    }

    #[test]
    fn test_two_pointer_params_c_interop() {
        assert_c_interop(
            r#"
            export func add_arrays(a: *i32, b: *i32, out: *mut i32, len: i32) {
                let mut i: i32 = 0
                while i < len {
                    out[i] = a[i] + b[i]
                    i = i + 1
                }
            }
        "#,
            r#"#include <stdio.h>
            extern void add_arrays(const int*, const int*, int*, int);
            int main() {
                int a[] = {1, 2, 3, 4};
                int b[] = {10, 20, 30, 40};
                int out[4];
                add_arrays(a, b, out, 4);
                printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
                return 0;
            }"#,
            "11 22 33 44",
        );
    }

    #[test]
    fn test_pointer_find_max_c_interop() {
        assert_c_interop(
            r#"
            export func find_max(data: *i32, len: i32) -> i32 {
                let mut best: i32 = data[0]
                let mut i: i32 = 1
                while i < len {
                    if data[i] > best {
                        best = data[i]
                    }
                    i = i + 1
                }
                return best
            }
        "#,
            r#"#include <stdio.h>
            extern int find_max(const int*, int);
            int main() {
                int arr[] = {3, 7, 2, 9, 4};
                printf("%d\n", find_max(arr, 5));
                return 0;
            }"#,
            "9",
        );
    }

    #[test]
    fn test_dot_product_c_interop() {
        assert_c_interop(
            r#"
            export func dot_product(a: *f32, b: *f32, len: i32) -> f32 {
                let mut sum: f32 = 0.0
                let mut i: i32 = 0
                while i < len {
                    sum = sum + a[i] * b[i]
                    i = i + 1
                }
                return sum
            }
        "#,
            r#"#include <stdio.h>
            extern float dot_product(const float*, const float*, int);
            int main() {
                float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
                float b[] = {5.0f, 6.0f, 7.0f, 8.0f};
                printf("%g\n", dot_product(a, b, 4));
                return 0;
            }"#,
            "70",
        );
    }

    #[test]
    fn test_copy_array_c_interop() {
        assert_c_interop(
            r#"
            export func copy(src: *i32, dst: *mut i32, len: i32) {
                let mut i: i32 = 0
                while i < len {
                    dst[i] = src[i]
                    i = i + 1
                }
            }
        "#,
            r#"#include <stdio.h>
            extern void copy(const int*, int*, int);
            int main() {
                int src[] = {100, 200, 300};
                int dst[3] = {0};
                copy(src, dst, 3);
                printf("%d %d %d\n", dst[0], dst[1], dst[2]);
                return 0;
            }"#,
            "100 200 300",
        );
    }
}
```

**Step 2: Run tests**

Run: `cargo test --features=llvm`
Expected: 49+ tests pass (39 existing + 10 new).

---

### Task 7: Final verification

**Step 1: Format and lint**

Run: `cargo fmt && cargo clippy --all-targets --all-features -- -D warnings`
Expected: Clean.

**Step 2: Line counts**

Run: `wc -l src/**/*.rs src/*.rs tests/*.rs tests/common/mod.rs`
Expected: No file exceeds 500 lines.

**Step 3: Full test suite**

Run: `cargo test --features=llvm`
Expected: 49+ tests, 0 failures.

---

## Key Technical Notes

**LLVM 14 typed pointers:** `*i32` maps to `i32*` via `context.i32_type().ptr_type(AddressSpace::default())`. In `BasicTypeEnum`, this is the `PointerType` variant. The `ptr_type()` method exists on all `BasicTypeEnum` variants.

**GEP for indexing:** `builder.build_gep(ptr, &[index], "elemptr")` returns a `PointerValue` to the element. Then `build_load` reads the value. The GEP is `unsafe` in inkwell because out-of-bounds access is undefined behavior (same as C).

**Pointer params via alloca:** The existing alloca/store/load pattern handles pointer parameters naturally. For a `*i32` param, `llvm_type` returns `i32*`, `alloca` creates `i32**`, store puts the pointer in, and load retrieves it. No special handling needed.

**`*T` vs `*mut T`:** Both compile to the same LLVM pointer type. Mutability is enforced at the type checker level only — `*T` prevents `IndexAssign`, while `*mut T` allows it. This matches C's `const T*` vs `T*` distinction.

**Recursion:** Already works because `compile_program` does a two-pass approach: declare all functions first, then compile bodies. No changes needed — just test coverage.
