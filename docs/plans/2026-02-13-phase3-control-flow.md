# Phase 3: Control Flow — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete Phase 3 (control flow) by implementing the remaining codegen for comparisons, logical operators, if/else, and while loops — then verify with 15+ new end-to-end tests.

**Architecture:** Lexer, parser, AST, and type checker are already complete. Only codegen needs work: ~70 lines across `codegen/expressions.rs` (comparisons, booleans, logical ops) and `codegen/statements.rs` (if/else, while). A `compile_block` helper reduces duplication.

**Tech Stack:** Rust, inkwell (LLVM 14 bindings), logos lexer, end-to-end tests via compile-link-execute.

---

## Current State

| Layer | Phase 3 Status |
|-------|---------------|
| Lexer | COMPLETE — all tokens (if, else, while, comparisons, logical) |
| Parser | COMPLETE — if/else, while, comparisons, logical, precedence |
| AST | COMPLETE — `Stmt::If`, `Stmt::While`, `Expr::Not`, all `BinaryOp` variants |
| Type Checker | COMPLETE — bool validation, condition checking, logical ops |
| **Codegen** | **INCOMPLETE — 4 non-exhaustive pattern errors prevent compilation** |
| Tests | 17 passing (Phase 1+2). 0 Phase 3 tests. |

**Files to modify:**
- `src/codegen/mod.rs:76-96` — add Bool to `llvm_type` + `resolve_annotation`
- `src/codegen/expressions.rs:24-124` — add Bool literal, Not, comparisons, logical ops
- `src/codegen/statements.rs:55-99` — add If/While + `compile_block` helper
- `tests/end_to_end.rs` — add 15+ Phase 3 tests

**Line counts after completion (estimated):**
- `codegen/expressions.rs`: 193 → ~255 lines
- `codegen/statements.rs`: 100 → ~170 lines
- `codegen/mod.rs`: 149 → ~155 lines
- `tests/end_to_end.rs`: 271 → ~460 lines
- All well under 500-line limit.

---

### Task 1: Add Bool type to codegen infrastructure

**Files:**
- Modify: `src/codegen/mod.rs:76-96`

**Step 1: Add Bool to `llvm_type` method**

In `src/codegen/mod.rs`, the `llvm_type` method (line 76) currently has a `_ =>` wildcard that silently maps Bool to i32. Add explicit Bool mapping before the wildcard:

```rust
pub(crate) fn llvm_type(&self, ty: &Type) -> BasicTypeEnum<'ctx> {
    match ty {
        Type::I32 | Type::IntLiteral => BasicTypeEnum::IntType(self.context.i32_type()),
        Type::I64 => BasicTypeEnum::IntType(self.context.i64_type()),
        Type::F32 => BasicTypeEnum::FloatType(self.context.f32_type()),
        Type::F64 | Type::FloatLiteral => BasicTypeEnum::FloatType(self.context.f64_type()),
        Type::Bool => BasicTypeEnum::IntType(self.context.bool_type()),
        _ => BasicTypeEnum::IntType(self.context.i32_type()),
    }
}
```

**Step 2: Add "bool" to `resolve_annotation` method**

In `src/codegen/mod.rs`, the `resolve_annotation` method (line 86) is missing `"bool"`:

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
    }
}
```

---

### Task 2: Add Bool literal and Not expression to codegen

**Files:**
- Modify: `src/codegen/expressions.rs:24-86`

**Step 1: Add Bool literal to `compile_expr_typed` match**

After the `Expr::Literal(Literal::StringLit(s))` arm (line 51-55), add:

```rust
Expr::Literal(Literal::Bool(b)) => {
    let val = self.context.bool_type().const_int(*b as u64, false);
    Ok(BasicValueEnum::IntValue(val))
}
```

**Step 2: Add Not expression to `compile_expr_typed` match**

After the new Bool literal arm, before the `Expr::Variable` arm, add:

```rust
Expr::Not(inner) => {
    let val = self.compile_expr(inner, function)?;
    let int_val = val.into_int_value();
    let result = self.builder.build_not(int_val, "not")
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
    Ok(BasicValueEnum::IntValue(result))
}
```

---

### Task 3: Add comparison and logical operators to codegen

**Files:**
- Modify: `src/codegen/expressions.rs:89-124`

**Step 1: Add imports**

At the top of `src/codegen/expressions.rs`, add to the existing inkwell import:

```rust
use inkwell::values::{BasicMetadataValueEnum, BasicValueEnum, FunctionValue};
use inkwell::{IntPredicate, FloatPredicate};
```

**Step 2: Refactor `compile_binary` to handle comparisons first**

Replace the `compile_binary` method (lines 89-124) with:

```rust
fn compile_binary(
    &mut self,
    lhs: &Expr,
    op: &BinaryOp,
    rhs: &Expr,
    type_hint: Option<&Type>,
    function: FunctionValue<'ctx>,
) -> crate::error::Result<BasicValueEnum<'ctx>> {
    let hint = self.infer_binary_hint(lhs, rhs, type_hint);
    let left = self.compile_expr_typed(lhs, hint.as_ref(), function)?;
    let right = self.compile_expr_typed(rhs, hint.as_ref(), function)?;

    // Comparison operators — return i1 bool regardless of operand type
    if matches!(op,
        BinaryOp::Less | BinaryOp::Greater | BinaryOp::LessEqual
        | BinaryOp::GreaterEqual | BinaryOp::Equal | BinaryOp::NotEqual
    ) {
        return self.compile_comparison(&left, &right, op);
    }

    // Logical operators — both operands are i1 bools
    if matches!(op, BinaryOp::And | BinaryOp::Or) {
        let l = left.into_int_value();
        let r = right.into_int_value();
        let result = match op {
            BinaryOp::And => self.builder.build_and(l, r, "and"),
            BinaryOp::Or => self.builder.build_or(l, r, "or"),
            _ => unreachable!(),
        }.map_err(|e| CompileError::codegen_error(e.to_string()))?;
        return Ok(BasicValueEnum::IntValue(result));
    }

    // Arithmetic operators
    match (&left, &right) {
        (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
            let result = match op {
                BinaryOp::Add => self.builder.build_int_add(*l, *r, "add"),
                BinaryOp::Subtract => self.builder.build_int_sub(*l, *r, "sub"),
                BinaryOp::Multiply => self.builder.build_int_mul(*l, *r, "mul"),
                BinaryOp::Divide => self.builder.build_int_signed_div(*l, *r, "div"),
                BinaryOp::Modulo => self.builder.build_int_signed_rem(*l, *r, "rem"),
                _ => unreachable!(),
            }.map_err(|e| CompileError::codegen_error(e.to_string()))?;
            Ok(BasicValueEnum::IntValue(result))
        }
        (BasicValueEnum::FloatValue(l), BasicValueEnum::FloatValue(r)) => {
            let result = match op {
                BinaryOp::Add => self.builder.build_float_add(*l, *r, "fadd"),
                BinaryOp::Subtract => self.builder.build_float_sub(*l, *r, "fsub"),
                BinaryOp::Multiply => self.builder.build_float_mul(*l, *r, "fmul"),
                BinaryOp::Divide => self.builder.build_float_div(*l, *r, "fdiv"),
                BinaryOp::Modulo => self.builder.build_float_rem(*l, *r, "frem"),
                _ => unreachable!(),
            }.map_err(|e| CompileError::codegen_error(e.to_string()))?;
            Ok(BasicValueEnum::FloatValue(result))
        }
        _ => Err(CompileError::codegen_error("mismatched operand types in binary expression")),
    }
}
```

**Step 3: Add `compile_comparison` method**

After the `infer_binary_hint` method (line 126-138), add:

```rust
fn compile_comparison(
    &mut self,
    left: &BasicValueEnum<'ctx>,
    right: &BasicValueEnum<'ctx>,
    op: &BinaryOp,
) -> crate::error::Result<BasicValueEnum<'ctx>> {
    match (left, right) {
        (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
            let pred = match op {
                BinaryOp::Less => IntPredicate::SLT,
                BinaryOp::Greater => IntPredicate::SGT,
                BinaryOp::LessEqual => IntPredicate::SLE,
                BinaryOp::GreaterEqual => IntPredicate::SGE,
                BinaryOp::Equal => IntPredicate::EQ,
                BinaryOp::NotEqual => IntPredicate::NE,
                _ => unreachable!(),
            };
            let result = self.builder.build_int_compare(pred, *l, *r, "cmp")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            Ok(BasicValueEnum::IntValue(result))
        }
        (BasicValueEnum::FloatValue(l), BasicValueEnum::FloatValue(r)) => {
            let pred = match op {
                BinaryOp::Less => FloatPredicate::OLT,
                BinaryOp::Greater => FloatPredicate::OGT,
                BinaryOp::LessEqual => FloatPredicate::OLE,
                BinaryOp::GreaterEqual => FloatPredicate::OGE,
                BinaryOp::Equal => FloatPredicate::OEQ,
                BinaryOp::NotEqual => FloatPredicate::ONE,
                _ => unreachable!(),
            };
            let result = self.builder.build_float_compare(pred, *l, *r, "fcmp")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            Ok(BasicValueEnum::IntValue(result))
        }
        _ => Err(CompileError::codegen_error("mismatched types in comparison")),
    }
}
```

---

### Task 4: Add If statement codegen

**Files:**
- Modify: `src/codegen/statements.rs:55-99`

**Step 1: Add `compile_block` helper method**

After the `compile_stmt` method (after line 99), add:

```rust
fn compile_block(
    &mut self,
    stmts: &[Stmt],
    function: FunctionValue<'ctx>,
) -> crate::error::Result<bool> {
    let mut terminated = false;
    for s in stmts {
        if terminated { break; }
        terminated = self.compile_stmt(s, function)?;
    }
    Ok(terminated)
}
```

**Step 2: Add If arm to `compile_stmt`**

In the `compile_stmt` match (line 60), before the `Stmt::Function { .. }` arm (line 95), add:

```rust
Stmt::If { condition, then_body, else_body } => {
    let cond_val = self.compile_expr(condition, function)?;
    let cond_int = cond_val.into_int_value();

    let then_bb = self.context.append_basic_block(function, "then");
    let else_bb = self.context.append_basic_block(function, "else");
    let merge_bb = self.context.append_basic_block(function, "merge");

    self.builder.build_conditional_branch(cond_int, then_bb, else_bb)
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;

    // Then branch
    self.builder.position_at_end(then_bb);
    let then_term = self.compile_block(then_body, function)?;
    if !then_term {
        self.builder.build_unconditional_branch(merge_bb)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
    }

    // Else branch
    self.builder.position_at_end(else_bb);
    if let Some(else_stmts) = else_body {
        let else_term = self.compile_block(else_stmts, function)?;
        if !else_term {
            self.builder.build_unconditional_branch(merge_bb)
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        }
    } else {
        self.builder.build_unconditional_branch(merge_bb)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
    }

    self.builder.position_at_end(merge_bb);
    Ok(false)
}
```

---

### Task 5: Add While loop codegen

**Files:**
- Modify: `src/codegen/statements.rs` (same match block as Task 4)

**Step 1: Add While arm to `compile_stmt`**

After the If arm, before the `Stmt::Function` arm, add:

```rust
Stmt::While { condition, body: while_body } => {
    let cond_bb = self.context.append_basic_block(function, "while_cond");
    let body_bb = self.context.append_basic_block(function, "while_body");
    let exit_bb = self.context.append_basic_block(function, "while_exit");

    self.builder.build_unconditional_branch(cond_bb)
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;

    // Condition check
    self.builder.position_at_end(cond_bb);
    let cond_val = self.compile_expr(condition, function)?;
    let cond_int = cond_val.into_int_value();
    self.builder.build_conditional_branch(cond_int, body_bb, exit_bb)
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;

    // Loop body
    self.builder.position_at_end(body_bb);
    let body_term = self.compile_block(while_body, function)?;
    if !body_term {
        self.builder.build_unconditional_branch(cond_bb)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
    }

    // Exit
    self.builder.position_at_end(exit_bb);
    Ok(false)
}
```

---

### Task 6: Verify compilation

**Step 1: Build the project**

Run: `cargo build --features=llvm`
Expected: Successful compilation with 0 errors.

**Step 2: Run existing tests**

Run: `cargo test --features=llvm`
Expected: All 17 Phase 1+2 tests pass (no regressions).

**Step 3: Run clippy**

Run: `cargo clippy --all-targets --all-features -- -D warnings`
Expected: No warnings.

---

### Task 7: Add simple if/else tests

**Files:**
- Modify: `tests/end_to_end.rs`

**Step 1: Add Phase 3 test section header and first tests**

After the Phase 2 tests section, add:

```rust
// === Phase 3 tests ===

#[test]
fn test_if_true_branch() {
    assert_output(r#"
        func main() {
            let x: i32 = 10
            if x > 5 {
                println(1)
            }
            println(0)
        }
    "#, "1\n0");
}

#[test]
fn test_if_false_branch() {
    assert_output(r#"
        func main() {
            let x: i32 = 3
            if x > 5 {
                println(1)
            }
            println(0)
        }
    "#, "0");
}

#[test]
fn test_if_else() {
    assert_output(r#"
        func main() {
            let x: i32 = 3
            if x > 5 {
                println(1)
            } else {
                println(2)
            }
        }
    "#, "2");
}
```

**Step 2: Run tests**

Run: `cargo test --features=llvm test_if`
Expected: All 3 new tests pass.

---

### Task 8: Add comparison operator tests

**Files:**
- Modify: `tests/end_to_end.rs`

**Step 1: Add comparison tests**

```rust
#[test]
fn test_comparisons() {
    assert_output(r#"
        func check(a: i32, b: i32) -> i32 {
            if a < b { return 1 }
            return 0
        }
        func main() {
            println(check(1, 2))
            println(check(2, 1))
        }
    "#, "1\n0");
}

#[test]
fn test_less_equal() {
    assert_output(r#"
        func check(a: i32, b: i32) -> i32 {
            if a <= b { return 1 }
            return 0
        }
        func main() {
            println(check(3, 3))
            println(check(4, 3))
        }
    "#, "1\n0");
}

#[test]
fn test_equality() {
    assert_output(r#"
        func check(a: i32, b: i32) -> i32 {
            if a == b { return 1 }
            if a != b { return 2 }
            return 0
        }
        func main() {
            println(check(5, 5))
            println(check(5, 6))
        }
    "#, "1\n2");
}
```

**Step 2: Run tests**

Run: `cargo test --features=llvm test_comparisons test_less_equal test_equality`
Expected: All pass.

---

### Task 9: Add while loop tests

**Files:**
- Modify: `tests/end_to_end.rs`

**Step 1: Add while loop tests**

```rust
#[test]
fn test_while_loop() {
    assert_output(r#"
        func main() {
            let mut i: i32 = 0
            while i < 5 {
                i = i + 1
            }
            println(i)
        }
    "#, "5");
}

#[test]
fn test_sum_range() {
    assert_output(r#"
        func main() {
            let mut sum: i32 = 0
            let mut i: i32 = 1
            while i <= 10 {
                sum = sum + i
                i = i + 1
            }
            println(sum)
        }
    "#, "55");
}
```

**Step 2: Run tests**

Run: `cargo test --features=llvm test_while_loop test_sum_range`
Expected: Both pass.

---

### Task 10: Add logical operator and NOT tests

**Files:**
- Modify: `tests/end_to_end.rs`

**Step 1: Add logical operator tests**

```rust
#[test]
fn test_logical_and() {
    assert_output(r#"
        func main() {
            let x: i32 = 5
            if x > 0 && x < 10 {
                println(1)
            } else {
                println(0)
            }
        }
    "#, "1");
}

#[test]
fn test_logical_or() {
    assert_output(r#"
        func main() {
            let x: i32 = 15
            if x < 0 || x > 10 {
                println(1)
            } else {
                println(0)
            }
        }
    "#, "1");
}

#[test]
fn test_not_operator() {
    assert_output(r#"
        func main() {
            let x: i32 = 5
            if !(x > 10) {
                println(1)
            } else {
                println(0)
            }
        }
    "#, "1");
}
```

**Step 2: Run tests**

Run: `cargo test --features=llvm test_logical test_not`
Expected: All pass.

---

### Task 11: Add nested control flow tests

**Files:**
- Modify: `tests/end_to_end.rs`

**Step 1: Add nested control flow tests**

```rust
#[test]
fn test_nested_if_in_while() {
    assert_output(r#"
        func main() {
            let mut i: i32 = 0
            let mut evens: i32 = 0
            while i < 10 {
                if i % 2 == 0 {
                    evens = evens + 1
                }
                i = i + 1
            }
            println(evens)
        }
    "#, "5");
}

#[test]
fn test_early_return_from_if() {
    assert_output(r#"
        func abs(x: i32) -> i32 {
            if x < 0 {
                return 0 - x
            }
            return x
        }
        func main() {
            println(abs(-5))
            println(abs(3))
        }
    "#, "5\n3");
}

#[test]
fn test_fizzbuzz_style() {
    assert_output(r#"
        func classify(n: i32) -> i32 {
            if n % 3 == 0 {
                return 3
            } else {
                if n % 5 == 0 {
                    return 5
                } else {
                    return n
                }
            }
        }
        func main() {
            println(classify(9))
            println(classify(10))
            println(classify(7))
        }
    "#, "3\n5\n7");
}
```

**Step 2: Run tests**

Run: `cargo test --features=llvm test_nested test_early_return test_fizzbuzz`
Expected: All pass.

---

### Task 12: Add C interop tests for Phase 3

**Files:**
- Modify: `tests/end_to_end.rs`

**Step 1: Add C interop tests**

```rust
#[test]
fn test_export_sum_range_c_interop() {
    assert_c_interop(
        r#"
            export func sum_range(start: i32, end: i32) -> i32 {
                let mut sum: i32 = 0
                let mut i: i32 = start
                while i <= end {
                    sum = sum + i
                    i = i + 1
                }
                return sum
            }
        "#,
        r#"#include <stdio.h>
        extern int sum_range(int, int);
        int main() { printf("%d\n", sum_range(1, 100)); return 0; }"#,
        "5050",
    );
}

#[test]
fn test_export_max_c_interop() {
    assert_c_interop(
        r#"
            export func max(a: i32, b: i32) -> i32 {
                if a > b {
                    return a
                }
                return b
            }
        "#,
        r#"#include <stdio.h>
        extern int max(int, int);
        int main() {
            printf("%d\n", max(10, 20));
            printf("%d\n", max(30, 5));
            return 0;
        }"#,
        "20\n30",
    );
}

#[test]
fn test_export_clamp_c_interop() {
    assert_c_interop(
        r#"
            export func clamp(val: i32, lo: i32, hi: i32) -> i32 {
                if val < lo { return lo }
                if val > hi { return hi }
                return val
            }
        "#,
        r#"#include <stdio.h>
        extern int clamp(int, int, int);
        int main() {
            printf("%d\n", clamp(-5, 0, 10));
            printf("%d\n", clamp(15, 0, 10));
            printf("%d\n", clamp(5, 0, 10));
            return 0;
        }"#,
        "0\n10\n5",
    );
}
```

**Step 2: Run tests**

Run: `cargo test --features=llvm test_export_sum_range test_export_max test_export_clamp`
Expected: All pass.

---

### Task 13: Final verification

**Step 1: Run full test suite**

Run: `cargo test --features=llvm`
Expected: 32+ tests pass (17 Phase 1+2 + 15 Phase 3). 0 failures.

**Step 2: Run clippy and fmt**

Run: `cargo fmt && cargo clippy --all-targets --all-features -- -D warnings`
Expected: Clean.

**Step 3: Check line counts**

Run: `wc -l src/**/*.rs src/*.rs`
Expected: No file exceeds 500 lines.

**Step 4: Commit**

```bash
git add src/codegen/mod.rs src/codegen/expressions.rs src/codegen/statements.rs tests/end_to_end.rs
git commit -m "feat: complete Phase 3 — control flow (if/else, while, comparisons, logical ops)

Add codegen for:
- Bool type (i1), bool literals, NOT operator
- Comparison operators (<, >, <=, >=, ==, !=) for int and float
- Logical operators (&&, ||)
- If/else statement with proper basic block structure
- While loop with condition/body/exit blocks
- compile_block helper for shared block compilation

15 new end-to-end tests including C interop. 32+ total tests passing."
```

---

## Key Technical Notes

**Bool representation:** LLVM `i1` via `context.bool_type()`. Comparisons naturally return `i1`. Conditional branches consume `i1`.

**Comparison return type mismatch:** `build_float_compare` returns `IntValue` (i1), not `FloatValue`. This is why comparisons are handled separately from arithmetic — they're extracted before the IntValue/FloatValue dispatch.

**If/else basic block structure:** Always creates then/else/merge blocks. When there's no else body, the else block just branches to merge. Merge block is always positioned at the end for subsequent code.

**While loop structure:** Three blocks — cond/body/exit. Entry branches to cond. Cond checks condition and branches to body or exit. Body executes and branches back to cond. Exit continues.

**No phi nodes needed:** Variables use alloca/store/load pattern (not SSA), so no phi nodes are required for mutable variables across control flow. LLVM's mem2reg pass handles promotion to SSA.
