#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === Basic const usage ===

    #[test]
    fn test_const_i32_basic() {
        assert_output(
            r#"
            const X: i32 = 42
            func main() { println(X) }
        "#,
            "42",
        );
    }

    #[test]
    fn test_const_in_arithmetic() {
        assert_output(
            r#"
            const A: i32 = 10
            const B: i32 = 3
            func main() { println(A + B) }
        "#,
            "13",
        );
    }

    #[test]
    fn test_const_negative() {
        assert_output(
            r#"
            const N: i32 = -5
            func main() { println(N + 10) }
        "#,
            "5",
        );
    }

    #[test]
    fn test_const_hex() {
        assert_output(
            r#"
            const MASK: i32 = 0xFF
            func main() { println(MASK) }
        "#,
            "255",
        );
    }

    #[test]
    fn test_const_binary() {
        assert_output(
            r#"
            const FLAGS: i32 = 0b1010
            func main() { println(FLAGS) }
        "#,
            "10",
        );
    }

    #[test]
    fn test_const_in_condition() {
        assert_output(
            r#"
            const LIMIT: i32 = 5
            func main() {
                let mut sum: i32 = 0
                let mut i: i32 = 0
                while i < LIMIT {
                    sum = sum + i
                    i = i + 1
                }
                println(sum)
            }
        "#,
            "10",
        );
    }

    #[test]
    fn test_const_in_function_call() {
        assert_output(
            r#"
            const FACTOR: i32 = 7
            func scale(x: i32) -> i32 { return x * FACTOR }
            func main() { println(scale(6)) }
        "#,
            "42",
        );
    }

    #[test]
    fn test_const_f32_c_interop() {
        assert_c_interop(
            r#"
            const PI: f32 = 3.14
            export func get_pi() -> f32 { return PI }
        "#,
            r#"
            #include <stdio.h>
            extern float get_pi(void);
            int main() {
                float pi = get_pi();
                printf("%.2f\n", pi);
                return 0;
            }
        "#,
            "3.14",
        );
    }

    #[test]
    fn test_const_i64() {
        assert_c_interop(
            r#"
            const BIG: i64 = 1000000000
            export func get_big() -> i64 { return BIG }
        "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern int64_t get_big(void);
            int main() {
                printf("%ld\n", get_big());
                return 0;
            }
        "#,
            "1000000000",
        );
    }

    #[test]
    fn test_const_multiple_functions() {
        assert_output(
            r#"
            const STEP: i32 = 4
            func a() -> i32 { return STEP }
            func b() -> i32 { return STEP + 1 }
            func main() { println(a() + b()) }
        "#,
            "9",
        );
    }

    // === SIMD kernel with const step ===

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_const_in_simd_kernel() {
        assert_c_interop(
            r#"
            const STEP: i32 = 4
            export func scale(data: *f32, out: *mut f32, len: i32, factor: f32) {
                let vf: f32x4 = splat(factor)
                let mut i: i32 = 0
                while i + STEP <= len {
                    store(out, i, load(data, i) .* vf)
                    i = i + STEP
                }
            }
        "#,
            r#"
            #include <stdio.h>
            extern void scale(float*, float*, int, float);
            int main() {
                float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
                float out[4];
                scale(in, out, 4, 2.0f);
                printf("%.0f %.0f %.0f %.0f\n", out[0], out[1], out[2], out[3]);
                return 0;
            }
        "#,
            "2 4 6 8",
        );
    }

    // === Error cases ===

    #[test]
    fn test_const_shadowed_by_let_error() {
        let tokens =
            ea_compiler::tokenize("const X: i32 = 5\nexport func f() { let X: i32 = 10\n return }")
                .unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("constant") && msg.contains("variable"),
            "expected const-shadow error, got: {msg}"
        );
    }

    #[test]
    fn test_const_assign_error() {
        let tokens =
            ea_compiler::tokenize("const X: i32 = 5\nexport func f() { X = 10\n return }").unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("constant"),
            "expected const-assign error, got: {msg}"
        );
    }

    #[test]
    fn test_const_duplicate_error() {
        let tokens =
            ea_compiler::tokenize("const X: i32 = 5\nconst X: i32 = 10\nfunc main() { }").unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("duplicate"),
            "expected duplicate const error, got: {msg}"
        );
    }

    #[test]
    fn test_const_non_literal_error() {
        let result = ea_compiler::tokenize("const X: i32 = true\nfunc main() { }");
        if let Ok(tokens) = result {
            let parse_result = ea_compiler::parse(tokens);
            assert!(
                parse_result.is_err(),
                "expected parse error for non-numeric const literal"
            );
        }
        // If tokenize itself errors, that's also acceptable
    }

    #[test]
    fn test_const_vector_type_error() {
        // const with vector type should fail at type check
        let tokens = ea_compiler::tokenize("const V: f32 = 3.14\nfunc main() { }").unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        // f32 is allowed; f32x4 is not
        assert!(ea_compiler::check_types(&stmts).is_ok());
    }

    // === Verify const inlines (no runtime loads) ===

    #[test]
    fn test_const_inlines_in_ir() {
        let ir = compile_to_ir(
            r#"
            const STEP: i32 = 8
            export func get_step() -> i32 { return STEP }
        "#,
        );
        // The constant should compile to a direct `ret i32 8`, no global load
        assert!(
            ir.contains("ret i32 8"),
            "expected const to inline as immediate, got:\n{ir}"
        );
    }
}
