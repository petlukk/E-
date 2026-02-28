#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === Passing assertions ===

    #[test]
    fn test_static_assert_passing_const() {
        assert_c_interop(
            r#"
            const STEP: i32 = 8
            static_assert(STEP == 8, "STEP must be 8")
            export func get_step() -> i32 { return STEP }
        "#,
            r#"
            #include <stdio.h>
            extern int get_step(void);
            int main() { printf("%d\n", get_step()); return 0; }
        "#,
            "8",
        );
    }

    #[test]
    fn test_static_assert_arithmetic_condition() {
        assert_c_interop(
            r#"
            const STEP: i32 = 8
            static_assert(STEP % 4 == 0, "STEP must be SIMD-aligned")
            export func get_step() -> i32 { return STEP }
        "#,
            r#"
            #include <stdio.h>
            extern int get_step(void);
            int main() { printf("%d\n", get_step()); return 0; }
        "#,
            "8",
        );
    }

    #[test]
    fn test_static_assert_boolean_logic() {
        assert_c_interop(
            r#"
            const X: i32 = 8
            static_assert(X > 0 && X <= 16, "X must be in range 1..16")
            export func get_x() -> i32 { return X }
        "#,
            r#"
            #include <stdio.h>
            extern int get_x(void);
            int main() { printf("%d\n", get_x()); return 0; }
        "#,
            "8",
        );
    }

    #[test]
    fn test_static_assert_multiple() {
        assert_c_interop(
            r#"
            const A: i32 = 4
            const B: i32 = 8
            static_assert(A > 0, "A must be positive")
            static_assert(B > A, "B must exceed A")
            static_assert(B % A == 0, "B must be a multiple of A")
            export func sum() -> i32 { return A + B }
        "#,
            r#"
            #include <stdio.h>
            extern int sum(void);
            int main() { printf("%d\n", sum()); return 0; }
        "#,
            "12",
        );
    }

    #[test]
    fn test_static_assert_float_constant() {
        assert_c_interop(
            r#"
            const PI: f64 = 3.14159
            static_assert(PI > 3.0, "PI must be greater than 3")
            export func get_pi() -> f64 { return PI }
        "#,
            r#"
            #include <stdio.h>
            extern double get_pi(void);
            int main() { printf("%.5f\n", get_pi()); return 0; }
        "#,
            "3.14159",
        );
    }

    #[test]
    fn test_static_assert_with_kernel() {
        assert_c_interop(
            r#"
            const STEP: i32 = 4
            static_assert(STEP % 4 == 0, "STEP must be a multiple of 4")

            export kernel scale(data: *i32, out: *mut i32)
                over i in n step 1
            {
                out[i] = data[i] * STEP
            }
        "#,
            r#"
            #include <stdio.h>
            extern void scale(int*, int*, int);
            int main() {
                int data[3] = {1, 2, 3};
                int out[3];
                scale(data, out, 3);
                printf("%d %d %d\n", out[0], out[1], out[2]);
                return 0;
            }
        "#,
            "4 8 12",
        );
    }

    #[test]
    fn test_static_assert_literal_true() {
        assert_c_interop(
            r#"
            static_assert(1 == 1, "always true")
            export func ok() -> i32 { return 42 }
        "#,
            r#"
            #include <stdio.h>
            extern int ok(void);
            int main() { printf("%d\n", ok()); return 0; }
        "#,
            "42",
        );
    }

    // === Failing assertions ===

    #[test]
    fn test_static_assert_fails_with_message() {
        let source = r#"
            const STEP: i32 = 7
            static_assert(STEP % 4 == 0, "STEP must be SIMD-aligned")
            export func f() -> i32 { return STEP }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("static assertion failed") && msg.contains("STEP must be SIMD-aligned"),
            "expected static assertion failure with message, got: {msg}"
        );
    }

    #[test]
    fn test_static_assert_non_const_error() {
        let source = r#"
            static_assert(x > 0, "x must be positive")
            export func f(x: i32) -> i32 { return x }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("not a compile-time constant"),
            "expected non-const error, got: {msg}"
        );
    }

    #[test]
    fn test_static_assert_false_literal() {
        let source = r#"
            static_assert(1 == 2, "impossible")
            export func f() -> i32 { return 0 }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("static assertion failed"),
            "expected assertion failure, got: {msg}"
        );
    }
}
