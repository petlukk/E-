#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === Phase 1 tests ===

    #[test]
    fn test_hello_world() {
        assert_output(
            "func main() { println(\"Hello, World!\") }",
            "Hello, World!",
        );
    }

    #[test]
    fn test_println_integer() {
        assert_output("func main() { println(42) }", "42");
    }

    #[test]
    fn test_add_function() {
        assert_output(
            r#"
            func add(a: i32, b: i32) -> i32 { return a + b }
            func main() { println(add(3, 4)) }
        "#,
            "7",
        );
    }

    #[test]
    fn test_export_c_interop() {
        assert_c_interop(
            "export func add(a: i32, b: i32) -> i32 { return a + b }",
            r#"#include <stdio.h>
            extern int add(int, int);
            int main() { printf("%d\n", add(3, 4)); return 0; }"#,
            "7",
        );
    }

    #[test]
    fn test_multiple_functions() {
        assert_output(
            r#"
            func double(x: i32) -> i32 { return x + x }
            func main() { println(double(21)) }
        "#,
            "42",
        );
    }

    #[test]
    fn test_arithmetic_expression() {
        assert_output("func main() { println(2 + 3 * 4) }", "14");
    }

    #[test]
    fn test_export_multiply_c_interop() {
        assert_c_interop(
            "export func multiply(a: i32, b: i32) -> i32 { return a * b }",
            r#"#include <stdio.h>
            extern int multiply(int, int);
            int main() { printf("%d\n", multiply(6, 7)); return 0; }"#,
            "42",
        );
    }

    // === Phase 2 tests ===

    #[test]
    fn test_let_i32() {
        assert_output("func main() { let x: i32 = 42\n println(x) }", "42");
    }

    #[test]
    fn test_let_i64() {
        assert_output(
            "func main() { let x: i64 = 1000000\n println(x) }",
            "1000000",
        );
    }

    #[test]
    fn test_let_f32() {
        assert_output("func main() { let x: f32 = 3.5\n println(x) }", "3.5");
    }

    #[test]
    fn test_let_f64() {
        assert_output("func main() { let x: f64 = 2.5\n println(x) }", "2.5");
    }

    #[test]
    fn test_mut_assignment() {
        assert_output(
            r#"
            func main() {
                let mut x: i32 = 1
                x = 2
                println(x)
            }
        "#,
            "2",
        );
    }

    #[test]
    fn test_int_division() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 10
                let y: i32 = 3
                println(x / y)
            }
        "#,
            "3",
        );
    }

    #[test]
    fn test_int_modulo() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 10
                let y: i32 = 3
                println(x % y)
            }
        "#,
            "1",
        );
    }

    #[test]
    fn test_float_arithmetic() {
        assert_output(
            r#"
            func main() {
                let a: f32 = 2.5
                let b: f32 = 1.5
                println(a + b)
            }
        "#,
            "4",
        );
    }

    #[test]
    fn test_float_division() {
        assert_output(
            r#"
            func main() {
                let a: f64 = 7.5
                let b: f64 = 2.5
                println(a / b)
            }
        "#,
            "3",
        );
    }

    #[test]
    fn test_scale_and_offset() {
        assert_c_interop(
            "export func scale_and_offset(v: f32, s: f32, o: f32) -> f32 { return v * s + o }",
            r#"#include <stdio.h>
            extern float scale_and_offset(float, float, float);
            int main() { printf("%g\n", scale_and_offset(2.0f, 3.0f, 0.5f)); return 0; }"#,
            "6.5",
        );
    }

    #[test]
    fn test_export_f32_c_interop() {
        assert_c_interop(
            "export func add_f32(a: f32, b: f32) -> f32 { return a + b }",
            r#"#include <stdio.h>
            extern float add_f32(float, float);
            int main() { printf("%g\n", add_f32(1.5f, 2.5f)); return 0; }"#,
            "4",
        );
    }

    #[test]
    fn test_export_i64_c_interop() {
        assert_c_interop(
            "export func add_i64(a: i64, b: i64) -> i64 { return a + b }",
            r#"#include <stdio.h>
            #include <stdint.h>
            extern int64_t add_i64(int64_t, int64_t);
            int main() { printf("%ld\n", add_i64(100000, 200000)); return 0; }"#,
            "300000",
        );
    }

    #[test]
    fn test_mut_counter() {
        assert_output(
            r#"
            func main() {
                let mut c: i32 = 0
                c = c + 1
                c = c + 1
                println(c)
            }
        "#,
            "2",
        );
    }

    #[test]
    fn test_negative_literal() {
        assert_output("func main() { let x: i32 = -5\n println(x) }", "-5");
    }

    #[test]
    fn test_multiple_types() {
        assert_output_lines(
            r#"
            func main() {
                let a: i32 = 10
                let b: f32 = 3.5
                println(a)
                println(b)
            }
        "#,
            &["10", "3.5"],
        );
    }
}
