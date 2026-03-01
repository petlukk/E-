#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === Error: assign to loop variable ===

    #[test]
    fn test_kernel_assign_to_loop_var_error() {
        let tokens = ea_compiler::tokenize(
            "export kernel k(out: *mut i32) over i in n step 1 { i = 10\n out[i] = 0 }",
        )
        .unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::desugar(stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("cannot assign to loop variable"),
            "expected loop variable error, got: {msg}"
        );
    }

    // === Error: step 0 ===

    #[test]
    fn test_kernel_step_zero_error() {
        let result = ea_compiler::tokenize(
            "export kernel k(out: *mut i32) over i in n step 0 { out[i] = 0 }",
        )
        .and_then(ea_compiler::parse);
        assert!(result.is_err(), "expected parse error for step 0");
    }

    // === Error: return with value inside kernel ===

    #[test]
    fn test_kernel_return_value_error() {
        let tokens = ea_compiler::tokenize(
            "export kernel k(data: *i32) over i in n step 1 { return data[i] }",
        )
        .unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("no return type") || msg.contains("returns"),
            "expected void return error, got: {msg}"
        );
    }

    // === Error: range bound collides with param ===

    #[test]
    fn test_kernel_range_bound_collision_error() {
        let tokens = ea_compiler::tokenize(
            "export kernel k(n: *i32, out: *mut i32) over i in n step 1 { out[i] = n[i] }",
        )
        .unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::desugar(stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("collides"),
            "expected collision error, got: {msg}"
        );
    }

    // === Kernel with if/while in body ===

    #[test]
    fn test_kernel_with_control_flow() {
        assert_c_interop(
            r#"
            export kernel clamp(data: *i32, out: *mut i32, lo: i32, hi: i32)
                over i in n step 1
            {
                let v: i32 = data[i]
                if v < lo {
                    out[i] = lo
                } else {
                    if v > hi {
                        out[i] = hi
                    } else {
                        out[i] = v
                    }
                }
            }
        "#,
            r#"
            #include <stdio.h>
            extern void clamp(int*, int*, int, int, int);
            int main() {
                int data[5] = {-5, 3, 10, 7, 20};
                int out[5];
                clamp(data, out, 0, 10, 5);
                printf("%d %d %d %d %d\n", out[0], out[1], out[2], out[3], out[4]);
                return 0;
            }
        "#,
            "0 3 10 7 10",
        );
    }

    // === Kernel calling a helper func ===

    #[test]
    fn test_kernel_calls_helper_func() {
        assert_c_interop(
            r#"
            func square(x: i32) -> i32 { return x * x }

            export kernel apply_square(data: *i32, out: *mut i32)
                over i in n step 1
            {
                out[i] = square(data[i])
            }
        "#,
            r#"
            #include <stdio.h>
            extern void apply_square(int*, int*, int);
            int main() {
                int data[4] = {1, 2, 3, 4};
                int out[4];
                apply_square(data, out, 4);
                printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
                return 0;
            }
        "#,
            "1 4 9 16",
        );
    }

    // === Kernel with const used in body ===

    #[test]
    fn test_kernel_uses_const() {
        assert_c_interop(
            r#"
            const OFFSET: i32 = 100

            export kernel add_offset(data: *i32, out: *mut i32)
                over i in n step 1
            {
                out[i] = data[i] + OFFSET
            }
        "#,
            r#"
            #include <stdio.h>
            extern void add_offset(int*, int*, int);
            int main() {
                int data[3] = {1, 2, 3};
                int out[3];
                add_offset(data, out, 3);
                printf("%d %d %d\n", out[0], out[1], out[2]);
                return 0;
            }
        "#,
            "101 102 103",
        );
    }

    // === Zero-length input ===

    #[test]
    fn test_kernel_zero_length() {
        assert_c_interop(
            r#"
            export kernel noop(data: *i32, out: *mut i32)
                over i in n step 1
            {
                out[i] = data[i]
            }
        "#,
            r#"
            #include <stdio.h>
            extern void noop(int*, int*, int);
            int main() {
                int data[1] = {42};
                int out[1] = {0};
                noop(data, out, 0);
                printf("%d\n", out[0]);
                return 0;
            }
        "#,
            "0",
        );
    }
}
