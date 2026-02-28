#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === Basic kernel: same output as hand-written func ===

    #[test]
    fn test_kernel_basic_scale() {
        assert_c_interop(
            r#"
            export kernel scale(data: *f32, out: *mut f32, factor: f32)
                over i in len step 1
            {
                out[i] = data[i] * factor
            }
        "#,
            r#"
            #include <stdio.h>
            extern void scale(float*, float*, float, int);
            int main() {
                float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
                float out[4];
                scale(in, out, 3.0f, 4);
                printf("%.0f %.0f %.0f %.0f\n", out[0], out[1], out[2], out[3]);
                return 0;
            }
        "#,
            "3 6 9 12",
        );
    }

    #[test]
    fn test_kernel_matches_func_output() {
        // Kernel version
        let kernel_result = compile_and_link_with_c(
            r#"
            export kernel double_it(data: *i32, out: *mut i32)
                over i in n step 1
            {
                out[i] = data[i] * 2
            }
        "#,
            r#"
            #include <stdio.h>
            extern void double_it(int*, int*, int);
            int main() {
                int in[5] = {10, 20, 30, 40, 50};
                int out[5];
                double_it(in, out, 5);
                for (int i = 0; i < 5; i++) printf("%d ", out[i]);
                printf("\n");
                return 0;
            }
        "#,
        );

        // Func version (identical behavior)
        let func_result = compile_and_link_with_c(
            r#"
            export func double_it(data: *i32, out: *mut i32, n: i32) {
                let mut i: i32 = 0
                while i + 1 <= n {
                    out[i] = data[i] * 2
                    i = i + 1
                }
            }
        "#,
            r#"
            #include <stdio.h>
            extern void double_it(int*, int*, int);
            int main() {
                int in[5] = {10, 20, 30, 40, 50};
                int out[5];
                double_it(in, out, 5);
                for (int i = 0; i < 5; i++) printf("%d ", out[i]);
                printf("\n");
                return 0;
            }
        "#,
        );

        assert_eq!(kernel_result.stdout.trim(), func_result.stdout.trim());
        assert_eq!(kernel_result.stdout.trim(), "20 40 60 80 100");
    }

    // === SIMD kernel ===

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_kernel_simd_scale() {
        assert_c_interop(
            r#"
            export kernel vscale(data: *f32, out: *mut f32, factor: f32)
                over i in len step 4
            {
                let vf: f32x4 = splat(factor)
                store(out, i, load(data, i) .* vf)
            }
        "#,
            r#"
            #include <stdio.h>
            extern void vscale(float*, float*, float, int);
            int main() {
                float in[8] = {1, 2, 3, 4, 5, 6, 7, 8};
                float out[8];
                vscale(in, out, 2.0f, 8);
                printf("%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
                    out[0], out[1], out[2], out[3],
                    out[4], out[5], out[6], out[7]);
                return 0;
            }
        "#,
            "2 4 6 8 10 12 14 16",
        );
    }

    // === Range bound becomes last parameter ===

    #[test]
    fn test_kernel_range_bound_is_last_param() {
        // The `n` range bound should appear as the last C parameter
        assert_c_interop(
            r#"
            export kernel add_arrays(a: *i32, b: *i32, out: *mut i32)
                over idx in n step 1
            {
                out[idx] = a[idx] + b[idx]
            }
        "#,
            r#"
            #include <stdio.h>
            extern void add_arrays(int*, int*, int*, int);
            int main() {
                int a[3] = {1, 2, 3};
                int b[3] = {10, 20, 30};
                int out[3];
                add_arrays(a, b, out, 3);
                printf("%d %d %d\n", out[0], out[1], out[2]);
                return 0;
            }
        "#,
            "11 22 33",
        );
    }

    // === Range bound readable inside body ===

    #[test]
    fn test_kernel_range_bound_accessible_in_body() {
        assert_c_interop(
            r#"
            export kernel fill_index(out: *mut i32)
                over i in n step 1
            {
                out[i] = n - i
            }
        "#,
            r#"
            #include <stdio.h>
            extern void fill_index(int*, int);
            int main() {
                int out[4];
                fill_index(out, 4);
                printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
                return 0;
            }
        "#,
            "4 3 2 1",
        );
    }

    // === Non-export kernel ===

    #[test]
    fn test_kernel_non_export_as_helper() {
        assert_c_interop(
            r#"
            kernel fill_internal(out: *mut i32, val: i32)
                over i in n step 1
            {
                out[i] = val
            }

            export func do_fill(out: *mut i32, val: i32, n: i32) {
                fill_internal(out, val, n)
            }
        "#,
            r#"
            #include <stdio.h>
            extern void do_fill(int*, int, int);
            int main() {
                int out[3];
                do_fill(out, 42, 3);
                printf("%d %d %d\n", out[0], out[1], out[2]);
                return 0;
            }
        "#,
            "42 42 42",
        );
    }

    // === Step > 1 without SIMD (scalar loop with stride) ===

    #[test]
    fn test_kernel_step_greater_than_one() {
        assert_c_interop(
            r#"
            export kernel stride2(data: *i32, out: *mut i32)
                over i in n step 2
            {
                out[i] = data[i] + data[i + 1]
                out[i + 1] = data[i] - data[i + 1]
            }
        "#,
            r#"
            #include <stdio.h>
            extern void stride2(int*, int*, int);
            int main() {
                int data[4] = {10, 3, 20, 7};
                int out[4];
                stride2(data, out, 4);
                printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
                return 0;
            }
        "#,
            "13 7 27 13",
        );
    }

    // === Kernel with no remainder: len not a multiple of step ===

    #[test]
    fn test_kernel_no_tail_skips_remainder() {
        assert_c_interop(
            r#"
            export kernel fill(out: *mut i32)
                over i in n step 3
            {
                out[i] = 1
                out[i + 1] = 1
                out[i + 2] = 1
            }
        "#,
            r#"
            #include <stdio.h>
            extern void fill(int*, int);
            int main() {
                int out[5] = {0, 0, 0, 0, 0};
                fill(out, 5);
                // Only first 3 elements processed (step 3, 3+3=6 > 5)
                printf("%d %d %d %d %d\n", out[0], out[1], out[2], out[3], out[4]);
                return 0;
            }
        "#,
            "1 1 1 0 0",
        );
    }

    // === Metadata: kernel export appears in JSON ===

    #[test]
    fn test_kernel_metadata_json() {
        let source = r#"
            export kernel transform(data: *f32, out: *mut f32)
                over i in len step 1
            {
                out[i] = data[i] * 2.0
            }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts).unwrap();
        let json = ea_compiler::metadata::generate_json(&stmts, "test_lib");
        assert!(
            json.contains("\"name\": \"transform\""),
            "expected kernel name in metadata, got:\n{json}"
        );
        assert!(
            json.contains("\"name\": \"len\""),
            "expected range bound 'len' in metadata args, got:\n{json}"
        );
        assert!(
            json.contains("\"type\": \"i32\""),
            "expected len param typed as i32, got:\n{json}"
        );
    }

    // === Header: kernel export appears in C header ===

    #[test]
    fn test_kernel_header_generation() {
        let source = r#"
            export kernel process(input: *f32, output: *mut f32)
                over i in n step 4
            {
                output[i] = input[i]
            }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts).unwrap();
        let header = ea_compiler::header::generate(&stmts, "test_mod");
        assert!(
            header.contains("process("),
            "expected kernel in header, got:\n{header}"
        );
        assert!(
            header.contains("int32_t n"),
            "expected range bound in header, got:\n{header}"
        );
    }

    // === exported_function_names includes kernel ===

    #[test]
    fn test_kernel_in_exported_function_names() {
        let source = r#"
            export kernel my_kernel(data: *i32, out: *mut i32)
                over i in n step 1
            { out[i] = data[i] }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts).unwrap();
        let exports = ea_compiler::ast::exported_function_names(&stmts);
        assert_eq!(exports, vec!["my_kernel"]);
    }

    // === IR verification: kernel produces a while loop ===

    #[test]
    fn test_kernel_ir_has_loop() {
        let ir = compile_to_ir(
            r#"
            export kernel inc(data: *i32, out: *mut i32)
                over i in n step 1
            {
                out[i] = data[i] + 1
            }
        "#,
        );
        assert!(
            ir.contains("while_cond") || ir.contains("br "),
            "expected loop in IR, got:\n{ir}"
        );
    }

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
