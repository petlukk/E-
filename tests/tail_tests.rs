#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === tail scalar: processes remainder elements ===

    #[test]
    fn test_tail_scalar_basic() {
        assert_c_interop(
            r#"
            export kernel fill_squared(data: *i32, out: *mut i32)
                over i in n step 4
                tail scalar {
                    out[i] = data[i] * data[i]
                }
            {
                out[i] = data[i] * data[i]
                out[i + 1] = data[i + 1] * data[i + 1]
                out[i + 2] = data[i + 2] * data[i + 2]
                out[i + 3] = data[i + 3] * data[i + 3]
            }
        "#,
            r#"
            #include <stdio.h>
            extern void fill_squared(int*, int*, int);
            int main() {
                int data[7] = {1, 2, 3, 4, 5, 6, 7};
                int out[7] = {0};
                fill_squared(data, out, 7);
                for (int i = 0; i < 7; i++) printf("%d ", out[i]);
                printf("\n");
                return 0;
            }
        "#,
            "1 4 9 16 25 36 49",
        );
    }

    #[test]
    fn test_tail_scalar_exact_multiple() {
        // When len is exact multiple of step, tail should not execute
        assert_c_interop(
            r#"
            export kernel copy(data: *i32, out: *mut i32)
                over i in n step 2
                tail scalar {
                    out[i] = data[i] + 1000
                }
            {
                out[i] = data[i]
                out[i + 1] = data[i + 1]
            }
        "#,
            r#"
            #include <stdio.h>
            extern void copy(int*, int*, int);
            int main() {
                int data[4] = {10, 20, 30, 40};
                int out[4] = {0};
                copy(data, out, 4);
                for (int i = 0; i < 4; i++) printf("%d ", out[i]);
                printf("\n");
                return 0;
            }
        "#,
            "10 20 30 40",
        );
    }

    #[test]
    fn test_tail_scalar_all_remainder() {
        // When len < step, only tail executes
        assert_c_interop(
            r#"
            export kernel inc(data: *i32, out: *mut i32)
                over i in n step 4
                tail scalar {
                    out[i] = data[i] + 1
                }
            {
                out[i] = data[i] + 100
                out[i + 1] = data[i + 1] + 100
                out[i + 2] = data[i + 2] + 100
                out[i + 3] = data[i + 3] + 100
            }
        "#,
            r#"
            #include <stdio.h>
            extern void inc(int*, int*, int);
            int main() {
                int data[3] = {1, 2, 3};
                int out[3] = {0};
                inc(data, out, 3);
                printf("%d %d %d\n", out[0], out[1], out[2]);
                return 0;
            }
        "#,
            "2 3 4",
        );
    }

    #[test]
    fn test_tail_scalar_zero_length() {
        assert_c_interop(
            r#"
            export kernel noop(data: *i32, out: *mut i32)
                over i in n step 2
                tail scalar {
                    out[i] = data[i]
                }
            {
                out[i] = data[i]
                out[i + 1] = data[i + 1]
            }
        "#,
            r#"
            #include <stdio.h>
            extern void noop(int*, int*, int);
            int main() {
                int out[1] = {99};
                noop((int*)0, out, 0);
                printf("%d\n", out[0]);
                return 0;
            }
        "#,
            "99",
        );
    }

    #[test]
    fn test_tail_scalar_uses_params() {
        assert_c_interop(
            r#"
            export kernel scale(data: *f32, out: *mut f32, factor: f32)
                over i in n step 2
                tail scalar {
                    out[i] = data[i] * factor
                }
            {
                out[i] = data[i] * factor
                out[i + 1] = data[i + 1] * factor
            }
        "#,
            r#"
            #include <stdio.h>
            extern void scale(float*, float*, float, int);
            int main() {
                float data[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
                float out[5];
                scale(data, out, 10.0f, 5);
                printf("%.0f %.0f %.0f %.0f %.0f\n",
                    out[0], out[1], out[2], out[3], out[4]);
                return 0;
            }
        "#,
            "10 20 30 40 50",
        );
    }

    // === tail scalar with SIMD main body ===

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tail_scalar_simd_main_body() {
        assert_c_interop(
            r#"
            export kernel vscale(data: *f32, out: *mut f32, factor: f32)
                over i in len step 4
                tail scalar {
                    out[i] = data[i] * factor
                }
            {
                let vf: f32x4 = splat(factor)
                store(out, i, load(data, i) .* vf)
            }
        "#,
            r#"
            #include <stdio.h>
            extern void vscale(float*, float*, float, int);
            int main() {
                float data[7] = {1, 2, 3, 4, 5, 6, 7};
                float out[7];
                vscale(data, out, 2.0f, 7);
                printf("%.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
                    out[0], out[1], out[2], out[3],
                    out[4], out[5], out[6]);
                return 0;
            }
        "#,
            "2 4 6 8 10 12 14",
        );
    }

    // === tail mask: single conditional block ===

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tail_mask_basic() {
        assert_c_interop(
            r#"
            export kernel vscale(data: *f32, out: *mut f32, factor: f32)
                over i in len step 4
                tail mask {
                    let rem: i32 = len - i
                    let vf: f32x4 = splat(factor)
                    let v: f32x4 = load_masked(data, i, rem)
                    store_masked(out, i, v .* vf, rem)
                }
            {
                let vf: f32x4 = splat(factor)
                store(out, i, load(data, i) .* vf)
            }
        "#,
            r#"
            #include <stdio.h>
            extern void vscale(float*, float*, float, int);
            int main() {
                float data[7] = {1, 2, 3, 4, 5, 6, 7};
                float out[7] = {0};
                vscale(data, out, 3.0f, 7);
                printf("%.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
                    out[0], out[1], out[2], out[3],
                    out[4], out[5], out[6]);
                return 0;
            }
        "#,
            "3 6 9 12 15 18 21",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tail_mask_exact_multiple() {
        // Mask body should not execute when len is exact multiple
        assert_c_interop(
            r#"
            export kernel vscale(data: *f32, out: *mut f32, factor: f32)
                over i in len step 4
                tail mask {
                    let rem: i32 = len - i
                    let vf: f32x4 = splat(factor)
                    store_masked(out, i, load_masked(data, i, rem) .* vf, rem)
                }
            {
                let vf: f32x4 = splat(factor)
                store(out, i, load(data, i) .* vf)
            }
        "#,
            r#"
            #include <stdio.h>
            extern void vscale(float*, float*, float, int);
            int main() {
                float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
                float out[8];
                vscale(data, out, 2.0f, 8);
                printf("%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
                    out[0], out[1], out[2], out[3],
                    out[4], out[5], out[6], out[7]);
                return 0;
            }
        "#,
            "2 4 6 8 10 12 14 16",
        );
    }

    // === tail pad: explicit no-tail ===

    #[test]
    fn test_tail_pad_skips_remainder() {
        assert_c_interop(
            r#"
            export kernel fill(out: *mut i32, val: i32)
                over i in n step 3
                tail pad
            {
                out[i] = val
                out[i + 1] = val
                out[i + 2] = val
            }
        "#,
            r#"
            #include <stdio.h>
            extern void fill(int*, int, int);
            int main() {
                int out[5] = {0, 0, 0, 0, 0};
                fill(out, 42, 5);
                printf("%d %d %d %d %d\n",
                    out[0], out[1], out[2], out[3], out[4]);
                return 0;
            }
        "#,
            "42 42 42 0 0",
        );
    }

    // === tail scalar matches hand-written func ===

    #[test]
    fn test_tail_scalar_matches_handwritten_func() {
        let kernel_result = compile_and_link_with_c(
            r#"
            export kernel double_it(data: *i32, out: *mut i32)
                over i in n step 4
                tail scalar {
                    out[i] = data[i] * 2
                }
            {
                out[i] = data[i] * 2
                out[i + 1] = data[i + 1] * 2
                out[i + 2] = data[i + 2] * 2
                out[i + 3] = data[i + 3] * 2
            }
        "#,
            r#"
            #include <stdio.h>
            extern void double_it(int*, int*, int);
            int main() {
                int data[9] = {1,2,3,4,5,6,7,8,9};
                int out[9];
                double_it(data, out, 9);
                for (int i = 0; i < 9; i++) printf("%d ", out[i]);
                printf("\n");
                return 0;
            }
        "#,
        );

        let func_result = compile_and_link_with_c(
            r#"
            export func double_it(data: *i32, out: *mut i32, n: i32) {
                let mut i: i32 = 0
                while i + 4 <= n {
                    out[i] = data[i] * 2
                    out[i + 1] = data[i + 1] * 2
                    out[i + 2] = data[i + 2] * 2
                    out[i + 3] = data[i + 3] * 2
                    i = i + 4
                }
                while i < n {
                    out[i] = data[i] * 2
                    i = i + 1
                }
            }
        "#,
            r#"
            #include <stdio.h>
            extern void double_it(int*, int*, int);
            int main() {
                int data[9] = {1,2,3,4,5,6,7,8,9};
                int out[9];
                double_it(data, out, 9);
                for (int i = 0; i < 9; i++) printf("%d ", out[i]);
                printf("\n");
                return 0;
            }
        "#,
        );

        assert_eq!(kernel_result.stdout.trim(), func_result.stdout.trim());
        assert_eq!(kernel_result.stdout.trim(), "2 4 6 8 10 12 14 16 18");
    }

    // === Error: tail scalar without body ===

    #[test]
    fn test_tail_scalar_missing_body_error() {
        let result = ea_compiler::tokenize(
            "export kernel k(out: *mut i32) over i in n step 2 tail scalar { out[i] = 0 }",
        )
        .and_then(ea_compiler::parse);
        // This should parse fine — the scalar body is `{ out[i] = 0 }`
        // and the main body is missing (should error on expecting '{')
        assert!(
            result.is_err(),
            "expected parse error when main body is missing after tail body"
        );
    }

    // === Error: tail with unknown strategy ===

    #[test]
    fn test_tail_unknown_strategy_error() {
        let result = ea_compiler::tokenize(
            "export kernel k(out: *mut i32) over i in n step 2 tail unknown { out[i] = 0 }",
        )
        .and_then(ea_compiler::parse);
        assert!(
            result.is_err(),
            "expected parse error for unknown tail strategy"
        );
    }

    // === Multiple tail lengths: verify all work ===

    #[test]
    fn test_tail_scalar_all_remainder_lengths() {
        // Test with various lengths to exercise all remainder cases (0..step-1)
        for len in 0..=12 {
            let c_source = format!(
                r#"
                #include <stdio.h>
                extern void add_one(int*, int*, int);
                int main() {{
                    int data[12] = {{1,2,3,4,5,6,7,8,9,10,11,12}};
                    int out[12] = {{0}};
                    add_one(data, out, {len});
                    for (int i = 0; i < {len}; i++) printf("%d ", out[i]);
                    printf("\n");
                    return 0;
                }}
            "#,
            );
            let result = compile_and_link_with_c(
                r#"
                export kernel add_one(data: *i32, out: *mut i32)
                    over i in n step 4
                    tail scalar {
                        out[i] = data[i] + 1
                    }
                {
                    out[i] = data[i] + 1
                    out[i + 1] = data[i + 1] + 1
                    out[i + 2] = data[i + 2] + 1
                    out[i + 3] = data[i + 3] + 1
                }
            "#,
                &c_source,
            );
            let expected: Vec<String> = (1..=len).map(|x| (x + 1).to_string()).collect();
            let expected_str = if expected.is_empty() {
                String::new()
            } else {
                expected.join(" ")
            };
            assert_eq!(result.stdout.trim(), expected_str, "failed for len={len}");
        }
    }
}
