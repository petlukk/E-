#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === Phase B: i16x8, i16x16, maddubs ===

    #[test]
    fn test_i16x8_literal_element() {
        assert_output(
            r#"
            func main() {
                let v: i16x8 = [100, 200, 300, 400, 500, 600, 700, 800]i16x8
                let x: i16 = v[0]
                println(x)
                let y: i16 = v[7]
                println(y)
            }
            "#,
            "100\n800",
        );
    }

    #[test]
    fn test_i16x8_add_dot() {
        assert_output(
            r#"
            func main() {
                let a: i16x8 = [1, 2, 3, 4, 5, 6, 7, 8]i16x8
                let b: i16x8 = [10, 20, 30, 40, 50, 60, 70, 80]i16x8
                let c: i16x8 = a .+ b
                let x: i16 = c[0]
                println(x)
                let y: i16 = c[7]
                println(y)
            }
            "#,
            "11\n88",
        );
    }

    #[test]
    fn test_i16x8_splat() {
        assert_output(
            r#"
            func main() {
                let s: i16 = 1000
                let v: i16x8 = splat(s)
                let x: i16 = v[0]
                println(x)
                let y: i16 = v[7]
                println(y)
            }
            "#,
            "1000\n1000",
        );
    }

    #[test]
    fn test_i16x8_reduce_add() {
        assert_output(
            r#"
            func main() {
                let v: i16x8 = [1, 2, 3, 4, 5, 6, 7, 8]i16x8
                let s: i16 = reduce_add(v)
                println(s)
            }
            "#,
            "36",
        );
    }

    #[test]
    fn test_i16x8_reduce_max() {
        assert_output(
            r#"
            func main() {
                let v: i16x8 = [10, 200, -30, 150, 5, 1800, 20, 100]i16x8
                let m: i16 = reduce_max(v)
                println(m)
            }
            "#,
            "1800",
        );
    }

    #[test]
    fn test_i16x8_reduce_min() {
        assert_output(
            r#"
            func main() {
                let v: i16x8 = [10, 200, -30, 150, 5, 1800, 20, 100]i16x8
                let m: i16 = reduce_min(v)
                println(m)
            }
            "#,
            "-30",
        );
    }

    #[test]
    fn test_i16x16_splat_and_element() {
        assert_output(
            r#"
            func main() {
                let s: i16 = 42
                let v: i16x16 = splat(s)
                let x: i16 = v[0]
                println(x)
                let y: i16 = v[15]
                println(y)
            }
            "#,
            "42\n42",
        );
    }

    #[test]
    fn test_i16x16_literal_add() {
        assert_output(
            r#"
            func main() {
                let a: i16x16 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]i16x16
                let b: i16x16 = splat(100)
                let c: i16x16 = a .+ b
                let x: i16 = c[0]
                println(x)
                let y: i16 = c[15]
                println(y)
            }
            "#,
            "101\n116",
        );
    }

    // === maddubs: u8x16 × i8x16 → i16x8 ===

    #[test]
    fn test_maddubs_basic() {
        // maddubs([2,3,...], [4,5,...]):
        // pair (2*4)+(3*5) = 8+15 = 23, (2*4)+(3*5)=23, etc.
        // [2,3, 2,3, 2,3, 2,3, 2,3, 2,3, 2,3, 2,3] u8x16
        // [4,5, 4,5, 4,5, 4,5, 4,5, 4,5, 4,5, 4,5] i8x16
        // each i16 lane = 2*4 + 3*5 = 23
        assert_output(
            r#"
            func main() {
                let a: u8x16 = [2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3]u8x16
                let b: i8x16 = [4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5]i8x16
                let c: i16x8 = maddubs(a, b)
                let x: i16 = c[0]
                println(x)
                let y: i16 = c[7]
                println(y)
            }
            "#,
            "23\n23",
        );
    }

    #[test]
    fn test_maddubs_dot_product() {
        // Dot product of 16 u8 values × 16 i8 weights, accumulate as i32.
        // [1,1,...] × [1,1,...] = 8 pairs each = 2, reduce_add(i16x8) = 16
        assert_output(
            r#"
            func main() {
                let a: u8x16 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]u8x16
                let b: i8x16 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]i8x16
                let c: i16x8 = maddubs(a, b)
                let s: i16 = reduce_add(c)
                println(s)
            }
            "#,
            "16",
        );
    }

    #[test]
    fn test_maddubs_c_interop() {
        assert_c_interop(
            r#"
            export func dot_u8i8_16(act: *u8, wt: *i8, n: i32) -> i16 {
                let mut acc: i16x8 = splat(0)
                let mut i: i32 = 0
                while i < n {
                    let a: u8x16 = load(act, i)
                    let b: i8x16 = load(wt, i)
                    acc = acc .+ maddubs(a, b)
                    i = i + 16
                }
                return reduce_add(acc)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern int16_t dot_u8i8_16(const uint8_t *act, const int8_t *wt, int n);
            int main() {
                uint8_t act[32];
                int8_t  wt[32];
                for (int i = 0; i < 32; i++) { act[i] = 1; wt[i] = 2; }
                int result = (int)dot_u8i8_16(act, wt, 32);
                printf("%d\n", result);
                return 0;
            }
            "#,
            "64",
        );
    }

    // === AVX-512: f32x16 ===

    /// Verify f32x16 type-checks and emits <16 x float> LLVM IR.
    /// Marked ignore because the resulting binary requires AVX-512 hardware to run,
    /// but IR generation works on any x86-64 host.
    #[test]
    fn test_f32x16_ir_contains_16_float_vector() {
        use ea_compiler::{CompileOptions, OutputMode};

        let source = r#"
            export func scale_f32x16(src: *restrict f32, dst: *mut f32, n: i32, scale: f32) {
                let vscale: f32x16 = splat(scale)
                let mut i: i32 = 0
                while i + 16 <= n {
                    let v: f32x16 = load(src, i)
                    store(dst, i, v .* vscale)
                    i = i + 16
                }
            }
        "#;

        let dir = tempfile::TempDir::new().unwrap();
        let ir_path = dir.path().join("scale.ll");
        let opts = CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: "+avx512f".to_string(),
        };
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts)
            .expect("f32x16 compilation failed");

        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("<16 x float>"),
            "expected <16 x float> in IR, got:\n{ir}"
        );
    }

    #[test]
    fn test_f32x16_splat_typecheck() {
        // Verify f32x16 splat + element access type-checks (no AVX-512 needed for IR check)
        use ea_compiler::{CompileOptions, OutputMode};

        let source = r#"
            export func sum_f32x16(v: f32x16) -> f32 {
                return reduce_add(v)
            }
        "#;
        let dir = tempfile::TempDir::new().unwrap();
        let ir_path = dir.path().join("sum.ll");
        let opts = CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: "+avx512f".to_string(),
        };
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts)
            .expect("f32x16 reduce_add compilation failed");

        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(ir.contains("<16 x float>"), "expected <16 x float> in IR");
        assert!(ir.contains("reduce.fadd"), "expected reduce.fadd in IR");
    }

    #[test]
    fn test_i16x8_load_store_c_interop() {
        assert_c_interop(
            r#"
            export func add_constant_i16x8(src: *i16, dst: *mut i16, n: i32, val: i16) {
                let mut i: i32 = 0
                let splat_val: i16x8 = splat(val)
                while i < n {
                    let chunk: i16x8 = load(src, i)
                    let result: i16x8 = chunk .+ splat_val
                    store(dst, i, result)
                    i = i + 8
                }
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void add_constant_i16x8(const int16_t *src, int16_t *dst, int n, int16_t val);
            int main() {
                int16_t src[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
                int16_t dst[16] = {0};
                add_constant_i16x8(src, dst, 16, 100);
                printf("%d %d %d %d\n", (int)dst[0], (int)dst[7], (int)dst[8], (int)dst[15]);
                return 0;
            }
            "#,
            "101 108 109 116",
        );
    }
}
