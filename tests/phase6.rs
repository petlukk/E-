#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    #[test]
    fn test_ir_inspection_works() {
        let ir = compile_to_ir(
            r#"
            func main() {
                let v: f32x4 = splat(1.0)
                println(v[0])
            }
        "#,
        );
        assert!(
            ir.contains("<4 x float>"),
            "IR should contain f32x4 vector type"
        );
    }

    #[test]
    fn test_fma_f32x4_correctness() {
        assert_output(
            r#"
            func main() {
                let a: f32x4 = [1.0, 2.0, 3.0, 4.0]f32x4
                let b: f32x4 = [10.0, 10.0, 10.0, 10.0]f32x4
                let c: f32x4 = [100.0, 100.0, 100.0, 100.0]f32x4
                let r: f32x4 = fma(a, b, c)
                println(r[0])
                println(r[3])
            }
        "#,
            "110\n140",
        );
    }

    #[test]
    fn test_fma_ir_uses_llvm_intrinsic() {
        let ir = compile_to_ir(
            r#"
            func main() {
                let a: f32x4 = splat(1.0)
                let b: f32x4 = splat(2.0)
                let c: f32x4 = splat(3.0)
                let r: f32x4 = fma(a, b, c)
                println(r[0])
            }
        "#,
        );
        assert!(
            ir.contains("llvm.fma.v4f32"),
            "IR must contain llvm.fma.v4f32 intrinsic call"
        );
    }

    #[test]
    fn test_reduce_add_f32x4() {
        assert_output(
            r#"
            func main() {
                let v: f32x4 = [1.0, 2.0, 3.0, 4.0]f32x4
                let sum: f32 = reduce_add(v)
                println(sum)
            }
        "#,
            "10",
        );
    }

    #[test]
    fn test_reduce_add_ir() {
        let ir = compile_to_ir(
            r#"
            func main() {
                let v: f32x4 = [1.0, 2.0, 3.0, 4.0]f32x4
                let sum: f32 = reduce_add(v)
                println(sum)
            }
        "#,
        );
        assert!(
            ir.contains("llvm.vector.reduce.fadd"),
            "IR must use vector reduce intrinsic, not scalar loop"
        );
    }

    #[test]
    fn test_reduce_max_f32x4() {
        assert_output(
            r#"
            func main() {
                let v: f32x4 = [3.0, 7.0, 1.0, 5.0]f32x4
                let m: f32 = reduce_max(v)
                println(m)
            }
        "#,
            "7",
        );
    }

    #[test]
    fn test_reduce_min_f32x4() {
        assert_output(
            r#"
            func main() {
                let v: f32x4 = [3.0, 7.0, 1.0, 5.0]f32x4
                let m: f32 = reduce_min(v)
                println(m)
            }
        "#,
            "1",
        );
    }

    #[test]
    fn test_reduce_add_i32x4() {
        assert_output(
            r#"
            func main() {
                let v: i32x4 = [10, 20, 30, 40]i32x4
                let sum: i32 = reduce_add(v)
                println(sum)
            }
        "#,
            "100",
        );
    }

    #[test]
    fn test_shuffle_reverse() {
        assert_output(
            r#"
            func main() {
                let v: f32x4 = [1.0, 2.0, 3.0, 4.0]f32x4
                let r: f32x4 = shuffle(v, [3, 2, 1, 0])
                println(r[0])
                println(r[3])
            }
        "#,
            "4\n1",
        );
    }

    #[test]
    fn test_shuffle_ir() {
        let ir = compile_to_ir(
            r#"
            func main() {
                let v: f32x4 = [1.0, 2.0, 3.0, 4.0]f32x4
                let r: f32x4 = shuffle(v, [3, 2, 1, 0])
                println(r[0])
            }
        "#,
        );
        assert!(
            ir.contains("shufflevector"),
            "IR must contain shufflevector"
        );
    }

    #[test]
    fn test_shuffle_broadcast_lane() {
        assert_output(
            r#"
            func main() {
                let v: f32x4 = [10.0, 20.0, 30.0, 40.0]f32x4
                let r: f32x4 = shuffle(v, [2, 2, 2, 2])
                println(r[0])
                println(r[3])
            }
        "#,
            "30\n30",
        );
    }

    #[test]
    fn test_select_max() {
        assert_output(
            r#"
            func main() {
                let a: f32x4 = [1.0, 5.0, 3.0, 8.0]f32x4
                let b: f32x4 = [4.0, 2.0, 6.0, 7.0]f32x4
                let result: f32x4 = select(a .> b, a, b)
                println(result[0])
                println(result[1])
                println(result[2])
                println(result[3])
            }
        "#,
            "4\n5\n6\n8",
        );
    }

    #[test]
    fn test_select_ir() {
        let ir = compile_to_ir(
            r#"
            func main() {
                let a: f32x4 = [1.0, 5.0, 3.0, 8.0]f32x4
                let b: f32x4 = [4.0, 2.0, 6.0, 7.0]f32x4
                let result: f32x4 = select(a .> b, a, b)
                println(result[0])
            }
        "#,
        );
        assert!(
            ir.contains("fcmp"),
            "IR must contain fcmp for vector comparison"
        );
        assert!(ir.contains("select"), "IR must contain select instruction");
    }

    // --- 256-bit vectors ---

    #[test]
    fn test_f32x8_literal() {
        assert_output(
            r#"
            func main() {
                let v: f32x8 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]f32x8
                println(v[0])
                println(v[7])
            }
        "#,
            "1\n8",
        );
    }

    #[test]
    fn test_f32x8_splat_and_add() {
        assert_output(
            r#"
            func main() {
                let a: f32x8 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]f32x8
                let b: f32x8 = splat(10.0)
                let c: f32x8 = a .+ b
                println(c[0])
                println(c[7])
            }
        "#,
            "11\n18",
        );
    }

    #[test]
    fn test_f32x8_reduce_add() {
        assert_output(
            r#"
            func main() {
                let v: f32x8 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]f32x8
                let sum: f32 = reduce_add(v)
                println(sum)
            }
        "#,
            "36",
        );
    }

    #[test]
    fn test_i32x8_literal() {
        assert_output(
            r#"
            func main() {
                let v: i32x8 = [10, 20, 30, 40, 50, 60, 70, 80]i32x8
                println(v[0])
                println(v[7])
            }
        "#,
            "10\n80",
        );
    }

    #[test]
    fn test_f32x8_c_interop() {
        assert_c_interop(
            r#"
            export func add_f32x8(a: *f32, b: *f32, out: *mut f32, len: i32) {
                let mut i: i32 = 0
                while i + 8 <= len {
                    let va: f32x8 = load(a, i)
                    let vb: f32x8 = load(b, i)
                    let result: f32x8 = va .+ vb
                    store(out, i, result)
                    i = i + 8
                }
            }
        "#,
            r#"#include <stdio.h>
            extern void add_f32x8(const float*, const float*, float*, int);
            int main() {
                float a[] = {1,2,3,4,5,6,7,8};
                float b[] = {10,20,30,40,50,60,70,80};
                float out[8];
                add_f32x8(a, b, out, 8);
                printf("%g %g %g %g %g %g %g %g\n",
                    out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
                return 0;
            }"#,
            "11 22 33 44 55 66 77 88",
        );
    }

    // --- C Interop Integration (spec example kernels) ---

    #[test]
    fn test_fma_kernel_c_interop() {
        assert_c_interop(
            r#"
            export func fma_kernel(a: *f32, b: *f32, c: *f32, out: *mut f32, len: i32) {
                let mut i: i32 = 0
                while i + 4 <= len {
                    let va: f32x4 = load(a, i)
                    let vb: f32x4 = load(b, i)
                    let vc: f32x4 = load(c, i)
                    let result: f32x4 = fma(va, vb, vc)
                    store(out, i, result)
                    i = i + 4
                }
            }
        "#,
            r#"#include <stdio.h>
            extern void fma_kernel(const float*, const float*, const float*, float*, int);
            int main() {
                float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
                float b[] = {10.0f, 10.0f, 10.0f, 10.0f};
                float c[] = {100.0f, 100.0f, 100.0f, 100.0f};
                float out[4];
                fma_kernel(a, b, c, out, 4);
                printf("%g %g %g %g\n", out[0], out[1], out[2], out[3]);
                return 0;
            }"#,
            "110 120 130 140",
        );
    }

    #[test]
    fn test_horizontal_sum_c_interop() {
        assert_c_interop(
            r#"
            export func horizontal_sum(data: *f32, len: i32) -> f32 {
                let mut acc: f32x4 = splat(0.0)
                let mut i: i32 = 0
                while i + 4 <= len {
                    let v: f32x4 = load(data, i)
                    acc = acc .+ v
                    i = i + 4
                }
                return reduce_add(acc)
            }
        "#,
            r#"#include <stdio.h>
            extern float horizontal_sum(const float*, int);
            int main() {
                float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
                printf("%g\n", horizontal_sum(data, 8));
                return 0;
            }"#,
            "36",
        );
    }

    #[test]
    fn test_clamp_kernel_c_interop() {
        assert_c_interop(
            r#"
            export func clamp(data: *mut f32, len: i32, lo: f32, hi: f32) {
                let vlo: f32x4 = splat(lo)
                let vhi: f32x4 = splat(hi)
                let mut i: i32 = 0
                while i + 4 <= len {
                    let v: f32x4 = load(data, i)
                    let clamped_lo: f32x4 = select(v .< vlo, vlo, v)
                    let clamped: f32x4 = select(clamped_lo .> vhi, vhi, clamped_lo)
                    store(data, i, clamped)
                    i = i + 4
                }
            }
        "#,
            r#"#include <stdio.h>
            extern void clamp(float*, int, float, float);
            int main() {
                float data[] = {-5.0f, 3.0f, 15.0f, 7.0f};
                clamp(data, 4, 0.0f, 10.0f);
                printf("%g %g %g %g\n", data[0], data[1], data[2], data[3]);
                return 0;
            }"#,
            "0 3 10 7",
        );
    }
}
