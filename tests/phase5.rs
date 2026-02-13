#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === Phase 5: Explicit SIMD tests ===

    #[test]
    fn test_f32x4_vector_literal() {
        assert_output(
            r#"
            func main() {
                let v: f32x4 = [1.0, 2.0, 3.0, 4.0]f32x4
                println(v[0])
                println(v[3])
            }
        "#,
            "1\n4",
        );
    }

    #[test]
    fn test_i32x4_vector_literal() {
        assert_output(
            r#"
            func main() {
                let v: i32x4 = [10, 20, 30, 40]i32x4
                println(v[0])
                println(v[2])
            }
        "#,
            "10\n30",
        );
    }

    #[test]
    fn test_splat_f32() {
        assert_output(
            r#"
            func main() {
                let v: f32x4 = splat(5.0)
                println(v[0])
                println(v[3])
            }
        "#,
            "5\n5",
        );
    }

    #[test]
    fn test_vector_add_dot() {
        assert_output(
            r#"
            func main() {
                let a: f32x4 = [1.0, 2.0, 3.0, 4.0]f32x4
                let b: f32x4 = [10.0, 20.0, 30.0, 40.0]f32x4
                let c: f32x4 = a .+ b
                println(c[0])
                println(c[3])
            }
        "#,
            "11\n44",
        );
    }

    #[test]
    fn test_vector_sub_dot() {
        assert_output(
            r#"
            func main() {
                let a: f32x4 = [10.0, 20.0, 30.0, 40.0]f32x4
                let b: f32x4 = [1.0, 2.0, 3.0, 4.0]f32x4
                let c: f32x4 = a .- b
                println(c[0])
                println(c[3])
            }
        "#,
            "9\n36",
        );
    }

    #[test]
    fn test_vector_mul_dot() {
        assert_output(
            r#"
            func main() {
                let a: f32x4 = [2.0, 3.0, 4.0, 5.0]f32x4
                let b: f32x4 = [10.0, 10.0, 10.0, 10.0]f32x4
                let c: f32x4 = a .* b
                println(c[0])
                println(c[3])
            }
        "#,
            "20\n50",
        );
    }

    #[test]
    fn test_vector_div_dot() {
        assert_output(
            r#"
            func main() {
                let a: f32x4 = [10.0, 20.0, 30.0, 40.0]f32x4
                let b: f32x4 = splat(10.0)
                let c: f32x4 = a ./ b
                println(c[0])
                println(c[3])
            }
        "#,
            "1\n4",
        );
    }

    #[test]
    fn test_vector_element_access_all() {
        assert_output(
            r#"
            func main() {
                let v: f32x4 = [1.5, 2.5, 3.5, 4.5]f32x4
                let sum: f32 = v[0] + v[1] + v[2] + v[3]
                println(sum)
            }
        "#,
            "12",
        );
    }

    #[test]
    fn test_vector_add_c_interop() {
        assert_c_interop(
            r#"
            export func vector_add(a: *f32, b: *f32, out: *mut f32, len: i32) {
                let mut i: i32 = 0
                while i + 4 <= len {
                    let va: f32x4 = load(a, i)
                    let vb: f32x4 = load(b, i)
                    let result: f32x4 = va .+ vb
                    store(out, i, result)
                    i = i + 4
                }
                while i < len {
                    out[i] = a[i] + b[i]
                    i = i + 1
                }
            }
        "#,
            r#"#include <stdio.h>
            extern void vector_add(const float*, const float*, float*, int);
            int main() {
                float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
                float b[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
                float out[8];
                vector_add(a, b, out, 8);
                printf("%g %g %g %g %g %g %g %g\n",
                    out[0], out[1], out[2], out[3],
                    out[4], out[5], out[6], out[7]);
                return 0;
            }"#,
            "11 22 33 44 55 66 77 88",
        );
    }

    #[test]
    fn test_vector_scale_c_interop() {
        assert_c_interop(
            r#"
            export func scale(data: *mut f32, len: i32, factor: f32) {
                let vfactor: f32x4 = splat(factor)
                let mut i: i32 = 0
                while i + 4 <= len {
                    let v: f32x4 = load(data, i)
                    let scaled: f32x4 = v .* vfactor
                    store(data, i, scaled)
                    i = i + 4
                }
                while i < len {
                    data[i] = data[i] * factor
                    i = i + 1
                }
            }
        "#,
            r#"#include <stdio.h>
            extern void scale(float*, int, float);
            int main() {
                float arr[] = {1.0f, 2.0f, 3.0f, 4.0f};
                scale(arr, 4, 3.0f);
                printf("%g %g %g %g\n", arr[0], arr[1], arr[2], arr[3]);
                return 0;
            }"#,
            "3 6 9 12",
        );
    }

    #[test]
    fn test_i32x4_add_c_interop() {
        assert_c_interop(
            r#"
            export func add_i32_vec(a: *i32, b: *i32, out: *mut i32, len: i32) {
                let mut i: i32 = 0
                while i + 4 <= len {
                    let va: i32x4 = load(a, i)
                    let vb: i32x4 = load(b, i)
                    let result: i32x4 = va .+ vb
                    store(out, i, result)
                    i = i + 4
                }
            }
        "#,
            r#"#include <stdio.h>
            extern void add_i32_vec(const int*, const int*, int*, int);
            int main() {
                int a[] = {1, 2, 3, 4};
                int b[] = {100, 200, 300, 400};
                int out[4];
                add_i32_vec(a, b, out, 4);
                printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
                return 0;
            }"#,
            "101 202 303 404",
        );
    }

    #[test]
    fn test_vector_dot_product_c_interop() {
        assert_c_interop(
            r#"
            export func dot_product_simd(a: *f32, b: *f32, len: i32) -> f32 {
                let mut acc: f32x4 = splat(0.0)
                let mut i: i32 = 0
                while i + 4 <= len {
                    let va: f32x4 = load(a, i)
                    let vb: f32x4 = load(b, i)
                    acc = acc .+ va .* vb
                    i = i + 4
                }
                return acc[0] + acc[1] + acc[2] + acc[3]
            }
        "#,
            r#"#include <stdio.h>
            extern float dot_product_simd(const float*, const float*, int);
            int main() {
                float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
                float b[] = {5.0f, 6.0f, 7.0f, 8.0f};
                printf("%g\n", dot_product_simd(a, b, 4));
                return 0;
            }"#,
            "70",
        );
    }
}
