#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === foreach basic ===

    #[test]
    fn test_foreach_array_add() {
        let ea_source = r#"
            export func add_arrays(a: *f32, b: *f32, out: *mut f32, n: i32) {
                foreach (i in 0..n) {
                    out[i] = a[i] + b[i]
                }
            }
        "#;
        let c_source = r#"
            #include <stdio.h>
            extern void add_arrays(const float*, const float*, float*, int);
            int main() {
                float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
                float b[] = {10.0f, 20.0f, 30.0f, 40.0f};
                float out[4];
                add_arrays(a, b, out, 4);
                printf("%.0f %.0f %.0f %.0f\n", out[0], out[1], out[2], out[3]);
                return 0;
            }
        "#;
        assert_c_interop(ea_source, c_source, "11 22 33 44");
    }

    #[test]
    fn test_foreach_scalar_scale() {
        let ea_source = r#"
            export func scale(data: *f32, out: *mut f32, n: i32, factor: f32) {
                foreach (i in 0..n) {
                    out[i] = data[i] * factor
                }
            }
        "#;
        let c_source = r#"
            #include <stdio.h>
            extern void scale(const float*, float*, int, float);
            int main() {
                float data[] = {1.0f, 2.0f, 3.0f};
                float out[3];
                scale(data, out, 3, 2.5f);
                printf("%.1f %.1f %.1f\n", out[0], out[1], out[2]);
                return 0;
            }
        "#;
        assert_c_interop(ea_source, c_source, "2.5 5.0 7.5");
    }

    #[test]
    fn test_foreach_with_start_offset() {
        let ea_source = r#"
            export func fill_range(out: *mut i32, start: i32, end: i32) {
                foreach (i in start..end) {
                    out[i] = i * i
                }
            }
        "#;
        let c_source = r#"
            #include <stdio.h>
            extern void fill_range(int*, int, int);
            int main() {
                int out[10] = {0};
                fill_range(out, 2, 5);
                printf("%d %d %d %d %d\n", out[0], out[1], out[2], out[3], out[4]);
                return 0;
            }
        "#;
        assert_c_interop(ea_source, c_source, "0 0 4 9 16");
    }

    #[test]
    fn test_foreach_empty_range() {
        let ea_source = r#"
            export func noop(out: *mut i32, n: i32) {
                foreach (i in n..n) {
                    out[i] = 999
                }
            }
        "#;
        let c_source = r#"
            #include <stdio.h>
            extern void noop(int*, int);
            int main() {
                int out[4] = {1, 2, 3, 4};
                noop(out, 2);
                printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
                return 0;
            }
        "#;
        assert_c_interop(ea_source, c_source, "1 2 3 4");
    }

    #[test]
    fn test_foreach_dot_product() {
        let ea_source = r#"
            export func dot(a: *f32, b: *f32, n: i32) -> f32 {
                let mut sum: f32 = 0.0
                foreach (i in 0..n) {
                    sum = sum + a[i] * b[i]
                }
                return sum
            }
        "#;
        let c_source = r#"
            #include <stdio.h>
            extern float dot(const float*, const float*, int);
            int main() {
                float a[] = {1,2,3,4,5,6,7,8,9,10};
                float b[] = {10,9,8,7,6,5,4,3,2,1};
                printf("%.0f\n", dot(a, b, 10));
                return 0;
            }
        "#;
        assert_c_interop(ea_source, c_source, "220");
    }

    #[test]
    fn test_foreach_standalone() {
        let source = r#"
            func main() {
                let mut sum: i32 = 0
                foreach (i in 0..10) {
                    sum = sum + i
                }
                println(sum)
            }
        "#;
        assert_output(source, "45");
    }

    // === foreach + unroll composition ===

    #[test]
    fn test_unroll_foreach_composition() {
        let ea_source = r#"
            export func add_unrolled(a: *f32, b: *f32, out: *mut f32, n: i32) {
                unroll(4) foreach (i in 0..n) {
                    out[i] = a[i] + b[i]
                }
            }
        "#;
        let c_source = r#"
            #include <stdio.h>
            extern void add_unrolled(const float*, const float*, float*, int);
            int main() {
                float a[] = {1,2,3,4,5};
                float b[] = {10,20,30,40,50};
                float out[5];
                add_unrolled(a, b, out, 5);
                printf("%.0f %.0f %.0f %.0f %.0f\n", out[0], out[1], out[2], out[3], out[4]);
                return 0;
            }
        "#;
        assert_c_interop(ea_source, c_source, "11 22 33 44 55");
    }
}
