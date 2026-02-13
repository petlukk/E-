#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === Phase 4: Functions + Pointers tests ===

    #[test]
    fn test_recursive_fibonacci() {
        assert_output_lines(
            r#"
            func fibonacci(n: i32) -> i32 {
                if n <= 1 { return n }
                return fibonacci(n - 1) + fibonacci(n - 2)
            }
            func main() {
                println(fibonacci(0))
                println(fibonacci(1))
                println(fibonacci(5))
                println(fibonacci(10))
            }
        "#,
            &["0", "1", "5", "55"],
        );
    }

    #[test]
    fn test_mutual_function_calls() {
        assert_output(
            r#"
            func double(x: i32) -> i32 { return x * 2 }
            func quad(x: i32) -> i32 { return double(double(x)) }
            func main() { println(quad(3)) }
        "#,
            "12",
        );
    }

    #[test]
    fn test_pointer_read_c_interop() {
        assert_c_interop(
            r#"
            export func sum_array(data: *i32, len: i32) -> i32 {
                let mut total: i32 = 0
                let mut i: i32 = 0
                while i < len {
                    total = total + data[i]
                    i = i + 1
                }
                return total
            }
        "#,
            r#"#include <stdio.h>
            extern int sum_array(const int*, int);
            int main() {
                int arr[] = {10, 20, 30, 40, 50};
                printf("%d\n", sum_array(arr, 5));
                return 0;
            }"#,
            "150",
        );
    }

    #[test]
    fn test_pointer_write_c_interop() {
        assert_c_interop(
            r#"
            export func fill_zeros(out: *mut i32, len: i32) {
                let mut i: i32 = 0
                while i < len {
                    out[i] = 0
                    i = i + 1
                }
            }
        "#,
            r#"#include <stdio.h>
            extern void fill_zeros(int*, int);
            int main() {
                int arr[] = {1, 2, 3, 4};
                fill_zeros(arr, 4);
                printf("%d %d %d %d\n", arr[0], arr[1], arr[2], arr[3]);
                return 0;
            }"#,
            "0 0 0 0",
        );
    }

    #[test]
    fn test_fill_fibonacci_c_interop() {
        assert_c_interop(
            r#"
            func fibonacci(n: i32) -> i32 {
                if n <= 1 { return n }
                return fibonacci(n - 1) + fibonacci(n - 2)
            }
            export func fill_fibonacci(out: *mut i32, count: i32) {
                let mut i: i32 = 0
                while i < count {
                    out[i] = fibonacci(i)
                    i = i + 1
                }
            }
        "#,
            r#"#include <stdio.h>
            extern void fill_fibonacci(int*, int);
            int main() {
                int buf[10];
                fill_fibonacci(buf, 10);
                for (int i = 0; i < 10; i++) printf("%d ", buf[i]);
                printf("\n");
                return 0;
            }"#,
            "0 1 1 2 3 5 8 13 21 34",
        );
    }

    #[test]
    fn test_float_pointer_c_interop() {
        assert_c_interop(
            r#"
            export func scale_array(data: *mut f32, len: i32, factor: f32) {
                let mut i: i32 = 0
                while i < len {
                    data[i] = data[i] * factor
                    i = i + 1
                }
            }
        "#,
            r#"#include <stdio.h>
            extern void scale_array(float*, int, float);
            int main() {
                float arr[] = {1.0f, 2.0f, 3.0f, 4.0f};
                scale_array(arr, 4, 2.5f);
                printf("%g %g %g %g\n", arr[0], arr[1], arr[2], arr[3]);
                return 0;
            }"#,
            "2.5 5 7.5 10",
        );
    }

    #[test]
    fn test_two_pointer_params_c_interop() {
        assert_c_interop(
            r#"
            export func add_arrays(a: *i32, b: *i32, out: *mut i32, len: i32) {
                let mut i: i32 = 0
                while i < len {
                    out[i] = a[i] + b[i]
                    i = i + 1
                }
            }
        "#,
            r#"#include <stdio.h>
            extern void add_arrays(const int*, const int*, int*, int);
            int main() {
                int a[] = {1, 2, 3, 4};
                int b[] = {10, 20, 30, 40};
                int out[4];
                add_arrays(a, b, out, 4);
                printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
                return 0;
            }"#,
            "11 22 33 44",
        );
    }

    #[test]
    fn test_pointer_find_max_c_interop() {
        assert_c_interop(
            r#"
            export func find_max(data: *i32, len: i32) -> i32 {
                let mut best: i32 = data[0]
                let mut i: i32 = 1
                while i < len {
                    if data[i] > best {
                        best = data[i]
                    }
                    i = i + 1
                }
                return best
            }
        "#,
            r#"#include <stdio.h>
            extern int find_max(const int*, int);
            int main() {
                int arr[] = {3, 7, 2, 9, 4};
                printf("%d\n", find_max(arr, 5));
                return 0;
            }"#,
            "9",
        );
    }

    #[test]
    fn test_dot_product_c_interop() {
        assert_c_interop(
            r#"
            export func dot_product(a: *f32, b: *f32, len: i32) -> f32 {
                let mut sum: f32 = 0.0
                let mut i: i32 = 0
                while i < len {
                    sum = sum + a[i] * b[i]
                    i = i + 1
                }
                return sum
            }
        "#,
            r#"#include <stdio.h>
            extern float dot_product(const float*, const float*, int);
            int main() {
                float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
                float b[] = {5.0f, 6.0f, 7.0f, 8.0f};
                printf("%g\n", dot_product(a, b, 4));
                return 0;
            }"#,
            "70",
        );
    }

    #[test]
    fn test_copy_array_c_interop() {
        assert_c_interop(
            r#"
            export func copy(src: *i32, dst: *mut i32, len: i32) {
                let mut i: i32 = 0
                while i < len {
                    dst[i] = src[i]
                    i = i + 1
                }
            }
        "#,
            r#"#include <stdio.h>
            extern void copy(const int*, int*, int);
            int main() {
                int src[] = {100, 200, 300};
                int dst[3] = {0};
                copy(src, dst, 3);
                printf("%d %d %d\n", dst[0], dst[1], dst[2]);
                return 0;
            }"#,
            "100 200 300",
        );
    }
}
