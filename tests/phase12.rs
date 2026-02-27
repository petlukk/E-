#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // --- load_masked: f32x4 ---

    #[test]
    fn test_load_masked_f32x4_full() {
        assert_c_interop(
            r#"
            export func test(data: *f32, out: *mut f32) {
                let v: f32x4 = load_masked(data, 0, 4)
                store(out, 0, v)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const float*, float*);
            int main() {
                float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
                float out[4] = {0};
                test(data, out);
                printf("%g %g %g %g\n", out[0], out[1], out[2], out[3]);
                return 0;
            }"#,
            "1 2 3 4",
        );
    }

    #[test]
    fn test_load_masked_f32x4_partial() {
        assert_c_interop(
            r#"
            export func test(data: *f32, out: *mut f32) {
                let v: f32x4 = load_masked(data, 0, 2)
                store(out, 0, v)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const float*, float*);
            int main() {
                float data[] = {10.0f, 20.0f, 30.0f, 40.0f};
                float out[4] = {-1, -1, -1, -1};
                test(data, out);
                printf("%g %g %g %g\n", out[0], out[1], out[2], out[3]);
                return 0;
            }"#,
            "10 20 0 0",
        );
    }

    #[test]
    fn test_load_masked_f32x4_one() {
        assert_c_interop(
            r#"
            export func test(data: *f32, out: *mut f32) {
                let v: f32x4 = load_masked(data, 0, 1)
                store(out, 0, v)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const float*, float*);
            int main() {
                float data[] = {42.0f, 99.0f, 99.0f, 99.0f};
                float out[4] = {-1, -1, -1, -1};
                test(data, out);
                printf("%g %g %g %g\n", out[0], out[1], out[2], out[3]);
                return 0;
            }"#,
            "42 0 0 0",
        );
    }

    #[test]
    fn test_load_masked_f32x4_zero() {
        assert_c_interop(
            r#"
            export func test(data: *f32, out: *mut f32) {
                let v: f32x4 = load_masked(data, 0, 0)
                store(out, 0, v)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const float*, float*);
            int main() {
                float data[] = {99.0f, 99.0f, 99.0f, 99.0f};
                float out[4] = {-1, -1, -1, -1};
                test(data, out);
                printf("%g %g %g %g\n", out[0], out[1], out[2], out[3]);
                return 0;
            }"#,
            "0 0 0 0",
        );
    }

    // --- store_masked: f32x4 ---

    #[test]
    fn test_store_masked_f32x4_partial() {
        assert_c_interop(
            r#"
            export func test(data: *f32, out: *mut f32) {
                let v: f32x4 = load(data, 0)
                store_masked(out, 0, v, 2)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const float*, float*);
            int main() {
                float data[] = {10.0f, 20.0f, 30.0f, 40.0f};
                float out[] = {-1.0f, -1.0f, -1.0f, -1.0f};
                test(data, out);
                printf("%g %g %g %g\n", out[0], out[1], out[2], out[3]);
                return 0;
            }"#,
            "10 20 -1 -1",
        );
    }

    #[test]
    fn test_store_masked_f32x4_zero() {
        assert_c_interop(
            r#"
            export func test(data: *f32, out: *mut f32) {
                let v: f32x4 = load(data, 0)
                store_masked(out, 0, v, 0)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const float*, float*);
            int main() {
                float data[] = {10.0f, 20.0f, 30.0f, 40.0f};
                float out[] = {-1.0f, -1.0f, -1.0f, -1.0f};
                test(data, out);
                printf("%g %g %g %g\n", out[0], out[1], out[2], out[3]);
                return 0;
            }"#,
            "-1 -1 -1 -1",
        );
    }

    // --- f32x8 ---

    #[test]
    fn test_load_masked_f32x8_partial() {
        assert_c_interop(
            r#"
            export func test(data: *f32, out: *mut f32) {
                let v: f32x8 = load_masked(data, 0, 3)
                store(out, 0, v)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const float*, float*);
            int main() {
                float data[] = {1,2,3,4,5,6,7,8};
                float out[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
                test(data, out);
                printf("%g %g %g %g %g %g %g %g\n",
                    out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
                return 0;
            }"#,
            "1 2 3 0 0 0 0 0",
        );
    }

    #[test]
    fn test_store_masked_f32x8_partial() {
        assert_c_interop(
            r#"
            export func test(data: *f32, out: *mut f32) {
                let v: f32x8 = load(data, 0)
                store_masked(out, 0, v, 5)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const float*, float*);
            int main() {
                float data[] = {10,20,30,40,50,60,70,80};
                float out[] = {-1,-1,-1,-1,-1,-1,-1,-1};
                test(data, out);
                printf("%g %g %g %g %g %g %g %g\n",
                    out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
                return 0;
            }"#,
            "10 20 30 40 50 -1 -1 -1",
        );
    }

    // --- i32x4 ---

    #[test]
    fn test_load_masked_i32x4_partial() {
        assert_c_interop(
            r#"
            export func test(data: *i32, out: *mut i32) {
                let v: i32x4 = load_masked(data, 0, 3)
                store(out, 0, v)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const int*, int*);
            int main() {
                int data[] = {100, 200, 300, 400};
                int out[4] = {-1, -1, -1, -1};
                test(data, out);
                printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
                return 0;
            }"#,
            "100 200 300 0",
        );
    }

    #[test]
    fn test_store_masked_i32x4_partial() {
        assert_c_interop(
            r#"
            export func test(data: *i32, out: *mut i32) {
                let v: i32x4 = load(data, 0)
                store_masked(out, 0, v, 1)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const int*, int*);
            int main() {
                int data[] = {100, 200, 300, 400};
                int out[] = {-1, -1, -1, -1};
                test(data, out);
                printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
                return 0;
            }"#,
            "100 -1 -1 -1",
        );
    }

    // --- Dynamic count (runtime value, not compile-time constant) ---

    #[test]
    fn test_load_masked_dynamic_count() {
        assert_c_interop(
            r#"
            export func test(data: *f32, out: *mut f32, count: i32) {
                let v: f32x4 = load_masked(data, 0, count)
                store(out, 0, v)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const float*, float*, int);
            int main() {
                float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
                float out[4];
                out[0] = -1; out[1] = -1; out[2] = -1; out[3] = -1;
                test(data, out, 3);
                printf("%g %g %g %g\n", out[0], out[1], out[2], out[3]);
                return 0;
            }"#,
            "1 2 3 0",
        );
    }

    // --- Spec example: scale kernel with masked tail ---

    #[test]
    fn test_scale_kernel_with_masked_tail() {
        assert_c_interop(
            r#"
            export func scale(data: *f32, out: *mut f32, n: i32, factor: f32) {
                let vfactor: f32x8 = splat(factor)
                let mut i: i32 = 0
                while i + 8 <= n {
                    let vd: f32x8 = load(data, i)
                    store(out, i, vd .* vfactor)
                    i = i + 8
                }
                let rem: i32 = n - i
                if rem > 0 {
                    let tail: f32x8 = load_masked(data, i, rem)
                    store_masked(out, i, tail .* vfactor, rem)
                }
            }
        "#,
            r#"#include <stdio.h>
            extern void scale(const float*, float*, int, float);
            int main() {
                float data[] = {1,2,3,4,5,6,7,8,9,10,11};
                float out[11] = {0};
                scale(data, out, 11, 2.0f);
                printf("%g %g %g %g %g %g %g %g %g %g %g\n",
                    out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],
                    out[8],out[9],out[10]);
                return 0;
            }"#,
            "2 4 6 8 10 12 14 16 18 20 22",
        );
    }

    // --- IR inspection ---

    #[test]
    fn test_load_masked_ir() {
        let ir = compile_to_ir(
            r#"
            export func test(data: *f32, out: *mut f32, count: i32) {
                let v: f32x4 = load_masked(data, 0, count)
                store(out, 0, v)
            }
        "#,
        );
        assert!(
            ir.contains("llvm.masked.load"),
            "IR must contain llvm.masked.load intrinsic call"
        );
    }

    #[test]
    fn test_store_masked_ir() {
        let ir = compile_to_ir(
            r#"
            export func test(data: *f32, out: *mut f32, count: i32) {
                let v: f32x4 = load(data, 0)
                store_masked(out, 0, v, count)
            }
        "#,
        );
        assert!(
            ir.contains("llvm.masked.store"),
            "IR must contain llvm.masked.store intrinsic call"
        );
    }

    // --- Type error tests ---

    #[test]
    fn test_load_masked_rejects_non_pointer() {
        let result = ea_compiler::compile_to_ir(
            r#"
            func main() {
                let v: f32x4 = load_masked(42, 0, 2)
                println(v[0])
            }
        "#,
        );
        assert!(result.is_err(), "load_masked with non-pointer should fail");
    }

    #[test]
    fn test_store_masked_rejects_immutable_pointer() {
        let result = ea_compiler::compile_to_ir(
            r#"
            export func test(data: *f32, out: *f32) {
                let v: f32x4 = load(data, 0)
                store_masked(out, 0, v, 2)
            }
        "#,
        );
        assert!(
            result.is_err(),
            "store_masked with immutable pointer should fail"
        );
    }

    #[test]
    fn test_load_masked_wrong_arg_count() {
        let result = ea_compiler::compile_to_ir(
            r#"
            export func test(data: *f32) {
                let v: f32x4 = load_masked(data, 0)
                println(v[0])
            }
        "#,
        );
        assert!(result.is_err(), "load_masked with 2 args should fail");
    }

    #[test]
    fn test_store_masked_wrong_arg_count() {
        let result = ea_compiler::compile_to_ir(
            r#"
            export func test(data: *f32, out: *mut f32) {
                let v: f32x4 = load(data, 0)
                store_masked(out, 0, v)
            }
        "#,
        );
        assert!(result.is_err(), "store_masked with 3 args should fail");
    }

    // --- Offset test (load_masked with non-zero offset) ---

    #[test]
    fn test_load_masked_with_offset() {
        assert_c_interop(
            r#"
            export func test(data: *f32, out: *mut f32) {
                let v: f32x4 = load_masked(data, 4, 2)
                store(out, 0, v)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const float*, float*);
            int main() {
                float data[] = {1,2,3,4,50,60,70,80};
                float out[4] = {-1,-1,-1,-1};
                test(data, out);
                printf("%g %g %g %g\n", out[0], out[1], out[2], out[3]);
                return 0;
            }"#,
            "50 60 0 0",
        );
    }
}
