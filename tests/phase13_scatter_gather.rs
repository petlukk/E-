#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // --- Gather correctness: f32x4 ---

    #[test]
    fn test_gather_f32x4() {
        assert_c_interop(
            r#"
            export func test(data: *f32, idx: *i32, out: *mut f32) {
                let indices: i32x4 = load(idx, 0)
                let gathered: f32x4 = gather(data, indices)
                store(out, 0, gathered)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const float*, const int*, float*);
            int main() {
                float data[] = {10.0f, 20.0f, 30.0f, 40.0f};
                int idx[] = {0, 1, 2, 3};
                float out[4] = {0};
                test(data, idx, out);
                printf("%g %g %g %g\n", out[0], out[1], out[2], out[3]);
                return 0;
            }"#,
            "10 20 30 40",
        );
    }

    #[test]
    fn test_gather_f32x8() {
        assert_c_interop(
            r#"
            export func test(data: *f32, idx: *i32, out: *mut f32) {
                let indices: i32x8 = load(idx, 0)
                let gathered: f32x8 = gather(data, indices)
                store(out, 0, gathered)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const float*, const int*, float*);
            int main() {
                float data[] = {0,10,20,30,40,50,60,70,80,90};
                int idx[] = {0, 2, 4, 6, 1, 3, 5, 7};
                float out[8] = {0};
                test(data, idx, out);
                printf("%g %g %g %g %g %g %g %g\n",
                    out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
                return 0;
            }"#,
            "0 20 40 60 10 30 50 70",
        );
    }

    #[test]
    fn test_gather_i32x4() {
        assert_c_interop(
            r#"
            export func test(data: *i32, idx: *i32, out: *mut i32) {
                let indices: i32x4 = load(idx, 0)
                let gathered: i32x4 = gather(data, indices)
                store(out, 0, gathered)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const int*, const int*, int*);
            int main() {
                int data[] = {100, 200, 300, 400, 500};
                int idx[] = {4, 2, 0, 3};
                int out[4] = {0};
                test(data, idx, out);
                printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
                return 0;
            }"#,
            "500 300 100 400",
        );
    }

    #[test]
    fn test_gather_out_of_order() {
        assert_c_interop(
            r#"
            export func test(data: *f32, idx: *i32, out: *mut f32) {
                let indices: i32x4 = load(idx, 0)
                let gathered: f32x4 = gather(data, indices)
                store(out, 0, gathered)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const float*, const int*, float*);
            int main() {
                float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
                int idx[] = {3, 0, 2, 1};
                float out[4] = {0};
                test(data, idx, out);
                printf("%g %g %g %g\n", out[0], out[1], out[2], out[3]);
                return 0;
            }"#,
            "4 1 3 2",
        );
    }

    // --- Gather IR ---

    #[test]
    fn test_gather_ir() {
        let ir = compile_to_ir(
            r#"
            export func test(data: *f32, idx: *i32, out: *mut f32) {
                let indices: i32x4 = load(idx, 0)
                let gathered: f32x4 = gather(data, indices)
                store(out, 0, gathered)
            }
        "#,
        );
        assert!(
            ir.contains("llvm.masked.gather"),
            "IR must contain llvm.masked.gather intrinsic call"
        );
    }

    // --- Scatter requires AVX-512 ---

    #[test]
    fn test_scatter_requires_avx512() {
        let result = ea_compiler::compile_to_ir(
            r#"
            export func test(data: *mut f32, idx: *i32, vals: *f32) {
                let indices: i32x4 = load(idx, 0)
                let values: f32x4 = load(vals, 0)
                scatter(data, indices, values)
            }
        "#,
        );
        assert!(result.is_err(), "scatter without --avx512 should fail");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("avx512") || err.contains("AVX-512"),
            "error should mention avx512, got: {err}"
        );
    }

    // --- Type error tests ---

    #[test]
    fn test_gather_non_pointer_error() {
        let result = ea_compiler::compile_to_ir(
            r#"
            export func test(data: i32, idx: *i32) {
                let indices: i32x4 = load(idx, 0)
                let gathered: f32x4 = gather(data, indices)
                println(gathered[0])
            }
        "#,
        );
        assert!(result.is_err(), "gather with non-pointer base should fail");
    }

    #[test]
    fn test_gather_non_vector_indices_error() {
        let result = ea_compiler::compile_to_ir(
            r#"
            export func test(data: *f32, idx: i32) {
                let gathered: f32x4 = gather(data, idx)
                println(gathered[0])
            }
        "#,
        );
        assert!(result.is_err(), "gather with scalar index should fail");
    }

    #[test]
    fn test_scatter_immutable_pointer_error() {
        let result = ea_compiler::compile_to_ir(
            r#"
            export func test(data: *f32, idx: *i32, vals: *f32) {
                let indices: i32x4 = load(idx, 0)
                let values: f32x4 = load(vals, 0)
                scatter(data, indices, values)
            }
        "#,
        );
        assert!(
            result.is_err(),
            "scatter with immutable pointer should fail"
        );
    }
}
