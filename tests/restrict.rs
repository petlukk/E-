#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::{assert_c_interop, compile_to_ir};

    #[test]
    fn test_restrict_noalias_in_ir() {
        let ir = compile_to_ir(
            r#"
            export func add_arrays(a: *restrict f32, b: *restrict f32, out: *restrict mut f32, len: i32) {
                let mut i: i32 = 0
                while i < len {
                    out[i] = a[i] + b[i]
                    i = i + 1
                }
            }
        "#,
        );
        assert!(
            ir.contains("noalias"),
            "IR should contain noalias attribute:\n{ir}"
        );
    }

    #[test]
    fn test_non_restrict_no_noalias() {
        let ir = compile_to_ir(
            r#"
            export func copy(src: *f32, dst: *mut f32, len: i32) {
                let mut i: i32 = 0
                while i < len {
                    dst[i] = src[i]
                    i = i + 1
                }
            }
        "#,
        );
        assert!(
            !ir.contains("noalias"),
            "IR should NOT contain noalias:\n{ir}"
        );
    }

    #[test]
    fn test_mixed_restrict_params() {
        let ir = compile_to_ir(
            r#"
            export func mixed(a: *restrict f32, b: *f32, out: *restrict mut f32, len: i32) {
                let mut i: i32 = 0
                while i < len {
                    out[i] = a[i] + b[i]
                    i = i + 1
                }
            }
        "#,
        );
        let noalias_count = ir.matches("noalias").count();
        assert_eq!(
            noalias_count, 2,
            "Expected 2 noalias attributes (a and out), got {noalias_count}:\n{ir}"
        );
    }

    #[test]
    fn test_restrict_c_interop() {
        assert_c_interop(
            r#"
            export func add_arrays(a: *restrict f32, b: *restrict f32, out: *restrict mut f32, len: i32) {
                let mut i: i32 = 0
                while i < len {
                    out[i] = a[i] + b[i]
                    i = i + 1
                }
            }
        "#,
            r#"
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
        "#,
            "11 22 33 44",
        );
    }
}
