#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    /// Showcase: multiple structs, const, helper func, three exported kernels in one file.
    /// Proves the full pipeline handles multi-kernel files end-to-end.
    #[test]
    fn test_multi_kernel_showcase() {
        assert_c_interop(
            r#"
            struct Vec2 { x: f32, y: f32 }
            struct Result { sum: f32, product: f32 }

            const SCALE: f32 = 2.0

            func clamp_f32(v: f32, lo: f32, hi: f32) -> f32 {
                if v < lo { return lo }
                if v > hi { return hi }
                return v
            }

            export func add_arrays(a: *f32, b: *f32, out: *mut f32, n: i32) {
                let mut i: i32 = 0
                while i < n {
                    out[i] = (a[i] + b[i]) * SCALE
                    i = i + 1
                }
            }

            export func negate_array(data: *f32, out: *mut f32, n: i32) {
                let mut i: i32 = 0
                while i < n {
                    out[i] = data[i] * -1.0
                    i = i + 1
                }
            }

            export func clamp_array(data: *f32, out: *mut f32, lo: f32, hi: f32, n: i32) {
                let mut i: i32 = 0
                while i < n {
                    out[i] = clamp_f32(data[i], lo, hi)
                    i = i + 1
                }
            }
        "#,
            r#"
            #include <stdio.h>
            extern void add_arrays(float*, float*, float*, int);
            extern void negate_array(float*, float*, int);
            extern void clamp_array(float*, float*, float, float, int);
            int main() {
                float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
                float b[4] = {10.0f, 20.0f, 30.0f, 40.0f};
                float out[4];

                // Test add_arrays: (a+b)*SCALE
                add_arrays(a, b, out, 4);
                printf("add: %.0f %.0f %.0f %.0f\n", out[0], out[1], out[2], out[3]);

                // Test negate_array
                negate_array(a, out, 4);
                printf("neg: %.0f %.0f %.0f %.0f\n", out[0], out[1], out[2], out[3]);

                // Test clamp_array
                float vals[4] = {-5.0f, 3.0f, 15.0f, 7.0f};
                clamp_array(vals, out, 0.0f, 10.0f, 4);
                printf("clamp: %.0f %.0f %.0f %.0f\n", out[0], out[1], out[2], out[3]);

                return 0;
            }
        "#,
            "add: 22 44 66 88\nneg: -1 -2 -3 -4\nclamp: 0 3 10 7",
        );
    }

    /// Multi-kernel file with kernels (not just funcs)
    #[test]
    fn test_multi_kernel_with_kernel_construct() {
        assert_c_interop(
            r#"
            const OFFSET: i32 = 100

            export kernel add_offset(data: *i32, out: *mut i32)
                over i in n step 1
            {
                out[i] = data[i] + OFFSET
            }

            export kernel double_it(data: *i32, out: *mut i32)
                over i in n step 1
            {
                out[i] = data[i] * 2
            }
        "#,
            r#"
            #include <stdio.h>
            extern void add_offset(int*, int*, int);
            extern void double_it(int*, int*, int);
            int main() {
                int data[3] = {1, 2, 3};
                int out[3];

                add_offset(data, out, 3);
                printf("offset: %d %d %d\n", out[0], out[1], out[2]);

                double_it(data, out, 3);
                printf("double: %d %d %d\n", out[0], out[1], out[2]);

                return 0;
            }
        "#,
            "offset: 101 102 103\ndouble: 2 4 6",
        );
    }

    /// Multi-export metadata contains all functions
    #[test]
    fn test_multi_kernel_metadata() {
        let source = r#"
            export func alpha(x: *f32, n: i32) { }
            export func beta(x: *f32, n: i32) { }
            export func gamma(x: *f32, n: i32) { }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts).unwrap();
        let json = ea_compiler::metadata::generate_json(&stmts, "multi");
        assert!(json.contains("\"name\": \"alpha\""), "missing alpha");
        assert!(json.contains("\"name\": \"beta\""), "missing beta");
        assert!(json.contains("\"name\": \"gamma\""), "missing gamma");
    }

    /// Multi-export header contains all functions
    #[test]
    fn test_multi_kernel_header() {
        let source = r#"
            export func alpha(x: *f32, n: i32) { }
            export func beta(x: *f32, n: i32) { }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts).unwrap();
        let header = ea_compiler::header::generate(&stmts, "multi");
        assert!(header.contains("alpha("), "missing alpha in header");
        assert!(header.contains("beta("), "missing beta in header");
    }
}
