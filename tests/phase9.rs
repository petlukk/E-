#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === opt-level fix ===

    #[test]
    fn test_opt_level_0_still_correct() {
        use std::process::Command;
        use tempfile::TempDir;

        let ea_source = r#"
            export func sum(data: *f32, n: i32) -> f32 {
                let mut result: f32 = 0.0
                let mut i: i32 = 0
                while i < n {
                    result = result + data[i]
                    i = i + 1
                }
                return result
            }
        "#;
        let c_source = r#"
            #include <stdio.h>
            extern float sum(const float*, int);
            int main() {
                float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
                printf("%.0f\n", sum(data, 4));
                return 0;
            }
        "#;

        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("kernel.o");
        let c_path = dir.path().join("harness.c");
        let bin = dir.path().join("test_bin");

        let opts = ea_compiler::CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: String::new(),
        };
        ea_compiler::compile_with_options(
            ea_source,
            &obj,
            ea_compiler::OutputMode::ObjectFile,
            &opts,
        )
        .expect("compile at O0 failed");

        std::fs::write(&c_path, c_source).unwrap();
        let status = Command::new("cc")
            .args([
                c_path.to_str().unwrap(),
                obj.to_str().unwrap(),
                "-o",
                bin.to_str().unwrap(),
                "-lm",
            ])
            .status()
            .unwrap();
        assert!(status.success());
        let out = Command::new(&bin).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "10");
    }

    #[test]
    fn test_opt_level_3_correct() {
        use std::process::Command;
        use tempfile::TempDir;

        let ea_source = r#"
            export func sum(data: *f32, n: i32) -> f32 {
                let mut result: f32 = 0.0
                let mut i: i32 = 0
                while i < n {
                    result = result + data[i]
                    i = i + 1
                }
                return result
            }
        "#;
        let c_source = r#"
            #include <stdio.h>
            extern float sum(const float*, int);
            int main() {
                float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
                printf("%.0f\n", sum(data, 5));
                return 0;
            }
        "#;

        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("kernel.o");
        let c_path = dir.path().join("harness.c");
        let bin = dir.path().join("test_bin");

        let opts = ea_compiler::CompileOptions {
            opt_level: 3,
            target_cpu: None,
            extra_features: String::new(),
        };
        ea_compiler::compile_with_options(
            ea_source,
            &obj,
            ea_compiler::OutputMode::ObjectFile,
            &opts,
        )
        .expect("compile at O3 failed");

        std::fs::write(&c_path, c_source).unwrap();
        let status = Command::new("cc")
            .args([
                c_path.to_str().unwrap(),
                obj.to_str().unwrap(),
                "-o",
                bin.to_str().unwrap(),
                "-lm",
            ])
            .status()
            .unwrap();
        assert!(status.success());
        let out = Command::new(&bin).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "15");
    }

    // === emit-asm ===

    #[test]
    fn test_emit_asm_produces_file() {
        use tempfile::TempDir;
        let dir = TempDir::new().unwrap();
        let asm_path = dir.path().join("kernel.s");

        let source = r#"
            export func add(a: i32, b: i32) -> i32 {
                return a + b
            }
        "#;
        ea_compiler::compile_with_options(
            source,
            &asm_path,
            ea_compiler::OutputMode::Asm,
            &ea_compiler::CompileOptions::default(),
        )
        .expect("asm emission failed");

        let asm = std::fs::read_to_string(&asm_path).unwrap();
        assert!(asm.contains("add"), "assembly should contain 'add' symbol");
        assert!(asm.len() > 20, "assembly should not be empty");
    }

    // === unroll(N) ===

    #[test]
    fn test_unroll_basic_while() {
        let ea_source = r#"
            export func sum_unrolled(data: *f32, n: i32) -> f32 {
                let mut result: f32 = 0.0
                let mut i: i32 = 0
                unroll(4) while i < n {
                    result = result + data[i]
                    i = i + 1
                }
                return result
            }
        "#;
        let c_source = r#"
            #include <stdio.h>
            extern float sum_unrolled(const float*, int);
            int main() {
                float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
                printf("%.0f\n", sum_unrolled(data, 8));
                return 0;
            }
        "#;
        assert_c_interop(ea_source, c_source, "36");
    }

    #[test]
    fn test_unroll_non_multiple_trip_count() {
        let ea_source = r#"
            export func sum_unrolled(data: *f32, n: i32) -> f32 {
                let mut result: f32 = 0.0
                let mut i: i32 = 0
                unroll(4) while i < n {
                    result = result + data[i]
                    i = i + 1
                }
                return result
            }
        "#;
        let c_source = r#"
            #include <stdio.h>
            extern float sum_unrolled(const float*, int);
            int main() {
                float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
                printf("%.0f\n", sum_unrolled(data, 5));
                return 0;
            }
        "#;
        assert_c_interop(ea_source, c_source, "15");
    }

    // === prefetch ===

    #[test]
    fn test_prefetch_basic() {
        let ea_source = r#"
            export func sum_prefetched(data: *f32, n: i32) -> f32 {
                let mut result: f32 = 0.0
                let mut i: i32 = 0
                while i < n {
                    prefetch(data, i + 16)
                    result = result + data[i]
                    i = i + 1
                }
                return result
            }
        "#;
        let c_source = r#"
            #include <stdio.h>
            extern float sum_prefetched(const float*, int);
            int main() {
                float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
                printf("%.0f\n", sum_prefetched(data, 4));
                return 0;
            }
        "#;
        assert_c_interop(ea_source, c_source, "10");
    }

    // === header generation ===

    #[test]
    fn test_header_generation_basic() {
        let source = r#"
            export func add(a: i32, b: i32) -> i32 {
                return a + b
            }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        ea_compiler::check_types(&stmts).unwrap();

        let header = ea_compiler::header::generate(&stmts, "test");
        assert!(header.contains("int32_t add(int32_t a, int32_t b)"));
        assert!(header.contains("#ifndef TEST_H"));
        assert!(header.contains("#include <stdint.h>"));
    }

    #[test]
    fn test_header_generation_pointers() {
        let source = r#"
            export func process(data: *f32, out: *mut f32, n: i32) {
                let mut i: i32 = 0
                while i < n {
                    out[i] = data[i]
                    i = i + 1
                }
            }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        ea_compiler::check_types(&stmts).unwrap();

        let header = ea_compiler::header::generate(&stmts, "test");
        assert!(header.contains("const float*"));
        assert!(header.contains("float*"));
    }

    #[test]
    fn test_header_generation_struct() {
        let source = r#"
            struct Vec2 {
                x: f32,
                y: f32,
            }
            export func get_x(v: *Vec2) -> f32 {
                return v.x
            }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        ea_compiler::check_types(&stmts).unwrap();

        let header = ea_compiler::header::generate(&stmts, "test");
        assert!(header.contains("struct Vec2"));
        assert!(header.contains("float x"));
        assert!(header.contains("float y"));
    }
}
