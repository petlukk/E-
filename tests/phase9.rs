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
}
