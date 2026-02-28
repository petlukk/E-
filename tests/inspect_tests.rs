#[cfg(feature = "llvm")]
mod tests {
    use ea_compiler::{inspect_source, CompileOptions};

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_inspect_simd_kernel_has_vector_instructions() {
        let source = r#"
            export kernel vscale(data: *f32, out: *mut f32, factor: f32)
                over i in n step 4
            {
                let v: f32x4 = load(data, i)
                store(out, i, v .* splat(factor))
            }
        "#;
        let report = inspect_source(source, &CompileOptions::default()).unwrap();
        let func = report
            .functions
            .iter()
            .find(|f| f.name == "vscale")
            .expect("vscale not found in report");
        assert!(
            func.vector_instructions > 0,
            "SIMD kernel should have vector instructions, got 0"
        );
        assert!(func.exported, "vscale should be exported");
    }

    #[test]
    fn test_inspect_scalar_function_no_vector() {
        let source = r#"
            export func add(a: i32, b: i32) -> i32 {
                return a + b
            }
        "#;
        let report = inspect_source(source, &CompileOptions::default()).unwrap();
        let func = report
            .functions
            .iter()
            .find(|f| f.name == "add")
            .expect("add not found in report");
        assert_eq!(
            func.vector_instructions, 0,
            "scalar function should have 0 vector instructions"
        );
    }

    #[test]
    fn test_inspect_kernel_has_loops() {
        let source = r#"
            export kernel inc(data: *i32, out: *mut i32)
                over i in n step 1
            {
                out[i] = data[i] + 1
            }
        "#;
        let report = inspect_source(source, &CompileOptions::default()).unwrap();
        let func = report
            .functions
            .iter()
            .find(|f| f.name == "inc")
            .expect("inc not found in report");
        assert!(
            func.loops >= 1,
            "kernel with while-loop should report >= 1 loop, got {}",
            func.loops
        );
    }

    #[test]
    fn test_inspect_multiple_exports() {
        let source = r#"
            export func alpha(x: i32) -> i32 { return x + 1 }
            export func beta(x: i32) -> i32 { return x * 2 }
        "#;
        let report = inspect_source(source, &CompileOptions::default()).unwrap();
        let names: Vec<&str> = report.functions.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains(&"alpha"), "missing alpha in report");
        assert!(names.contains(&"beta"), "missing beta in report");
    }

    #[test]
    fn test_inspect_display_format() {
        let source = r#"
            export func identity(x: i32) -> i32 { return x }
        "#;
        let report = inspect_source(source, &CompileOptions::default()).unwrap();
        let output = format!("{report}");
        assert!(
            output.contains("=== identity (exported) ==="),
            "display should contain function name with export tag, got:\n{output}"
        );
        assert!(
            output.contains("vector instructions:"),
            "display should contain 'vector instructions:', got:\n{output}"
        );
        assert!(
            output.contains("scalar instructions:"),
            "display should contain 'scalar instructions:', got:\n{output}"
        );
        assert!(
            output.contains("loops:"),
            "display should contain 'loops:', got:\n{output}"
        );
    }
}
