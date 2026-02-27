#[cfg(feature = "llvm")]
mod tests {
    use ea_compiler::error::format_with_source;

    fn compile_err(source: &str) -> ea_compiler::error::CompileError {
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        ea_compiler::check_types(&stmts).unwrap_err()
    }

    fn error_msg(source: &str) -> String {
        format!("{}", compile_err(source))
    }

    fn error_with_source(source: &str) -> String {
        let err = compile_err(source);
        format_with_source(&err, "test.ea", source)
    }

    // --- Type names use Eä syntax, not Rust debug format ---

    #[test]
    fn test_error_shows_ea_type_names() {
        let src = "export func bad(x: f32) {\n    let y: i32 = x\n    return\n}";
        let msg = error_msg(src);
        assert!(
            msg.contains("f32") && !msg.contains("F32"),
            "expected Eä type name f32, got: {msg}"
        );
        assert!(
            msg.contains("i32") && !msg.contains("I32"),
            "expected Eä type name i32, got: {msg}"
        );
    }

    #[test]
    fn test_error_shows_vector_type_name() {
        let src = "export func bad(v: f32x4) {\n    let x: i32 = v\n    return\n}";
        let msg = error_msg(src);
        assert!(
            msg.contains("f32x4"),
            "expected vector type f32x4 in error, got: {msg}"
        );
    }

    #[test]
    fn test_error_shows_pointer_type_name() {
        let src = "export func bad(p: *mut f32) {\n    let x: i32 = p\n    return\n}";
        let msg = error_msg(src);
        assert!(
            msg.contains("*mut f32"),
            "expected pointer type *mut f32 in error, got: {msg}"
        );
    }

    // --- Real line numbers ---

    #[test]
    fn test_error_reports_correct_line() {
        let src = "export func bad() {\n    let x: i32 = 1\n    let y: f32 = x\n    return\n}";
        let msg = error_msg(src);
        // The error should be on line 3 (let y: f32 = x)
        assert!(msg.contains("3:"), "expected line 3 in error, got: {msg}");
    }

    #[test]
    fn test_error_not_line_1() {
        // Regression: before Phase 11, all errors were at 1:1
        let src = "export func first() {\n    return\n}\nexport func second() {\n    let x: i32 = 1\n    let y: f32 = x\n    return\n}";
        let msg = error_msg(src);
        assert!(
            !msg.starts_with("error[type] 1:1"),
            "error should not be at 1:1, got: {msg}"
        );
    }

    // --- Source context display ---

    #[test]
    fn test_error_shows_source_line() {
        let src = "export func bad() {\n    let x: i32 = 1\n    let y: f32 = x\n    return\n}";
        let formatted = error_with_source(src);
        assert!(
            formatted.contains("let y: f32 = x"),
            "expected source line in error, got:\n{formatted}"
        );
    }

    #[test]
    fn test_error_shows_caret() {
        let src = "export func bad() {\n    let y: f32 = 1\n    let z: i32 = y\n    return\n}";
        let formatted = error_with_source(src);
        assert!(
            formatted.contains('^'),
            "expected caret in error, got:\n{formatted}"
        );
    }

    #[test]
    fn test_error_shows_filename() {
        let src = "export func bad() {\n    let y: f32 = 1\n    let z: i32 = y\n    return\n}";
        let formatted = error_with_source(src);
        assert!(
            formatted.starts_with("test.ea:"),
            "expected filename prefix, got:\n{formatted}"
        );
    }

    // --- Helpful suggestions ---

    #[test]
    fn test_immutable_pointer_suggestion() {
        let src = "export func bad(p: *f32, n: i32) {\n    p[0] = 1.0\n    return\n}";
        let msg = error_msg(src);
        assert!(
            msg.contains("*mut"),
            "expected *mut suggestion in immutable pointer error, got: {msg}"
        );
    }

    #[test]
    fn test_store_immutable_pointer_suggestion() {
        let src = "export func bad(p: *f32, n: i32) {\n    let v: f32x4 = splat(1.0)\n    store(p, 0, v)\n    return\n}";
        let msg = error_msg(src);
        assert!(
            msg.contains("*mut"),
            "expected *mut suggestion in store error, got: {msg}"
        );
    }

    #[test]
    fn test_index_non_pointer_suggestion() {
        let src = "export func bad(x: f32) -> f32 {\n    return x[0]\n}";
        let msg = error_msg(src);
        assert!(
            msg.contains("Only pointers and vectors support indexing"),
            "expected indexing suggestion, got: {msg}"
        );
    }

    #[test]
    fn test_select_mask_suggestion() {
        let src = "export func bad(a: f32x4, b: f32x4) -> f32x4 {\n    return select(a, a, b)\n}";
        let msg = error_msg(src);
        assert!(
            msg.contains(".>") || msg.contains(".=="),
            "expected comparison operator suggestion, got: {msg}"
        );
    }

    #[test]
    fn test_fma_integer_vector_suggestion() {
        let src =
            "export func bad(a: i32x4, b: i32x4, c: i32x4) -> i32x4 {\n    return fma(a, b, c)\n}";
        let msg = error_msg(src);
        assert!(
            msg.contains("f32") || msg.contains("f64"),
            "expected float vector suggestion in fma error, got: {msg}"
        );
    }

    // --- Type conversion hints ---

    #[test]
    fn test_assign_conversion_hint_to_i32() {
        let src = "export func bad(x: f32) {\n    let mut y: i32 = 0\n    y = x\n    return\n}";
        let msg = error_msg(src);
        assert!(
            msg.contains("Use to_i32() to convert"),
            "expected to_i32() conversion hint, got: {msg}"
        );
    }

    #[test]
    fn test_let_init_conversion_hint_to_f32() {
        let src = "export func bad(x: i32) {\n    let y: f32 = x\n    return\n}";
        let msg = error_msg(src);
        assert!(
            msg.contains("Use to_f32() to convert"),
            "expected to_f32() conversion hint, got: {msg}"
        );
    }

    #[test]
    fn test_no_conversion_hint_for_non_numeric() {
        // Pointer-to-int mismatch should not suggest conversion
        let src = "export func bad(p: *f32) {\n    let y: i32 = p\n    return\n}";
        let msg = error_msg(src);
        assert!(
            !msg.contains("to convert"),
            "should not suggest conversion for non-numeric mismatch, got: {msg}"
        );
    }

    // --- load error includes received type ---

    #[test]
    fn test_load_non_pointer_shows_type() {
        let src = "export func bad(x: i32) {\n    let v: f32x4 = load(x, 0)\n    return\n}";
        let msg = error_msg(src);
        assert!(
            msg.contains("got i32"),
            "expected 'got i32' in load error, got: {msg}"
        );
    }

    // --- Undefined variable ---

    #[test]
    fn test_undefined_variable_error() {
        let src = "export func bad() -> i32 {\n    return xyz\n}";
        let msg = error_msg(src);
        assert!(
            msg.contains("undefined variable 'xyz'"),
            "expected undefined variable message, got: {msg}"
        );
    }

    // --- Return type mismatch ---

    #[test]
    fn test_return_type_mismatch() {
        let src = "export func bad() -> i32 {\n    return 1.5\n}";
        let msg = error_msg(src);
        assert!(
            msg.contains("returns") && msg.contains("expected"),
            "expected return type mismatch message, got: {msg}"
        );
    }

    // --- Wrong argument count ---

    #[test]
    fn test_wrong_arg_count() {
        let src = "func helper(a: i32, b: i32) -> i32 {\n    return a + b\n}\nexport func bad() -> i32 {\n    return helper(1)\n}";
        let msg = error_msg(src);
        assert!(
            msg.contains("expects 2 arguments, got 1"),
            "expected argument count error, got: {msg}"
        );
    }

    // --- exported_function_names helper ---

    #[test]
    fn test_exported_function_names() {
        let src = "func internal(x: i32) -> i32 { return x }\nexport func alpha(x: i32) -> i32 { return x }\nexport func beta(x: i32, y: i32) -> i32 { return x + y }";
        let tokens = ea_compiler::tokenize(src).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let exports = ea_compiler::ast::exported_function_names(&stmts);
        assert_eq!(exports, vec!["alpha", "beta"]);
    }
}
