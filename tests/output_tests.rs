// Tests for output annotations (parsing, display, metadata, type checking).

// === Parse tests ===

#[test]
fn test_parse_out_annotation_basic() {
    let source = r#"
        export func transform(data: *f32, out result: *mut f32 [cap: n], n: i32) {
            let mut i: i32 = 0
            while i < n {
                result[i] = data[i] * 2.0
                i = i + 1
            }
        }
    "#;
    let tokens = ea_compiler::tokenize(source).unwrap();
    let stmts = ea_compiler::parse(tokens).unwrap();

    if let ea_compiler::ast::Stmt::Function { params, .. } = &stmts[0] {
        assert_eq!(params.len(), 3);

        // data: regular input
        assert!(!params[0].output);
        assert!(params[0].cap.is_none());
        assert!(params[0].count.is_none());

        // out result: output with cap
        assert!(params[1].output);
        assert_eq!(params[1].name, "result");
        assert_eq!(params[1].cap.as_deref(), Some("n"));
        assert!(params[1].count.is_none());

        // n: regular input
        assert!(!params[2].output);
    } else {
        panic!("expected Function");
    }
}

#[test]
fn test_parse_out_with_cap_and_count() {
    let source = r#"
        export func sparse(vals: *f32, out indices: *mut i32 [cap: n, count: actual_count], n: i32, actual_count: *mut i32) {
            let mut i: i32 = 0
            while i < n {
                indices[i] = i
                i = i + 1
            }
        }
    "#;
    let tokens = ea_compiler::tokenize(source).unwrap();
    let stmts = ea_compiler::parse(tokens).unwrap();

    if let ea_compiler::ast::Stmt::Function { params, .. } = &stmts[0] {
        assert!(params[1].output);
        assert_eq!(params[1].cap.as_deref(), Some("n"));
        assert_eq!(params[1].count.as_deref(), Some("actual_count"));
    } else {
        panic!("expected Function");
    }
}

#[test]
fn test_parse_cap_expression() {
    let source = r#"
        export func padded(data: *f32, out result: *mut f32 [cap: n + 16], n: i32) {
            let mut i: i32 = 0
            while i < n {
                result[i] = data[i]
                i = i + 1
            }
        }
    "#;
    let tokens = ea_compiler::tokenize(source).unwrap();
    let stmts = ea_compiler::parse(tokens).unwrap();

    if let ea_compiler::ast::Stmt::Function { params, .. } = &stmts[0] {
        assert!(params[1].output);
        assert_eq!(params[1].cap.as_deref(), Some("n + 16"));
    } else {
        panic!("expected Function");
    }
}

#[test]
fn test_parse_out_without_cap() {
    // out without cap bracket — caller provides buffer
    let source = r#"
        export func copy(data: *f32, out result: *mut f32, n: i32) {
            let mut i: i32 = 0
            while i < n {
                result[i] = data[i]
                i = i + 1
            }
        }
    "#;
    let tokens = ea_compiler::tokenize(source).unwrap();
    let stmts = ea_compiler::parse(tokens).unwrap();

    if let ea_compiler::ast::Stmt::Function { params, .. } = &stmts[0] {
        assert!(params[1].output);
        assert_eq!(params[1].name, "result");
        assert!(params[1].cap.is_none());
    } else {
        panic!("expected Function");
    }
}

#[test]
fn test_parse_param_named_out() {
    // A param literally named "out" (not the keyword) — "out: *mut f32" without lookahead
    let source = r#"
        export func add(a: *f32, out: *mut f32, n: i32) {
            let mut i: i32 = 0
            while i < n {
                out[i] = a[i]
                i = i + 1
            }
        }
    "#;
    let tokens = ea_compiler::tokenize(source).unwrap();
    let stmts = ea_compiler::parse(tokens).unwrap();

    if let ea_compiler::ast::Stmt::Function { params, .. } = &stmts[0] {
        // "out" is just a name, not the keyword
        assert_eq!(params[1].name, "out");
        assert!(!params[1].output);
    } else {
        panic!("expected Function");
    }
}

// === Display test ===

#[test]
fn test_param_display_output() {
    let source = r#"
        export func f(data: *f32, out result: *mut f32 [cap: n], n: i32) {
            let mut i: i32 = 0
            while i < n {
                result[i] = data[i]
                i = i + 1
            }
        }
    "#;
    let tokens = ea_compiler::tokenize(source).unwrap();
    let stmts = ea_compiler::parse(tokens).unwrap();

    if let ea_compiler::ast::Stmt::Function { params, .. } = &stmts[0] {
        let s = format!("{}", params[1]);
        assert!(s.contains("out "), "expected 'out' prefix in display: {s}");
        assert!(s.contains("[cap: n]"), "expected [cap: n] in display: {s}");
    } else {
        panic!("expected Function");
    }
}

// === Metadata JSON tests ===

#[test]
fn test_metadata_direction_in_for_regular_params() {
    let source = r#"
        export func scale(data: *mut f32, n: i32, alpha: f32) {
            let mut i: i32 = 0
            while i < n {
                data[i] = data[i] * alpha
                i = i + 1
            }
        }
    "#;
    let tokens = ea_compiler::tokenize(source).unwrap();
    let stmts = ea_compiler::parse(tokens).unwrap();
    let json = ea_compiler::metadata::generate_json(&stmts, "scale.so");

    // All params should have direction "in"
    assert!(
        json.contains("\"direction\": \"in\""),
        "expected direction field, got:\n{json}"
    );
    // No "out" direction
    assert!(
        !json.contains("\"direction\": \"out\""),
        "did not expect out direction, got:\n{json}"
    );
    // cap and count should be null
    assert!(
        json.contains("\"cap\": null"),
        "expected null cap, got:\n{json}"
    );
    assert!(
        json.contains("\"count\": null"),
        "expected null count, got:\n{json}"
    );
}

#[test]
fn test_metadata_direction_out_with_cap() {
    let source = r#"
        export func transform(data: *f32, out result: *mut f32 [cap: n], n: i32) {
            let mut i: i32 = 0
            while i < n {
                result[i] = data[i] * 2.0
                i = i + 1
            }
        }
    "#;
    let tokens = ea_compiler::tokenize(source).unwrap();
    let stmts = ea_compiler::parse(tokens).unwrap();
    let json = ea_compiler::metadata::generate_json(&stmts, "transform.so");

    assert!(
        json.contains("\"direction\": \"out\""),
        "expected direction out for result, got:\n{json}"
    );
    assert!(
        json.contains("\"cap\": \"n\""),
        "expected cap field, got:\n{json}"
    );
}

#[test]
fn test_metadata_direction_out_with_cap_and_count() {
    let source = r#"
        export func sparse(vals: *f32, out idx: *mut i32 [cap: n, count: actual], n: i32, actual: *mut i32) {
            let mut i: i32 = 0
            while i < n {
                idx[i] = i
                i = i + 1
            }
        }
    "#;
    let tokens = ea_compiler::tokenize(source).unwrap();
    let stmts = ea_compiler::parse(tokens).unwrap();
    let json = ea_compiler::metadata::generate_json(&stmts, "sparse.so");

    assert!(json.contains("\"cap\": \"n\""), "got:\n{json}");
    assert!(json.contains("\"count\": \"actual\""), "got:\n{json}");
}

// === Type checker error tests ===

#[test]
fn test_typeck_out_on_immutable_pointer_error() {
    let source = r#"
        export func bad(out result: *f32 [cap: n], n: i32) {
            let mut i: i32 = 0
            while i < n { i = i + 1 }
        }
    "#;
    let tokens = ea_compiler::tokenize(source).unwrap();
    let stmts = ea_compiler::parse(tokens).unwrap();
    let err = ea_compiler::check_types(&stmts).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("*mut"), "expected *mut error, got: {msg}");
}

#[test]
fn test_typeck_out_on_non_pointer_error() {
    let source = r#"
        export func bad(out x: i32, n: i32) {
            let mut i: i32 = 0
            while i < n { i = i + 1 }
        }
    "#;
    let tokens = ea_compiler::tokenize(source).unwrap();
    let stmts = ea_compiler::parse(tokens).unwrap();
    let err = ea_compiler::check_types(&stmts).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("*mut pointer"),
        "expected pointer type error, got: {msg}"
    );
}
