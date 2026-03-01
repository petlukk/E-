// Tests for output annotation binding generation (Python, Rust, C++) and backward compatibility.

// === Python binding generation with out params ===

#[test]
fn test_python_out_param_auto_allocate() {
    let json = r#"{
  "library": "transform.so",
  "exports": [
    {
      "name": "transform",
      "args": [
        {"name": "data", "type": "*f32", "direction": "in", "cap": null, "count": null},
        {"name": "result", "type": "*mut f32", "direction": "out", "cap": "n", "count": null},
        {"name": "n", "type": "i32", "direction": "in", "cap": null, "count": null}
      ],
      "return_type": null
    }
  ],
  "structs": []
}"#;

    let py = ea_compiler::bind_python::generate(json, "transform").unwrap();

    // result should NOT appear in the function signature
    assert!(
        !py.contains("result: _np.ndarray"),
        "auto-allocated out param should not be in signature:\n{py}"
    );
    // Should auto-allocate
    assert!(
        py.contains("result = _np.empty(n, dtype=_np.float32)"),
        "expected auto-allocation of result:\n{py}"
    );
    // Should return result
    assert!(
        py.contains("return result"),
        "expected return result:\n{py}"
    );
    // Return annotation
    assert!(
        py.contains("-> _np.ndarray"),
        "expected ndarray return annotation:\n{py}"
    );
}

#[test]
fn test_python_out_param_no_cap_keeps_in_signature() {
    let json = r#"{
  "library": "copy.so",
  "exports": [
    {
      "name": "copy_data",
      "args": [
        {"name": "src", "type": "*f32", "direction": "in", "cap": null, "count": null},
        {"name": "dst", "type": "*mut f32", "direction": "out", "cap": null, "count": null},
        {"name": "n", "type": "i32", "direction": "in", "cap": null, "count": null}
      ],
      "return_type": null
    }
  ],
  "structs": []
}"#;

    let py = ea_compiler::bind_python::generate(json, "copy").unwrap();

    // dst should remain in the signature since it has no cap
    assert!(
        py.contains("dst: _np.ndarray"),
        "out param without cap should stay in signature:\n{py}"
    );
}

// === Rust binding generation with out params ===

#[test]
fn test_rust_out_param_auto_allocate() {
    let json = r#"{
  "library": "transform.so",
  "exports": [
    {
      "name": "transform",
      "args": [
        {"name": "data", "type": "*f32", "direction": "in", "cap": null, "count": null},
        {"name": "result", "type": "*mut f32", "direction": "out", "cap": "n", "count": null},
        {"name": "n", "type": "i32", "direction": "in", "cap": null, "count": null}
      ],
      "return_type": null
    }
  ],
  "structs": []
}"#;

    let rs = ea_compiler::bind_rust::generate(json, "transform").unwrap();

    // result should not be in safe wrapper params
    assert!(
        !rs.contains("result: &mut [f32]"),
        "auto-allocated out should not be in safe params:\n{rs}"
    );
    // Should allocate Vec
    assert!(
        rs.contains("let mut result: Vec<f32> = vec![Default::default(); n]"),
        "expected Vec allocation:\n{rs}"
    );
    // Return type should be Vec
    assert!(
        rs.contains("-> Vec<f32>"),
        "expected Vec return type:\n{rs}"
    );
}

// === C++ binding generation with out params ===

#[test]
fn test_cpp_out_param_auto_allocate() {
    let json = r#"{
  "library": "transform.so",
  "exports": [
    {
      "name": "transform",
      "args": [
        {"name": "data", "type": "*f32", "direction": "in", "cap": null, "count": null},
        {"name": "result", "type": "*mut f32", "direction": "out", "cap": "n", "count": null},
        {"name": "n", "type": "i32", "direction": "in", "cap": null, "count": null}
      ],
      "return_type": null
    }
  ],
  "structs": []
}"#;

    let hpp = ea_compiler::bind_cpp::generate(json, "transform").unwrap();

    // Should allocate vector
    assert!(
        hpp.contains("std::vector<float> result(n)"),
        "expected vector allocation:\n{hpp}"
    );
    // Return type should be vector
    assert!(
        hpp.contains("std::vector<float>"),
        "expected vector return type:\n{hpp}"
    );
    // Should return result
    assert!(
        hpp.contains("return result"),
        "expected return result:\n{hpp}"
    );
}

// === Backward compatibility ===

#[test]
fn test_backward_compat_no_direction_field() {
    // JSON without direction/cap/count fields (old format)
    let json = r#"{
  "library": "kernel.so",
  "exports": [
    {
      "name": "scale",
      "args": [{"name": "data", "type": "*mut f32"}, {"name": "len", "type": "i32"}, {"name": "alpha", "type": "f32"}],
      "return_type": null
    }
  ],
  "structs": []
}"#;

    // Should parse without error and behave same as before
    let py = ea_compiler::bind_python::generate(json, "kernel").unwrap();
    assert!(py.contains("def scale(data: _np.ndarray, alpha: float):"));

    let rs = ea_compiler::bind_rust::generate(json, "kernel").unwrap();
    assert!(rs.contains("data: &mut [f32]"));
}

// === Round-trip: source → parse → metadata → binding ===

#[test]
fn test_roundtrip_out_param_source_to_python() {
    let source = r#"
        export func double_it(data: *f32, out result: *mut f32 [cap: n], n: i32) {
            let mut i: i32 = 0
            while i < n {
                result[i] = data[i] * 2.0
                i = i + 1
            }
        }
    "#;

    let tokens = ea_compiler::tokenize(source).unwrap();
    let stmts = ea_compiler::parse(tokens).unwrap();
    ea_compiler::check_types(&stmts).unwrap();
    let json = ea_compiler::metadata::generate_json(&stmts, "double.so");

    let py = ea_compiler::bind_python::generate(&json, "double").unwrap();
    assert!(
        py.contains("result = _np.empty(n"),
        "expected auto-allocation in roundtrip:\n{py}"
    );
    assert!(
        py.contains("return result"),
        "expected return in roundtrip:\n{py}"
    );
    // data should still be in signature
    assert!(
        py.contains("data: _np.ndarray"),
        "data should be in sig:\n{py}"
    );
    // n should be collapsed (it follows pointer args)
    assert!(!py.contains("n: int"), "n should be collapsed:\n{py}");
}

// === Kernel with out param ===

#[test]
fn test_kernel_with_out_annotation() {
    let source = r#"
        export kernel double_it(data: *f32, out result: *mut f32 [cap: n])
            over i in n step 1
        {
            result[i] = data[i] * 2.0
        }
    "#;
    let tokens = ea_compiler::tokenize(source).unwrap();
    let stmts = ea_compiler::parse(tokens).unwrap();
    let stmts = ea_compiler::desugar(stmts).unwrap();

    // After desugar, out annotation should be preserved on result param
    if let ea_compiler::ast::Stmt::Function { params, .. } = &stmts[0] {
        let result_param = params.iter().find(|p| p.name == "result").unwrap();
        assert!(result_param.output, "result should be marked as output");
        assert_eq!(result_param.cap.as_deref(), Some("n"));
    } else {
        panic!("expected desugared Function");
    }
}
