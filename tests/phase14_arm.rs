#[cfg(feature = "llvm")]
mod tests {
    use ea_compiler::{CompileOptions, OutputMode};
    use tempfile::TempDir;

    fn arm_opts() -> CompileOptions {
        CompileOptions {
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
            ..CompileOptions::default()
        }
    }

    fn try_compile(
        source: &str,
        opts: &CompileOptions,
    ) -> Result<(), ea_compiler::error::CompileError> {
        let dir = TempDir::new().unwrap();
        let obj_path = dir.path().join("test.o");
        ea_compiler::compile_with_options(source, &obj_path, OutputMode::ObjectFile, opts)
    }

    // === ARM: >128-bit vectors should error ===

    #[test]
    fn test_arm_rejects_f32x8() {
        let err =
            try_compile("export func f(v: f32x8) -> f32x8 { return v }", &arm_opts()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("AVX2"), "expected AVX2 mention, got: {msg}");
        assert!(msg.contains("f32x4"), "expected f32x4 hint, got: {msg}");
    }

    #[test]
    fn test_arm_rejects_f32x16() {
        let err = try_compile(
            "export func f(v: f32x16) -> f32x16 { return v }",
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("AVX-512"),
            "expected AVX-512 mention, got: {msg}"
        );
    }

    #[test]
    fn test_arm_rejects_i32x8() {
        let err =
            try_compile("export func f(v: i32x8) -> i32x8 { return v }", &arm_opts()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("AVX2"), "expected AVX2 mention, got: {msg}");
    }

    #[test]
    fn test_arm_rejects_i8x32() {
        let err =
            try_compile("export func f(v: i8x32) -> i8x32 { return v }", &arm_opts()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("AVX2"), "expected AVX2 mention, got: {msg}");
    }

    #[test]
    fn test_arm_rejects_i16x16() {
        let err = try_compile(
            "export func f(v: i16x16) -> i16x16 { return v }",
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("AVX2"), "expected AVX2 mention, got: {msg}");
    }

    #[test]
    fn test_arm_rejects_f32x8_local() {
        let err = try_compile(
            r#"
            export func f() {
                let v: f32x8 = splat(1.0)
            }
            "#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("AVX2"), "expected AVX2 mention, got: {msg}");
    }

    // === ARM: x86-only intrinsics should error ===

    #[test]
    fn test_arm_rejects_maddubs_i16() {
        let err = try_compile(
            r#"
            export func f(a: u8x16, b: i8x16) -> i16x8 {
                return maddubs_i16(a, b)
            }
            "#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("x86"), "expected x86 mention, got: {msg}");
    }

    #[test]
    fn test_arm_rejects_maddubs_i32() {
        let err = try_compile(
            r#"
            export func f(a: u8x16, b: i8x16) -> i32x4 {
                return maddubs_i32(a, b)
            }
            "#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("x86"), "expected x86 mention, got: {msg}");
    }

    // === ARM: gather/scatter should error ===

    #[test]
    fn test_arm_rejects_gather() {
        let err = try_compile(
            r#"
            export func f(base: *f32, idx: i32x4) -> f32x4 {
                return gather(base, idx)
            }
            "#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("NEON") || msg.contains("ARM") || msg.contains("scalar"),
            "expected ARM/NEON mention, got: {msg}"
        );
    }

    #[test]
    fn test_arm_rejects_scatter() {
        let err = try_compile(
            r#"
            export func f(base: *mut f32, idx: i32x4, vals: f32x4) {
                scatter(base, idx, vals)
            }
            "#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("NEON") || msg.contains("ARM") || msg.contains("scalar"),
            "expected ARM/NEON mention, got: {msg}"
        );
    }

    // === ARM: 128-bit vectors should compile OK ===

    #[test]
    fn test_arm_accepts_f32x4() {
        try_compile("export func f(v: f32x4) -> f32x4 { return v }", &arm_opts())
            .expect("f32x4 should compile on ARM");
    }

    #[test]
    fn test_arm_accepts_i32x4() {
        try_compile("export func f(v: i32x4) -> i32x4 { return v }", &arm_opts())
            .expect("i32x4 should compile on ARM");
    }

    #[test]
    fn test_arm_accepts_i8x16() {
        try_compile("export func f(v: i8x16) -> i8x16 { return v }", &arm_opts())
            .expect("i8x16 should compile on ARM");
    }

    #[test]
    fn test_arm_accepts_i16x8() {
        try_compile("export func f(v: i16x8) -> i16x8 { return v }", &arm_opts())
            .expect("i16x8 should compile on ARM");
    }

    // === x86: no regression — wider vectors still work ===

    #[test]
    fn test_x86_still_accepts_f32x8() {
        try_compile(
            "export func f(v: f32x8) -> f32x8 { return v }",
            &CompileOptions::default(),
        )
        .expect("f32x8 should still compile on x86");
    }

    #[test]
    fn test_x86_still_accepts_i32x8() {
        try_compile(
            "export func f(v: i32x8) -> i32x8 { return v }",
            &CompileOptions::default(),
        )
        .expect("i32x8 should still compile on x86");
    }
}
