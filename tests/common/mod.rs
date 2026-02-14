use std::process::Command;
use tempfile::TempDir;

pub struct TestOutput {
    pub stdout: String,
    #[allow(dead_code)]
    pub stderr: String,
    #[allow(dead_code)]
    pub exit_code: i32,
}

pub fn compile_and_run(source: &str) -> TestOutput {
    let dir = TempDir::new().expect("failed to create temp dir");
    let obj_path = dir.path().join("test.o");
    let bin_path = dir.path().join("test_bin");

    ea_compiler::compile(source, &obj_path, ea_compiler::OutputMode::ObjectFile)
        .expect("compilation failed");

    let link_status = Command::new("cc")
        .args([
            obj_path.to_str().unwrap(),
            "-o",
            bin_path.to_str().unwrap(),
            "-lm",
        ])
        .status()
        .expect("failed to invoke linker");
    assert!(link_status.success(), "linking failed");

    let output = Command::new(&bin_path)
        .output()
        .expect("failed to execute binary");
    TestOutput {
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        exit_code: output.status.code().unwrap_or(-1),
    }
}

pub fn compile_and_link_with_c(ea_source: &str, c_source: &str) -> TestOutput {
    let dir = TempDir::new().expect("failed to create temp dir");
    let obj_path = dir.path().join("kernel.o");
    let c_path = dir.path().join("harness.c");
    let bin_path = dir.path().join("test_bin");

    ea_compiler::compile(ea_source, &obj_path, ea_compiler::OutputMode::ObjectFile)
        .expect("compilation failed");
    std::fs::write(&c_path, c_source).expect("failed to write C harness");

    let link_status = Command::new("cc")
        .args([
            c_path.to_str().unwrap(),
            obj_path.to_str().unwrap(),
            "-o",
            bin_path.to_str().unwrap(),
            "-lm",
        ])
        .status()
        .expect("failed to invoke linker");
    assert!(link_status.success(), "linking C harness failed");

    let output = Command::new(&bin_path)
        .output()
        .expect("failed to execute binary");
    TestOutput {
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        exit_code: output.status.code().unwrap_or(-1),
    }
}

pub fn assert_output(source: &str, expected: &str) {
    let result = compile_and_run(source);
    assert_eq!(result.stdout.trim(), expected);
}

#[allow(dead_code)]
pub fn assert_output_lines(source: &str, expected: &[&str]) {
    let result = compile_and_run(source);
    let lines: Vec<&str> = result.stdout.trim().lines().collect();
    assert_eq!(lines, expected);
}

pub fn assert_c_interop(ea_source: &str, c_source: &str, expected: &str) {
    let result = compile_and_link_with_c(ea_source, c_source);
    assert_eq!(result.stdout.trim(), expected);
}

#[allow(dead_code)]
pub fn assert_shared_lib_interop(ea_source: &str, c_source: &str, expected: &str) {
    let dir = TempDir::new().expect("failed to create temp dir");
    let so_path = dir.path().join("libkernel.so");
    let c_path = dir.path().join("harness.c");
    let bin_path = dir.path().join("test_bin");

    ea_compiler::compile(
        ea_source,
        &dir.path().join("kernel.o"),
        ea_compiler::OutputMode::SharedLib(so_path.to_str().unwrap().to_string()),
    )
    .expect("shared lib compilation failed");

    assert!(so_path.exists(), "shared library was not created");

    std::fs::write(&c_path, c_source).expect("failed to write C harness");

    let link_status = Command::new("cc")
        .args([
            c_path.to_str().unwrap(),
            "-o",
            bin_path.to_str().unwrap(),
            so_path.to_str().unwrap(),
            "-lm",
            &format!("-Wl,-rpath,{}", dir.path().to_str().unwrap()),
        ])
        .status()
        .expect("failed to invoke linker");
    assert!(link_status.success(), "linking shared lib harness failed");

    let output = Command::new(&bin_path)
        .output()
        .expect("failed to execute binary");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert_eq!(stdout.trim(), expected);
}

#[allow(dead_code)]
pub fn compile_to_ir(source: &str) -> String {
    ea_compiler::compile_to_ir(source).expect("compilation failed")
}
