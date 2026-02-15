pub mod ast;
pub mod error;
pub mod lexer;
pub mod parser;
pub mod typeck;

#[cfg(feature = "llvm")]
pub mod codegen;
#[cfg(feature = "llvm")]
pub mod target;

use ast::Stmt;
use lexer::{Lexer, Token};
use parser::Parser;
use typeck::TypeChecker;

pub fn tokenize(source: &str) -> error::Result<Vec<Token>> {
    Lexer::new(source).tokenize()
}

pub fn parse(tokens: Vec<Token>) -> error::Result<Vec<Stmt>> {
    Parser::new(tokens).parse_program()
}

pub fn check_types(stmts: &[Stmt]) -> error::Result<()> {
    TypeChecker::new().check_program(stmts)
}

#[cfg(feature = "llvm")]
pub enum OutputMode {
    ObjectFile,
    Executable(String),
    SharedLib(String),
    LlvmIr,
}

#[cfg(feature = "llvm")]
#[derive(Clone, Debug)]
pub struct CompileOptions {
    pub opt_level: u8,
    pub target_cpu: Option<String>,
}

#[cfg(feature = "llvm")]
impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            opt_level: 3,
            target_cpu: None, // native
        }
    }
}

#[cfg(feature = "llvm")]
static INIT_LLVM: std::sync::Once = std::sync::Once::new();

#[cfg(feature = "llvm")]
fn init_llvm() {
    INIT_LLVM.call_once(|| {
        use inkwell::targets::{InitializationConfig, Target};
        Target::initialize_native(&InitializationConfig::default())
            .expect("Failed to initialize LLVM native target");
    });
}

#[cfg(feature = "llvm")]
pub fn compile(source: &str, output_path: &std::path::Path, mode: OutputMode) -> error::Result<()> {
    compile_with_options(source, output_path, mode, &CompileOptions::default())
}

pub fn compile_with_options(
    source: &str,
    output_path: &std::path::Path,
    mode: OutputMode,
    opts: &CompileOptions,
) -> error::Result<()> {
    init_llvm();

    let tokens = tokenize(source)?;
    let stmts = parse(tokens)?;
    check_types(&stmts)?;

    let context = inkwell::context::Context::create();
    let mut gen = codegen::CodeGenerator::new(&context, "ea_module");
    gen.compile_program(&stmts)?;

    match mode {
        OutputMode::ObjectFile => {
            target::write_object_file(gen.module(), output_path, opts)?;
        }
        OutputMode::Executable(ref exe_name) => {
            let tmp_dir = std::env::temp_dir().join("ea_build");
            std::fs::create_dir_all(&tmp_dir).map_err(|e| {
                error::CompileError::codegen_error(format!("failed to create temp dir: {e}"))
            })?;
            let obj_path = tmp_dir.join("temp.o");
            target::write_object_file(gen.module(), &obj_path, opts)?;

            let status = std::process::Command::new("cc")
                .arg(&obj_path)
                .arg("-o")
                .arg(exe_name)
                .arg("-lm")
                .status()
                .map_err(|e| {
                    error::CompileError::codegen_error(format!("failed to invoke linker: {e}"))
                })?;

            let _ = std::fs::remove_dir_all(&tmp_dir);

            if !status.success() {
                return Err(error::CompileError::codegen_error("linker failed"));
            }
        }
        OutputMode::SharedLib(ref lib_name) => {
            target::write_object_file(gen.module(), output_path, opts)?;

            let status = std::process::Command::new("cc")
                .arg("-shared")
                .arg(output_path)
                .arg("-o")
                .arg(lib_name)
                .arg("-lm")
                .status()
                .map_err(|e| {
                    error::CompileError::codegen_error(format!("failed to invoke linker: {e}"))
                })?;

            let _ = std::fs::remove_file(output_path);

            if !status.success() {
                return Err(error::CompileError::codegen_error(
                    "shared library linking failed",
                ));
            }
        }
        OutputMode::LlvmIr => {
            let ir = gen.print_ir();
            std::fs::write(output_path, ir).map_err(|e| {
                error::CompileError::codegen_error(format!("failed to write IR: {e}"))
            })?;
        }
    }

    Ok(())
}

#[cfg(feature = "llvm")]
pub fn compile_to_ir(source: &str) -> error::Result<String> {
    init_llvm(); // Thread-safe one-time initialization
    
    let tokens = tokenize(source)?;
    let stmts = parse(tokens)?;
    check_types(&stmts)?;

    let context = inkwell::context::Context::create();
    let mut gen = codegen::CodeGenerator::new(&context, "ea_module");
    gen.compile_program(&stmts)?;

    Ok(gen.print_ir())
}
