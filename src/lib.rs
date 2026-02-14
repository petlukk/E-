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
pub fn compile(source: &str, output_path: &std::path::Path, mode: OutputMode) -> error::Result<()> {
    let tokens = tokenize(source)?;
    let stmts = parse(tokens)?;
    check_types(&stmts)?;

    let context = inkwell::context::Context::create();
    let mut gen = codegen::CodeGenerator::new(&context, "ea_module");
    gen.compile_program(&stmts)?;

    match mode {
        OutputMode::ObjectFile => {
            target::write_object_file(gen.module(), output_path)?;
        }
        OutputMode::Executable(ref exe_name) => {
            let tmp_dir = std::env::temp_dir().join("ea_build");
            std::fs::create_dir_all(&tmp_dir).map_err(|e| {
                error::CompileError::codegen_error(format!("failed to create temp dir: {e}"))
            })?;
            let obj_path = tmp_dir.join("temp.o");
            target::write_object_file(gen.module(), &obj_path)?;

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
            target::write_object_file(gen.module(), output_path)?;

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
    let tokens = tokenize(source)?;
    let stmts = parse(tokens)?;
    check_types(&stmts)?;

    let context = inkwell::context::Context::create();
    let mut gen = codegen::CodeGenerator::new(&context, "ea_module");
    gen.compile_program(&stmts)?;

    Ok(gen.print_ir())
}
