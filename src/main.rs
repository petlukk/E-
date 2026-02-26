use ea_compiler::error::{format_with_source, CompileError};
use std::process;

fn print_error(e: &CompileError, filename: &str, source: &str) {
    eprintln!("{}", format_with_source(e, filename, source));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        print_usage();
        process::exit(1);
    }

    match args[0].as_str() {
        "--help" | "-h" => {
            print_usage();
            return;
        }
        "--version" | "-V" => {
            println!("ea {}", env!("CARGO_PKG_VERSION"));
            return;
        }
        _ => {}
    }

    let input_file = &args[0];
    let source = match std::fs::read_to_string(input_file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read '{input_file}': {e}");
            process::exit(1);
        }
    };

    let mut output_exe: Option<String> = None;
    let mut lib_mode = false;
    let mut emit_llvm = false;
    let mut emit_asm = false;
    let mut emit_header = false;
    let mut emit_ast = false;
    let mut emit_tokens = false;
    let mut opt_level: u8 = 3;
    let mut target_cpu: Option<String> = None;
    let mut extra_features = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: -o requires an argument");
                    process::exit(1);
                }
                output_exe = Some(args[i].clone());
            }
            "--lib" => lib_mode = true,
            "--emit-llvm" => emit_llvm = true,
            "--emit-asm" => emit_asm = true,
            "--header" => emit_header = true,
            "--emit-ast" => emit_ast = true,
            "--emit-tokens" => emit_tokens = true,
            s if s.starts_with("--opt-level=") => {
                let val = &s["--opt-level=".len()..];
                match val.parse::<u8>() {
                    Ok(v) if v <= 3 => opt_level = v,
                    _ => {
                        eprintln!("error: --opt-level must be 0, 1, 2, or 3");
                        process::exit(1);
                    }
                }
            }
            s if s.starts_with("--target=") => {
                let val = &s["--target=".len()..];
                if val == "native" {
                    target_cpu = None;
                } else {
                    target_cpu = Some(val.to_string());
                }
            }
            "--avx512" => {
                extra_features = "+avx512f,+avx512vl,+avx512bw".to_string();
            }
            other => {
                eprintln!("error: unknown option '{other}'");
                process::exit(1);
            }
        }
        i += 1;
    }

    if emit_tokens {
        match ea_compiler::tokenize(&source) {
            Ok(tokens) => {
                for t in &tokens {
                    println!("{t}");
                }
            }
            Err(e) => {
                print_error(&e, input_file, &source);
                process::exit(1);
            }
        }
        return;
    }

    if emit_ast {
        match ea_compiler::tokenize(&source).and_then(ea_compiler::parse) {
            Ok(stmts) => {
                for s in &stmts {
                    println!("{s}");
                }
            }
            Err(e) => {
                print_error(&e, input_file, &source);
                process::exit(1);
            }
        }
        return;
    }

    #[cfg(feature = "llvm")]
    {
        use ea_compiler::{CompileOptions, OutputMode};
        use std::path::PathBuf;

        let opts = CompileOptions {
            opt_level,
            target_cpu,
            extra_features,
        };

        let stem = std::path::Path::new(input_file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");

        if emit_llvm {
            let ir_path = PathBuf::from(format!("{stem}.ll"));
            match ea_compiler::compile_with_options(&source, &ir_path, OutputMode::LlvmIr, &opts) {
                Ok(()) => {
                    let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
                    print!("{ir}");
                }
                Err(e) => {
                    print_error(&e, input_file, &source);
                    process::exit(1);
                }
            }
            return;
        }

        if emit_asm {
            let asm_path = PathBuf::from(format!("{stem}.s"));
            match ea_compiler::compile_with_options(&source, &asm_path, OutputMode::Asm, &opts) {
                Ok(()) => {
                    eprintln!("wrote {}", asm_path.display());
                }
                Err(e) => {
                    print_error(&e, input_file, &source);
                    process::exit(1);
                }
            }
            return;
        }

        if emit_header {
            let tokens = match ea_compiler::tokenize(&source) {
                Ok(t) => t,
                Err(e) => {
                    print_error(&e, input_file, &source);
                    process::exit(1);
                }
            };
            let stmts = match ea_compiler::parse(tokens) {
                Ok(s) => s,
                Err(e) => {
                    print_error(&e, input_file, &source);
                    process::exit(1);
                }
            };
            if let Err(e) = ea_compiler::check_types(&stmts) {
                print_error(&e, input_file, &source);
                process::exit(1);
            }
            let header = ea_compiler::header::generate(&stmts, stem);
            let header_path = PathBuf::from(format!("{stem}.h"));
            if let Err(e) = std::fs::write(&header_path, header) {
                eprintln!("error: cannot write '{}': {e}", header_path.display());
                process::exit(1);
            }
            eprintln!("wrote {}", header_path.display());
        }

        let (output_path, mode) = if lib_mode {
            let ext = if cfg!(target_os = "windows") {
                "dll"
            } else {
                "so"
            };
            let lib_name = if let Some(name) = output_exe {
                name
            } else {
                format!("{stem}.{ext}")
            };
            let obj_path = PathBuf::from(format!("{stem}.o"));
            (obj_path, OutputMode::SharedLib(lib_name))
        } else if let Some(exe) = output_exe {
            let obj_path = PathBuf::from(format!("{stem}.o"));
            (obj_path, OutputMode::Executable(exe))
        } else {
            let obj_path = PathBuf::from(format!("{stem}.o"));
            (obj_path, OutputMode::ObjectFile)
        };

        let mode_desc = match &mode {
            OutputMode::ObjectFile => "object",
            OutputMode::Executable(_) => "executable",
            OutputMode::SharedLib(_) => "shared library",
            OutputMode::LlvmIr => "llvm-ir",
            OutputMode::Asm => "assembly",
        };
        let output_display = match &mode {
            OutputMode::Executable(ref name) | OutputMode::SharedLib(ref name) => {
                name.clone()
            }
            _ => output_path.display().to_string(),
        };

        match ea_compiler::compile_with_options(&source, &output_path, mode, &opts) {
            Ok(()) => {
                let stmts =
                    ea_compiler::parse(ea_compiler::tokenize(&source).unwrap()).unwrap();
                let exports = ea_compiler::ast::exported_function_names(&stmts);
                if exports.is_empty() {
                    eprintln!(
                        "compiled {} -> {} ({})",
                        input_file, output_display, mode_desc
                    );
                } else {
                    eprintln!(
                        "compiled {} -> {} ({}, {} exported: {})",
                        input_file,
                        output_display,
                        mode_desc,
                        exports.len(),
                        exports.join(", ")
                    );
                }
            }
            Err(e) => {
                print_error(&e, input_file, &source);
                process::exit(1);
            }
        }
    }

    #[cfg(not(feature = "llvm"))]
    {
        eprintln!("error: compilation requires the 'llvm' feature");
        process::exit(1);
    }
}

fn print_usage() {
    eprintln!("Usage: ea <file.ea> [options]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  -o <name>          Compile and link to executable");
    eprintln!("  --lib              Produce shared library (.so/.dll)");
    eprintln!("  --opt-level=N      Optimization level 0-3 (default: 3)");
    eprintln!(
        "  --target=CPU       Target CPU (default: native)
  --avx512           Enable AVX-512 (f32x16) — requires AVX-512 capable CPU"
    );
    eprintln!("  --emit-llvm        Print LLVM IR");
    eprintln!("  --emit-asm         Emit assembly (.s file)");
    eprintln!("  --header           Generate C header file (.h)");
    eprintln!("  --emit-ast         Print parsed AST");
    eprintln!("  --emit-tokens      Print lexer tokens");
    eprintln!("  --help, -h         Show this message");
    eprintln!("  --version, -V      Show version");
}
