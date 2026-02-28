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
        "bind" => {
            handle_bind(&args[1..]);
            return;
        }
        "inspect" => {
            handle_inspect(&args[1..]);
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
    let mut target_triple: Option<String> = None;

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
            s if s.starts_with("--target-triple=") => {
                target_triple = Some(s["--target-triple=".len()..].to_string());
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
            target_triple,
        };

        if opts.is_arm() && opts.extra_features.contains("avx512") {
            eprintln!("error: --avx512 is incompatible with ARM target");
            process::exit(1);
        }

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
            let stmts = match ea_compiler::desugar(stmts) {
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
            OutputMode::Executable(ref name) | OutputMode::SharedLib(ref name) => name.clone(),
            _ => output_path.display().to_string(),
        };

        match ea_compiler::compile_with_options(&source, &output_path, mode, &opts) {
            Ok(()) => {
                let stmts = ea_compiler::desugar(
                    ea_compiler::parse(ea_compiler::tokenize(&source).unwrap()).unwrap(),
                )
                .unwrap();
                let exports = ea_compiler::ast::exported_function_names(&stmts);
                if exports.is_empty() {
                    eprintln!("compiled {input_file} -> {output_display} ({mode_desc})");
                } else {
                    let count = exports.len();
                    let names = exports.join(", ");
                    eprintln!(
                        "compiled {input_file} -> {output_display} ({mode_desc}, {count} exported: {names})"
                    );
                }
                if lib_mode {
                    let lib_display_name = output_display.clone();
                    let json = ea_compiler::metadata::generate_json(&stmts, &lib_display_name);
                    let json_path = format!("{input_file}.json");
                    if let Err(e) = std::fs::write(&json_path, &json) {
                        eprintln!("warning: could not write {json_path}: {e}");
                    } else {
                        eprintln!("wrote {json_path}");
                    }
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

fn handle_inspect(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: ea inspect <file.ea> [--avx512] [--target=CPU] [--opt-level=N]");
        process::exit(1);
    }
    let input_file = &args[0];
    let source = match std::fs::read_to_string(input_file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read '{input_file}': {e}");
            process::exit(1);
        }
    };
    let mut opt_level: u8 = 3;
    let mut target_cpu: Option<String> = None;
    let mut extra_features = String::new();
    let mut target_triple: Option<String> = None;
    for arg in &args[1..] {
        if let Some(val) = arg.strip_prefix("--opt-level=") {
            match val.parse::<u8>() {
                Ok(v) if v <= 3 => opt_level = v,
                _ => {
                    eprintln!("error: --opt-level must be 0, 1, 2, or 3");
                    process::exit(1);
                }
            }
        } else if let Some(val) = arg.strip_prefix("--target=") {
            target_cpu = (val != "native").then(|| val.to_string());
        } else if let Some(val) = arg.strip_prefix("--target-triple=") {
            target_triple = Some(val.to_string());
        } else if arg == "--avx512" {
            extra_features = "+avx512f,+avx512vl,+avx512bw".to_string();
        } else {
            eprintln!("error: unknown inspect option '{arg}'");
            process::exit(1);
        }
    }
    #[cfg(feature = "llvm")]
    {
        let opts = ea_compiler::CompileOptions {
            opt_level,
            target_cpu,
            extra_features,
            target_triple,
        };
        match ea_compiler::inspect_source(&source, &opts) {
            Ok(report) => print!("{report}"),
            Err(e) => {
                print_error(&e, input_file, &source);
                process::exit(1);
            }
        }
    }
    #[cfg(not(feature = "llvm"))]
    {
        let _ = (opt_level, target_cpu, extra_features, target_triple);
        eprintln!("error: inspect requires the 'llvm' feature");
        process::exit(1);
    }
}

fn handle_bind(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: ea bind <file.ea> --python [--rust] [--pytorch] [--cmake] [--cpp]");
        process::exit(1);
    }
    let input_file = &args[0];
    let mut python = false;
    let mut rust = false;
    let mut pytorch = false;
    let mut cmake = false;
    let mut cpp = false;
    for arg in &args[1..] {
        match arg.as_str() {
            "--python" => python = true,
            "--rust" => rust = true,
            "--pytorch" => pytorch = true,
            "--cmake" => cmake = true,
            "--cpp" => cpp = true,
            other => {
                eprintln!("error: unknown bind option '{other}'");
                process::exit(1);
            }
        }
    }
    if !python && !rust && !pytorch && !cmake && !cpp {
        eprintln!(
            "error: ea bind requires at least one of --python, --rust, --pytorch, --cmake, --cpp"
        );
        process::exit(1);
    }

    let stem = std::path::Path::new(input_file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    // Read JSON metadata (needed by all generators except --cmake)
    let json_str = if python || rust || pytorch || cpp {
        let json_path = format!("{input_file}.json");
        match std::fs::read_to_string(&json_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: cannot read '{json_path}': {e}");
                eprintln!("hint: compile with --lib first to generate JSON metadata");
                process::exit(1);
            }
        }
    } else {
        String::new()
    };

    if python {
        let code = match ea_compiler::bind_python::generate(&json_str, stem) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: failed to generate Python bindings: {e}");
                process::exit(1);
            }
        };
        let path = format!("{stem}.py");
        write_or_exit(&path, &code);
    }

    if rust {
        let code = match ea_compiler::bind_rust::generate(&json_str, stem) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: failed to generate Rust bindings: {e}");
                process::exit(1);
            }
        };
        let path = format!("{stem}.rs");
        write_or_exit(&path, &code);
    }

    if pytorch {
        let code = match ea_compiler::bind_pytorch::generate(&json_str, stem) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: failed to generate PyTorch bindings: {e}");
                process::exit(1);
            }
        };
        let path = format!("{stem}_torch.py");
        write_or_exit(&path, &code);
    }

    if cmake {
        let (cmakelists, ea_cmake) = ea_compiler::bind_cmake::generate(stem);
        write_or_exit("CMakeLists.txt", &cmakelists);
        write_or_exit("EaCompiler.cmake", &ea_cmake);
    }

    if cpp {
        let code = match ea_compiler::bind_cpp::generate(&json_str, stem) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: failed to generate C++ bindings: {e}");
                process::exit(1);
            }
        };
        let path = format!("{stem}.hpp");
        write_or_exit(&path, &code);
    }
}

fn write_or_exit(path: &str, content: &str) {
    if let Err(e) = std::fs::write(path, content) {
        eprintln!("error: cannot write '{path}': {e}");
        process::exit(1);
    }
    eprintln!("wrote {path}");
}

fn print_usage() {
    eprint!(
        "\
Usage: ea <file.ea> [options]
       ea bind <file.ea> --python [--rust] [--pytorch] [--cmake] [--cpp]
       ea inspect <file.ea> [--avx512] [--target=CPU]

Options:
  -o <name>          Compile and link to executable
  --lib              Produce shared library (.so/.dll) + JSON metadata
  --opt-level=N      Optimization level 0-3 (default: 3)
  --target=CPU       Target CPU (default: native)
  --target-triple=T  Cross-compile target (e.g. aarch64-unknown-linux-gnu)
  --avx512           Enable AVX-512 (f32x16)
  --emit-llvm        Print LLVM IR
  --emit-asm         Emit assembly (.s file)
  --header           Generate C header (.h)
  --emit-ast/--emit-tokens  Print AST or lexer tokens
  --help, -h / --version, -V

Subcommands:
  inspect <file.ea>        Analyze kernel: instruction mix, loops, registers
  bind <file.ea> --python  Generate Python/NumPy bindings
  bind <file.ea> --rust    Generate Rust FFI + safe wrappers
  bind <file.ea> --pytorch Generate PyTorch autograd wrappers
  bind <file.ea> --cmake   Generate CMakeLists.txt + EaCompiler.cmake
  bind <file.ea> --cpp     Generate C++ header with std::span
"
    );
}
