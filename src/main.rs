use std::process;

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
            println!("ea 0.1.0");
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
    let mut emit_llvm = false;
    let mut emit_ast = false;
    let mut emit_tokens = false;

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
            "--emit-llvm" => emit_llvm = true,
            "--emit-ast" => emit_ast = true,
            "--emit-tokens" => emit_tokens = true,
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
                eprintln!("{e}");
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
                eprintln!("{e}");
                process::exit(1);
            }
        }
        return;
    }

    #[cfg(feature = "llvm")]
    {
        use ea_compiler::OutputMode;
        use std::path::PathBuf;

        let stem = std::path::Path::new(input_file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");

        if emit_llvm {
            let ir_path = PathBuf::from(format!("{stem}.ll"));
            match ea_compiler::compile(&source, &ir_path, OutputMode::LlvmIr) {
                Ok(()) => {
                    let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
                    print!("{ir}");
                }
                Err(e) => {
                    eprintln!("{e}");
                    process::exit(1);
                }
            }
            return;
        }

        let (output_path, mode) = if let Some(exe) = output_exe {
            let obj_path = PathBuf::from(format!("{stem}.o"));
            (obj_path, OutputMode::Executable(exe))
        } else {
            let obj_path = PathBuf::from(format!("{stem}.o"));
            (obj_path, OutputMode::ObjectFile)
        };

        match ea_compiler::compile(&source, &output_path, mode) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("{e}");
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
    eprintln!("  -o <name>       Compile and link to executable");
    eprintln!("  --emit-llvm     Print LLVM IR");
    eprintln!("  --emit-ast      Print parsed AST");
    eprintln!("  --emit-tokens   Print lexer tokens");
    eprintln!("  --help, -h      Show this message");
    eprintln!("  --version, -V   Show version");
}
