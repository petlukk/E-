#[cfg(feature = "llvm")]
use inkwell::targets::{CodeModel, FileType, RelocMode, Target, TargetMachine, TargetTriple};

#[cfg(feature = "llvm")]
use inkwell::passes::PassBuilderOptions;

#[cfg(feature = "llvm")]
use inkwell::module::Module;

#[cfg(feature = "llvm")]
use crate::error::CompileError;

#[cfg(feature = "llvm")]
use crate::CompileOptions;

#[cfg(feature = "llvm")]
fn opt_level_to_inkwell(level: u8) -> inkwell::OptimizationLevel {
    match level {
        0 => inkwell::OptimizationLevel::None,
        1 => inkwell::OptimizationLevel::Less,
        2 => inkwell::OptimizationLevel::Default,
        _ => inkwell::OptimizationLevel::Aggressive,
    }
}

#[cfg(feature = "llvm")]
pub fn create_target_machine(opts: &CompileOptions) -> crate::error::Result<TargetMachine> {
    let triple = if let Some(ref t) = opts.target_triple {
        TargetTriple::create(t)
    } else {
        TargetMachine::get_default_triple()
    };
    let target = Target::from_triple(&triple)
        .map_err(|e| CompileError::codegen_error(format!("failed to get target: {e}")))?;

    let cross_compiling = opts.target_triple.is_some();

    let (cpu_str, base_features) = if let Some(ref cpu) = opts.target_cpu {
        (cpu.clone(), String::new())
    } else if cross_compiling {
        ("generic".to_string(), String::new())
    } else {
        let cpu = TargetMachine::get_host_cpu_name();
        let features = TargetMachine::get_host_cpu_features();
        (cpu.to_string(), features.to_string())
    };
    let features_str = if opts.extra_features.is_empty() {
        base_features
    } else if base_features.is_empty() {
        opts.extra_features.clone()
    } else {
        format!("{},{}", base_features, opts.extra_features)
    };

    target
        .create_target_machine(
            &triple,
            &cpu_str,
            &features_str,
            opt_level_to_inkwell(opts.opt_level),
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or_else(|| CompileError::codegen_error("failed to create target machine"))
}

#[cfg(feature = "llvm")]
pub fn write_object_file(
    module: &Module,
    path: &std::path::Path,
    opts: &CompileOptions,
) -> crate::error::Result<()> {
    let machine = create_target_machine(opts)?;

    if opts.opt_level > 0 {
        optimize_module(module, &machine, opts.opt_level)?;
    }

    machine
        .write_to_file(module, FileType::Object, path)
        .map_err(|e| CompileError::codegen_error(format!("failed to write object file: {e}")))
}

#[cfg(feature = "llvm")]
pub fn write_asm_file(
    module: &Module,
    path: &std::path::Path,
    opts: &CompileOptions,
) -> crate::error::Result<()> {
    let machine = create_target_machine(opts)?;

    if opts.opt_level > 0 {
        optimize_module(module, &machine, opts.opt_level)?;
    }

    machine
        .write_to_file(module, FileType::Assembly, path)
        .map_err(|e| CompileError::codegen_error(format!("failed to write assembly: {e}")))
}

#[cfg(feature = "llvm")]
pub fn optimize_module(
    module: &Module,
    machine: &TargetMachine,
    opt_level: u8,
) -> crate::error::Result<()> {
    let passes = format!("default<O{}>", opt_level.min(3));
    let opts = PassBuilderOptions::create();
    module
        .run_passes(&passes, machine, opts)
        .map_err(|e| CompileError::codegen_error(format!("pass pipeline failed: {e}")))?;
    Ok(())
}
