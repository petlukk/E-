#[cfg(feature = "llvm")]
use inkwell::targets::{
    CodeModel, FileType, RelocMode, Target, TargetMachine,
};

#[cfg(feature = "llvm")]
use inkwell::module::Module;

#[cfg(feature = "llvm")]
use crate::error::CompileError;

#[cfg(feature = "llvm")]
pub fn create_target_machine() -> crate::error::Result<TargetMachine> {
    // Note: LLVM native target initialization is now handled in lib.rs via std::sync::Once
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple)
        .map_err(|e| CompileError::codegen_error(format!("failed to get target: {e}")))?;

    let cpu = TargetMachine::get_host_cpu_name();
    let cpu_str = cpu.to_string();
    let features = TargetMachine::get_host_cpu_features();
    let features_str = features.to_string();

    target
        .create_target_machine(
            &triple,
            &cpu_str,
            &features_str,
            inkwell::OptimizationLevel::Default,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or_else(|| CompileError::codegen_error("failed to create target machine"))
}

#[cfg(feature = "llvm")]
pub fn write_object_file(module: &Module, path: &std::path::Path) -> crate::error::Result<()> {
    let machine = create_target_machine()?;
    machine
        .write_to_file(module, FileType::Object, path)
        .map_err(|e| CompileError::codegen_error(format!("failed to write object file: {e}")))
}
