#[cfg(feature = "llvm")]
use inkwell::targets::{
    CodeModel, FileType, RelocMode, Target, TargetMachine,
};

#[cfg(feature = "llvm")]
use inkwell::passes::PassManager;

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
            inkwell::OptimizationLevel::Aggressive,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or_else(|| CompileError::codegen_error("failed to create target machine"))
}

#[cfg(feature = "llvm")]
pub fn write_object_file(module: &Module, path: &std::path::Path) -> crate::error::Result<()> {
    // Run O3 optimization passes before writing object file
    optimize_module(module)?;
    
    let machine = create_target_machine()?;
    machine
        .write_to_file(module, FileType::Object, path)
        .map_err(|e| CompileError::codegen_error(format!("failed to write object file: {e}")))
}

#[cfg(feature = "llvm")]
fn optimize_module(module: &Module) -> crate::error::Result<()> {
    // Curated pipeline for SIMD kernel IR.
    // Ea already emits vector IR — no auto-vectorizers needed.
    // SLP/loop vectorizers hurt explicit SIMD without fast-math flags.
    let fpm = PassManager::create(module);

    // Scalar cleanup
    fpm.add_promote_memory_to_register_pass();  // mem2reg — critical
    fpm.add_instruction_combining_pass();       // instcombine
    fpm.add_reassociate_pass();                 // reorder expressions
    fpm.add_gvn_pass();                         // GVN
    fpm.add_cfg_simplification_pass();          // simplifycfg
    fpm.add_early_cse_pass();                   // common subexpression elimination

    // Loop optimization (no loop_rotate — it breaks fcmp+select vectorization)
    fpm.add_licm_pass();                        // hoist invariants
    fpm.add_ind_var_simplify_pass();            // simplify induction variables
    fpm.add_loop_unroll_pass();                 // unroll loops

    // Cleanup
    fpm.add_instruction_combining_pass();       // instcombine again
    fpm.add_dead_store_elimination_pass();      // dead stores
    fpm.add_aggressive_dce_pass();              // dead code
    fpm.add_cfg_simplification_pass();          // final simplification


    fpm.initialize();
    for function in module.get_functions() {
        fpm.run_on(&function);
    }
    fpm.finalize();

    Ok(())
}
