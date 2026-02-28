#[cfg(feature = "llvm")]
use std::collections::{HashMap, HashSet};
#[cfg(feature = "llvm")]
use std::fmt;

#[cfg(feature = "llvm")]
use either::Either::{Left, Right};
#[cfg(feature = "llvm")]
use inkwell::module::Module;
#[cfg(feature = "llvm")]
use inkwell::targets::TargetMachine;
#[cfg(feature = "llvm")]
use inkwell::types::AnyTypeEnum;
#[cfg(feature = "llvm")]
use inkwell::values::InstructionOpcode;

#[cfg(feature = "llvm")]
use crate::error::CompileError;

#[cfg(feature = "llvm")]
#[derive(Debug)]
pub struct FunctionReport {
    pub name: String,
    pub exported: bool,
    pub vector_instructions: u32,
    pub scalar_instructions: u32,
    pub vector_width: Option<String>,
    pub loops: u32,
    pub registers: Vec<String>,
}

#[cfg(feature = "llvm")]
#[derive(Debug)]
pub struct InspectReport {
    pub functions: Vec<FunctionReport>,
}

#[cfg(feature = "llvm")]
impl fmt::Display for InspectReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, func) in self.functions.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            let export_tag = if func.exported { " (exported)" } else { "" };
            writeln!(f, "=== {}{} ===", func.name, export_tag)?;
            writeln!(f, "  vector instructions:  {}", func.vector_instructions)?;
            writeln!(f, "  scalar instructions:  {}", func.scalar_instructions)?;
            if let Some(ref width) = func.vector_width {
                writeln!(f, "  vector width:         {width}")?;
            }
            let loop_detail = if func.loops == 0 {
                "0".to_string()
            } else if func.loops == 1 {
                "1".to_string()
            } else {
                let tail = func.loops - 1;
                format!("{} ({} main, {} tail)", func.loops, 1, tail)
            };
            writeln!(f, "  loops:                {loop_detail}")?;
            if !func.registers.is_empty() {
                let reg_list = func.registers.join(", ");
                writeln!(
                    f,
                    "  vector registers:     {} ({} used)",
                    reg_list,
                    func.registers.len()
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "llvm")]
pub fn analyze_module(
    module: &Module,
    machine: &TargetMachine,
) -> crate::error::Result<InspectReport> {
    let mut functions = Vec::new();

    // Get assembly text for register analysis
    let asm_text = get_asm_text(module, machine)?;
    let asm_registers = parse_asm_registers(&asm_text);

    let mut func = module.get_first_function();
    while let Some(function) = func {
        let name = function.get_name().to_str().unwrap_or("").to_string();

        // Skip LLVM intrinsics and external declarations
        if name.starts_with("llvm.") || name == "printf" || function.count_basic_blocks() == 0 {
            func = function.get_next_function();
            continue;
        }

        let exported = function.get_linkage() == inkwell::module::Linkage::External;

        let mut vector_instructions: u32 = 0;
        let mut scalar_instructions: u32 = 0;
        let mut max_vector_bits: u32 = 0;
        let mut vector_elem_type = String::new();
        let mut vector_width: u32 = 0;

        let loops = detect_loops(&function);

        let mut bb = function.get_first_basic_block();
        while let Some(block) = bb {
            let mut instr = block.get_first_instruction();
            while let Some(instruction) = instr {
                if is_vector_instruction(&instruction) {
                    vector_instructions += 1;
                    let ty = instruction.get_type();
                    if let AnyTypeEnum::VectorType(vt) = ty {
                        let size = vt.get_size();
                        let elem = vt.get_element_type();
                        let elem_bits = if elem.is_float_type() {
                            let ft = elem.into_float_type();
                            if ft == function.get_type().get_context().f32_type() {
                                32
                            } else {
                                64
                            }
                        } else if elem.is_int_type() {
                            elem.into_int_type().get_bit_width()
                        } else {
                            32
                        };
                        let total_bits = size * elem_bits;
                        if total_bits > max_vector_bits {
                            max_vector_bits = total_bits;
                            let et = if elem_bits == 32 && vt.get_element_type().is_float_type() {
                                "f32"
                            } else if elem_bits == 64 && vt.get_element_type().is_float_type() {
                                "f64"
                            } else {
                                match elem_bits {
                                    8 => "i8",
                                    16 => "i16",
                                    32 => "i32",
                                    64 => "i64",
                                    _ => "?",
                                }
                            };
                            vector_elem_type = et.to_string();
                            vector_width = size;
                        }
                    }
                } else {
                    let opcode = instruction.get_opcode();
                    if is_countable_scalar_opcode(opcode) {
                        scalar_instructions += 1;
                    }
                }
                instr = instruction.get_next_instruction();
            }
            bb = block.get_next_basic_block();
        }

        let width_str = if max_vector_bits > 0 {
            Some(format!(
                "{max_vector_bits}-bit ({vector_elem_type}x{vector_width})"
            ))
        } else {
            None
        };

        let regs = asm_registers.get(&name).cloned().unwrap_or_default();

        functions.push(FunctionReport {
            name,
            exported,
            vector_instructions,
            scalar_instructions,
            vector_width: width_str,
            loops,
            registers: regs,
        });

        func = function.get_next_function();
    }

    Ok(InspectReport { functions })
}

#[cfg(feature = "llvm")]
fn is_countable_scalar_opcode(opcode: InstructionOpcode) -> bool {
    matches!(
        opcode,
        InstructionOpcode::Add
            | InstructionOpcode::Sub
            | InstructionOpcode::Mul
            | InstructionOpcode::FAdd
            | InstructionOpcode::FSub
            | InstructionOpcode::FMul
            | InstructionOpcode::FDiv
            | InstructionOpcode::SDiv
            | InstructionOpcode::UDiv
            | InstructionOpcode::SRem
            | InstructionOpcode::URem
            | InstructionOpcode::Load
            | InstructionOpcode::Store
            | InstructionOpcode::ICmp
            | InstructionOpcode::FCmp
            | InstructionOpcode::Call
            | InstructionOpcode::GetElementPtr
            | InstructionOpcode::SExt
            | InstructionOpcode::ZExt
            | InstructionOpcode::Trunc
            | InstructionOpcode::FPToSI
            | InstructionOpcode::SIToFP
            | InstructionOpcode::BitCast
    )
}

#[cfg(feature = "llvm")]
fn is_vector_instruction(instr: &inkwell::values::InstructionValue) -> bool {
    // Check if the instruction produces a vector result
    let ty = instr.get_type();
    if matches!(ty, AnyTypeEnum::VectorType(_)) {
        return true;
    }
    // Check if any operand is a vector type via get_operands iterator
    for operand in instr.get_operands().flatten() {
        if let Left(val) = operand {
            if val.get_type().is_vector_type() {
                return true;
            }
        }
    }
    false
}

#[cfg(feature = "llvm")]
fn detect_loops(function: &inkwell::values::FunctionValue) -> u32 {
    let mut block_order: Vec<inkwell::basic_block::BasicBlock> = Vec::new();
    let mut bb = function.get_first_basic_block();
    while let Some(block) = bb {
        block_order.push(block);
        bb = block.get_next_basic_block();
    }

    let block_index: HashMap<_, _> = block_order
        .iter()
        .enumerate()
        .map(|(i, b)| (*b, i))
        .collect();

    let mut loops = 0u32;
    for (idx, block) in block_order.iter().enumerate() {
        if let Some(term) = block.get_terminator() {
            if term.get_opcode() == InstructionOpcode::Br {
                for operand in term.get_operands().flatten() {
                    if let Right(succ_bb) = operand {
                        if let Some(&succ_idx) = block_index.get(&succ_bb) {
                            if succ_idx <= idx {
                                loops += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    loops
}

#[cfg(feature = "llvm")]
fn get_asm_text(module: &Module, machine: &TargetMachine) -> crate::error::Result<String> {
    use inkwell::targets::FileType;
    let buf = machine
        .write_to_memory_buffer(module, FileType::Assembly)
        .map_err(|e| CompileError::codegen_error(format!("failed to emit assembly: {e}")))?;
    Ok(String::from_utf8_lossy(buf.as_slice()).to_string())
}

#[cfg(feature = "llvm")]
fn parse_asm_registers(asm: &str) -> HashMap<String, Vec<String>> {
    let mut result: HashMap<String, Vec<String>> = HashMap::new();
    let mut current_func = String::new();

    for line in asm.lines() {
        let trimmed = line.trim();
        // Detect function labels (e.g., "funcname:" at start of line)
        if !trimmed.is_empty()
            && !trimmed.starts_with('.')
            && !trimmed.starts_with('#')
            && trimmed.ends_with(':')
            && !trimmed.contains(' ')
        {
            current_func = trimmed.trim_end_matches(':').to_string();
            continue;
        }

        if current_func.is_empty() {
            continue;
        }

        let mut regs: HashSet<String> = HashSet::new();
        for word in trimmed.split(|c: char| !c.is_alphanumeric()) {
            let is_vec_reg =
                word.starts_with("ymm") || word.starts_with("zmm") || word.starts_with("xmm");
            if is_vec_reg {
                let prefix_len = 3;
                let rest = &word[prefix_len..];
                if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
                    regs.insert(word.to_string());
                }
            }
        }

        if !regs.is_empty() {
            let entry = result.entry(current_func.clone()).or_default();
            for r in regs {
                if !entry.contains(&r) {
                    entry.push(r);
                }
            }
        }
    }

    for regs in result.values_mut() {
        regs.sort();
    }

    result
}
