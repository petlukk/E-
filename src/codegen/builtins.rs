use inkwell::values::{BasicMetadataValueEnum, BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn compile_println(
        &mut self,
        arg: &Expr,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let printf = *self
            .functions
            .get("printf")
            .ok_or_else(|| CompileError::codegen_error("printf not declared"))?;

        let unsigned = self.arg_is_unsigned(arg);
        let val = self.compile_expr(arg, function)?;

        match val {
            BasicValueEnum::IntValue(iv) => {
                let bit_width = iv.get_type().get_bit_width();
                let (print_iv, fmt_str) = if bit_width < 32 {
                    let extended = if unsigned {
                        self.builder
                            .build_int_z_extend(iv, self.context.i32_type(), "zext")
                    } else {
                        self.builder
                            .build_int_s_extend(iv, self.context.i32_type(), "sext")
                    }
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                    (extended, "%d\n")
                } else if bit_width == 64 {
                    (iv, if unsigned { "%lu\n" } else { "%ld\n" })
                } else {
                    (iv, if unsigned { "%u\n" } else { "%d\n" })
                };
                let fmt = self
                    .builder
                    .build_global_string_ptr(fmt_str, "fmt")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                self.builder
                    .build_call(
                        printf,
                        &[fmt.as_pointer_value().into(), print_iv.into()],
                        "printf_call",
                    )
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            }
            BasicValueEnum::FloatValue(fv) => {
                let fmt = self
                    .builder
                    .build_global_string_ptr("%g\n", "fmt_float")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                let print_val = if fv.get_type() == self.context.f32_type() {
                    let extended = self
                        .builder
                        .build_float_ext(fv, self.context.f64_type(), "fpext")
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                    BasicMetadataValueEnum::from(extended)
                } else {
                    BasicMetadataValueEnum::from(fv)
                };
                self.builder
                    .build_call(
                        printf,
                        &[fmt.as_pointer_value().into(), print_val],
                        "printf_call",
                    )
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            }
            BasicValueEnum::PointerValue(_) => {
                let fmt = self
                    .builder
                    .build_global_string_ptr("%s\n", "fmt_str")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                self.builder
                    .build_call(
                        printf,
                        &[fmt.as_pointer_value().into(), val.into()],
                        "printf_call",
                    )
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            }
            _ => {
                return Err(CompileError::codegen_error(
                    "unsupported println argument type",
                ));
            }
        }

        Ok(BasicValueEnum::IntValue(
            self.context.i32_type().const_int(0, false),
        ))
    }
}
