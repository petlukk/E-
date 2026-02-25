use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// sqrt(x) for scalar f32/f64 and float vectors.
    pub(super) fn compile_sqrt(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let val = self.compile_expr(&args[0], function)?;
        match val {
            BasicValueEnum::FloatValue(fv) => {
                let float_ty = fv.get_type();
                let intrinsic_name = if float_ty == self.context.f32_type() {
                    "llvm.sqrt.f32"
                } else {
                    "llvm.sqrt.f64"
                };
                let fn_type = float_ty.fn_type(&[float_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function(intrinsic_name)
                    .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));
                let result = self
                    .builder
                    .build_call(intrinsic, &[fv.into()], "sqrt")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .left()
                    .ok_or_else(|| CompileError::codegen_error("sqrt did not return a value"))?;
                Ok(result)
            }
            BasicValueEnum::VectorValue(vv) => {
                let vec_ty = vv.get_type();
                let intrinsic_name = self.llvm_vector_intrinsic_name("llvm.sqrt", vec_ty);
                let fn_type = vec_ty.fn_type(&[vec_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function(&intrinsic_name)
                    .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));
                let result = self
                    .builder
                    .build_call(intrinsic, &[vv.into()], "vsqrt")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .left()
                    .ok_or_else(|| CompileError::codegen_error("sqrt did not return a value"))?;
                Ok(result)
            }
            _ => Err(CompileError::codegen_error(
                "sqrt expects float or float vector",
            )),
        }
    }

    /// rsqrt(x) = 1.0 / sqrt(x). Accurate; LLVM may lower to vrsqrtps + refinement.
    pub(super) fn compile_rsqrt(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let sqrt_val = self.compile_sqrt(args, function)?;
        match sqrt_val {
            BasicValueEnum::FloatValue(fv) => {
                let one = fv.get_type().const_float(1.0);
                let result = self
                    .builder
                    .build_float_div(one, fv, "rsqrt")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(BasicValueEnum::FloatValue(result))
            }
            BasicValueEnum::VectorValue(vv) => {
                let vec_ty = vv.get_type();
                let elem_ty = vec_ty.get_element_type().into_float_type();
                let one_scalar = elem_ty.const_float(1.0);
                let one_vec =
                    self.build_splat(BasicValueEnum::FloatValue(one_scalar), vec_ty.get_size())?;
                let result = self
                    .builder
                    .build_float_div(one_vec, vv, "vrsqrt")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(BasicValueEnum::VectorValue(result))
            }
            _ => Err(CompileError::codegen_error(
                "rsqrt expects float or float vector",
            )),
        }
    }

    /// Type conversion intrinsics: to_f32, to_f64, to_i32, to_i64.
    pub(super) fn compile_conversion(
        &mut self,
        name: &str,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let is_unsigned_src = self.arg_is_unsigned(&args[0]);
        let val = self.compile_expr(&args[0], function)?;

        match name {
            "to_f32" => {
                let target = self.context.f32_type();
                match val {
                    BasicValueEnum::IntValue(iv) => {
                        let result = if is_unsigned_src {
                            self.builder
                                .build_unsigned_int_to_float(iv, target, "uitofp")
                        } else {
                            self.builder.build_signed_int_to_float(iv, target, "sitofp")
                        }
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                        Ok(BasicValueEnum::FloatValue(result))
                    }
                    BasicValueEnum::FloatValue(fv) => {
                        if fv.get_type() == self.context.f64_type() {
                            let result = self
                                .builder
                                .build_float_trunc(fv, target, "fptrunc")
                                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::FloatValue(result))
                        } else {
                            Ok(val)
                        }
                    }
                    _ => Err(CompileError::codegen_error(
                        "to_f32: unsupported source type",
                    )),
                }
            }
            "to_f64" => {
                let target = self.context.f64_type();
                match val {
                    BasicValueEnum::IntValue(iv) => {
                        let result = if is_unsigned_src {
                            self.builder
                                .build_unsigned_int_to_float(iv, target, "uitofp")
                        } else {
                            self.builder.build_signed_int_to_float(iv, target, "sitofp")
                        }
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                        Ok(BasicValueEnum::FloatValue(result))
                    }
                    BasicValueEnum::FloatValue(fv) => {
                        if fv.get_type() == self.context.f32_type() {
                            let result = self
                                .builder
                                .build_float_ext(fv, target, "fpext")
                                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::FloatValue(result))
                        } else {
                            Ok(val)
                        }
                    }
                    _ => Err(CompileError::codegen_error(
                        "to_f64: unsupported source type",
                    )),
                }
            }
            "to_i32" => {
                let target = self.context.i32_type();
                match val {
                    BasicValueEnum::FloatValue(fv) => {
                        let result = self
                            .builder
                            .build_float_to_signed_int(fv, target, "fptosi")
                            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                        Ok(BasicValueEnum::IntValue(result))
                    }
                    BasicValueEnum::IntValue(iv) => {
                        let src_width = iv.get_type().get_bit_width();
                        if src_width > 32 {
                            let result = self
                                .builder
                                .build_int_truncate(iv, target, "trunc")
                                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::IntValue(result))
                        } else if src_width < 32 {
                            let result = if is_unsigned_src {
                                self.builder.build_int_z_extend(iv, target, "zext")
                            } else {
                                self.builder.build_int_s_extend(iv, target, "sext")
                            }
                            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::IntValue(result))
                        } else {
                            Ok(val)
                        }
                    }
                    _ => Err(CompileError::codegen_error(
                        "to_i32: unsupported source type",
                    )),
                }
            }
            "to_i64" => {
                let target = self.context.i64_type();
                match val {
                    BasicValueEnum::FloatValue(fv) => {
                        let result = self
                            .builder
                            .build_float_to_signed_int(fv, target, "fptosi")
                            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                        Ok(BasicValueEnum::IntValue(result))
                    }
                    BasicValueEnum::IntValue(iv) => {
                        let src_width = iv.get_type().get_bit_width();
                        if src_width < 64 {
                            let result = if is_unsigned_src {
                                self.builder.build_int_z_extend(iv, target, "zext")
                            } else {
                                self.builder.build_int_s_extend(iv, target, "sext")
                            }
                            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::IntValue(result))
                        } else {
                            Ok(val)
                        }
                    }
                    _ => Err(CompileError::codegen_error(
                        "to_i64: unsupported source type",
                    )),
                }
            }
            _ => unreachable!(),
        }
    }
}
