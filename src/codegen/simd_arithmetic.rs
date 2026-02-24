use inkwell::types::VectorType;
use inkwell::values::{BasicValueEnum, FunctionValue, VectorValue};

use crate::ast::{BinaryOp, Expr};
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn compile_vector_binary(
        &mut self,
        l: VectorValue<'ctx>,
        r: VectorValue<'ctx>,
        op: &BinaryOp,
        is_unsigned: bool,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let is_float = l.get_type().get_element_type().is_float_type();
        let result = match op {
            BinaryOp::AddDot => {
                if is_float {
                    self.builder.build_float_add(l, r, "vadd")
                } else {
                    self.builder.build_int_add(l, r, "vadd")
                }
            }
            BinaryOp::SubDot => {
                if is_float {
                    self.builder.build_float_sub(l, r, "vsub")
                } else {
                    self.builder.build_int_sub(l, r, "vsub")
                }
            }
            BinaryOp::MulDot => {
                if is_float {
                    self.builder.build_float_mul(l, r, "vmul")
                } else {
                    self.builder.build_int_mul(l, r, "vmul")
                }
            }
            BinaryOp::DivDot => {
                if is_float {
                    self.builder.build_float_div(l, r, "vdiv")
                } else if is_unsigned {
                    self.builder.build_int_unsigned_div(l, r, "vdiv")
                } else {
                    self.builder.build_int_signed_div(l, r, "vdiv")
                }
            }
            BinaryOp::AndDot => {
                // Type checker guarantees integer vectors only
                self.builder.build_and(l, r, "vand")
            }
            BinaryOp::OrDot => {
                self.builder.build_or(l, r, "vor")
            }
            BinaryOp::XorDot => {
                self.builder.build_xor(l, r, "vxor")
            }
            BinaryOp::LessDot
            | BinaryOp::GreaterDot
            | BinaryOp::LessEqualDot
            | BinaryOp::GreaterEqualDot
            | BinaryOp::EqualDot
            | BinaryOp::NotEqualDot => {
                return self.compile_vector_compare(l, r, op, is_unsigned);
            }
            _ => {
                return Err(CompileError::codegen_error(
                    "unsupported vector binary operation",
                ))
            }
        };
        Ok(BasicValueEnum::VectorValue(
            result.map_err(|e| CompileError::codegen_error(e.to_string()))?,
        ))
    }

    fn compile_vector_compare(
        &self,
        l: VectorValue<'ctx>,
        r: VectorValue<'ctx>,
        op: &BinaryOp,
        is_unsigned: bool,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let is_float = l.get_type().get_element_type().is_float_type();
        if is_float {
            use inkwell::FloatPredicate;
            let pred = match op {
                BinaryOp::LessDot => FloatPredicate::OLT,
                BinaryOp::GreaterDot => FloatPredicate::OGT,
                BinaryOp::LessEqualDot => FloatPredicate::OLE,
                BinaryOp::GreaterEqualDot => FloatPredicate::OGE,
                BinaryOp::EqualDot => FloatPredicate::OEQ,
                BinaryOp::NotEqualDot => FloatPredicate::ONE,
                _ => unreachable!(),
            };
            let cmp = self
                .builder
                .build_float_compare(pred, l, r, "vcmp")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            Ok(cmp.into())
        } else {
            use inkwell::IntPredicate;
            let pred = match op {
                BinaryOp::LessDot => {
                    if is_unsigned {
                        IntPredicate::ULT
                    } else {
                        IntPredicate::SLT
                    }
                }
                BinaryOp::GreaterDot => {
                    if is_unsigned {
                        IntPredicate::UGT
                    } else {
                        IntPredicate::SGT
                    }
                }
                BinaryOp::LessEqualDot => {
                    if is_unsigned {
                        IntPredicate::ULE
                    } else {
                        IntPredicate::SLE
                    }
                }
                BinaryOp::GreaterEqualDot => {
                    if is_unsigned {
                        IntPredicate::UGE
                    } else {
                        IntPredicate::SGE
                    }
                }
                BinaryOp::EqualDot => IntPredicate::EQ,
                BinaryOp::NotEqualDot => IntPredicate::NE,
                _ => unreachable!(),
            };
            let cmp = self
                .builder
                .build_int_compare(pred, l, r, "vcmp")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            Ok(cmp.into())
        }
    }

    pub(super) fn compile_select(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let mask = self.compile_expr(&args[0], function)?.into_vector_value();
        let a = self.compile_expr(&args[1], function)?.into_vector_value();
        let b = self.compile_expr(&args[2], function)?.into_vector_value();

        let result = self
            .builder
            .build_select(mask, a, b, "vselect")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(result)
    }

    /// Widen lower 4 bytes of i8x16/u8x16 to f32x4.
    /// unsigned=true uses zero-extension (for u8); false uses sign-extension (for i8).
    pub(super) fn compile_widen_i8_f32x4(
        &mut self,
        args: &[Expr],
        unsigned: bool,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let vec16 = self.compile_expr(&args[0], function)?.into_vector_value();

        // Extract lower 4 bytes via shufflevector: <16 x i8> → <4 x i8>
        let undef16 = vec16.get_type().get_undef();
        let mask_vals: Vec<_> = (0u64..4)
            .map(|i| self.context.i32_type().const_int(i, false))
            .collect();
        let mask = VectorType::const_vector(&mask_vals);
        let lower4 = self
            .builder
            .build_shuffle_vector(vec16, undef16, mask, "widen_lower4")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // Extend <4 x i8> to <4 x i32>
        let i32x4_type = self.context.i32_type().vec_type(4);
        let i32x4 = if unsigned {
            self.builder
                .build_int_z_extend(lower4, i32x4_type, "zext_i8_i32")
        } else {
            self.builder
                .build_int_s_extend(lower4, i32x4_type, "sext_i8_i32")
        }
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // Convert <4 x i32> to <4 x float>
        let f32x4_type = self.context.f32_type().vec_type(4);
        let f32x4 = self
            .builder
            .build_signed_int_to_float(i32x4, f32x4_type, "sitofp_i32_f32")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        Ok(BasicValueEnum::VectorValue(f32x4))
    }

    /// Narrow f32x4 to i8x16 (lower 4 elements are the narrowed values, rest undef).
    pub(super) fn compile_narrow_f32x4_i8(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let f32x4 = self.compile_expr(&args[0], function)?.into_vector_value();

        // fptosi <4 x float> to <4 x i32>
        let i32x4_type = self.context.i32_type().vec_type(4);
        let i32x4 = self
            .builder
            .build_float_to_signed_int(f32x4, i32x4_type, "fptosi_f32_i32")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // trunc <4 x i32> to <4 x i8>
        let i8x4_type = self.context.i8_type().vec_type(4);
        let i8x4 = self
            .builder
            .build_int_truncate(i32x4, i8x4_type, "trunc_i32_i8")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // Expand to <16 x i8>: elements 0-3 from i8x4, elements 4-15 are undef
        let undef4 = i8x4.get_type().get_undef();
        let mask_vals: Vec<_> = (0u64..16)
            .map(|i| {
                let idx = if i < 4 { i } else { 4 }; // elements 4-15 come from undef4[0]
                self.context.i32_type().const_int(idx, false)
            })
            .collect();
        let mask = VectorType::const_vector(&mask_vals);
        let i8x16 = self
            .builder
            .build_shuffle_vector(i8x4, undef4, mask, "narrow_expand")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        Ok(BasicValueEnum::VectorValue(i8x16))
    }

    /// maddubs_i16(u8x16, i8x16) -> i16x8
    /// Multiplies unsigned 8-bit × signed 8-bit pairs, adds adjacent products → signed 16-bit.
    /// Maps to SSSE3 pmaddubsw (_mm_maddubs_epi16). Fast but accumulator overflows at i16.
    pub(super) fn compile_maddubs_i16(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value(); // u8x16
        let b = self.compile_expr(&args[1], function)?.into_vector_value(); // i8x16

        let i8x16_ty = self.context.i8_type().vec_type(16);
        let i16x8_ty = self.context.i16_type().vec_type(8);

        let fn_type = i16x8_ty.fn_type(&[i8x16_ty.into(), i8x16_ty.into()], false);
        let intrinsic = self
            .module
            .get_function("llvm.x86.ssse3.pmadd.ub.sw.128")
            .unwrap_or_else(|| {
                self.module
                    .add_function("llvm.x86.ssse3.pmadd.ub.sw.128", fn_type, None)
            });

        let result = self
            .builder
            .build_call(intrinsic, &[a.into(), b.into()], "maddubs_i16")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .left()
            .ok_or_else(|| CompileError::codegen_error("maddubs_i16 did not return a value"))?;

        Ok(result)
    }

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
            _ => Err(CompileError::codegen_error("sqrt expects float or float vector")),
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
                let one_vec = self.build_splat(BasicValueEnum::FloatValue(one_scalar), vec_ty.get_size())?;
                let result = self
                    .builder
                    .build_float_div(one_vec, vv, "vrsqrt")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(BasicValueEnum::VectorValue(result))
            }
            _ => Err(CompileError::codegen_error("rsqrt expects float or float vector")),
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
                            self.builder.build_unsigned_int_to_float(iv, target, "uitofp")
                        } else {
                            self.builder.build_signed_int_to_float(iv, target, "sitofp")
                        }
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                        Ok(BasicValueEnum::FloatValue(result))
                    }
                    BasicValueEnum::FloatValue(fv) => {
                        if fv.get_type() == self.context.f64_type() {
                            let result = self.builder
                                .build_float_trunc(fv, target, "fptrunc")
                                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::FloatValue(result))
                        } else {
                            Ok(val)
                        }
                    }
                    _ => Err(CompileError::codegen_error("to_f32: unsupported source type")),
                }
            }
            "to_f64" => {
                let target = self.context.f64_type();
                match val {
                    BasicValueEnum::IntValue(iv) => {
                        let result = if is_unsigned_src {
                            self.builder.build_unsigned_int_to_float(iv, target, "uitofp")
                        } else {
                            self.builder.build_signed_int_to_float(iv, target, "sitofp")
                        }
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                        Ok(BasicValueEnum::FloatValue(result))
                    }
                    BasicValueEnum::FloatValue(fv) => {
                        if fv.get_type() == self.context.f32_type() {
                            let result = self.builder
                                .build_float_ext(fv, target, "fpext")
                                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::FloatValue(result))
                        } else {
                            Ok(val)
                        }
                    }
                    _ => Err(CompileError::codegen_error("to_f64: unsupported source type")),
                }
            }
            "to_i32" => {
                let target = self.context.i32_type();
                match val {
                    BasicValueEnum::FloatValue(fv) => {
                        let result = self.builder
                            .build_float_to_signed_int(fv, target, "fptosi")
                            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                        Ok(BasicValueEnum::IntValue(result))
                    }
                    BasicValueEnum::IntValue(iv) => {
                        let src_width = iv.get_type().get_bit_width();
                        if src_width > 32 {
                            let result = self.builder
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
                    _ => Err(CompileError::codegen_error("to_i32: unsupported source type")),
                }
            }
            "to_i64" => {
                let target = self.context.i64_type();
                match val {
                    BasicValueEnum::FloatValue(fv) => {
                        let result = self.builder
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
                    _ => Err(CompileError::codegen_error("to_i64: unsupported source type")),
                }
            }
            _ => unreachable!(),
        }
    }

    /// maddubs_i32(u8x16, i8x16) -> i32x4
    /// Two-intrinsic chain: pmaddubsw → pmaddwd(ones) → i32x4.
    /// Safe against accumulator overflow; programmer explicitly chooses this wider type.
    pub(super) fn compile_maddubs_i32(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value(); // u8x16
        let b = self.compile_expr(&args[1], function)?.into_vector_value(); // i8x16

        let i8x16_ty = self.context.i8_type().vec_type(16);
        let i16x8_ty = self.context.i16_type().vec_type(8);
        let i32x4_ty = self.context.i32_type().vec_type(4);

        // Step 1: pmaddubsw — same as maddubs_i16
        let pmaddubsw_fn_type = i16x8_ty.fn_type(&[i8x16_ty.into(), i8x16_ty.into()], false);
        let pmaddubsw = self
            .module
            .get_function("llvm.x86.ssse3.pmadd.ub.sw.128")
            .unwrap_or_else(|| {
                self.module
                    .add_function("llvm.x86.ssse3.pmadd.ub.sw.128", pmaddubsw_fn_type, None)
            });

        let t = self
            .builder
            .build_call(pmaddubsw, &[a.into(), b.into()], "maddubs_i32_step1")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .left()
            .ok_or_else(|| CompileError::codegen_error("pmaddubsw did not return a value"))?
            .into_vector_value(); // i16x8

        // Step 2: pmaddwd with compile-time constant ones vector (i16x8)
        let one_i16 = self.context.i16_type().const_int(1, false);
        let ones_vals = [one_i16; 8];
        let ones = VectorType::const_vector(&ones_vals);

        let pmaddwd_fn_type = i32x4_ty.fn_type(&[i16x8_ty.into(), i16x8_ty.into()], false);
        let pmaddwd = self
            .module
            .get_function("llvm.x86.sse2.pmadd.wd")
            .unwrap_or_else(|| {
                self.module
                    .add_function("llvm.x86.sse2.pmadd.wd", pmaddwd_fn_type, None)
            });

        let result = self
            .builder
            .build_call(pmaddwd, &[t.into(), ones.into()], "maddubs_i32_step2")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .left()
            .ok_or_else(|| CompileError::codegen_error("pmaddwd did not return a value"))?;

        Ok(result)
    }
}
