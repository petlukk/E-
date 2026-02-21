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

    /// maddubs(u8x16, i8x16) -> i16x8
    /// Multiplies unsigned 8-bit × signed 8-bit pairs, adds adjacent products → signed 16-bit.
    /// Maps to SSSE3 pmaddubsw (_mm_maddubs_epi16).
    pub(super) fn compile_maddubs(
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
            .build_call(intrinsic, &[a.into(), b.into()], "maddubs")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .left()
            .ok_or_else(|| CompileError::codegen_error("maddubs did not return a value"))?;

        Ok(result)
    }
}
