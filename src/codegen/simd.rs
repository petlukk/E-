use inkwell::types::VectorType;
use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::{Expr, TypeAnnotation};
use crate::error::CompileError;
use crate::typeck::Type;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// Returns true if the argument expression is a vector variable with an unsigned element type.
    pub(crate) fn infer_unsigned_elem_from_arg(&self, arg: &Expr) -> bool {
        if let Expr::Variable(name, _) = arg {
            if let Some((_, Type::Vector { elem, .. })) = self.variables.get(name) {
                return matches!(elem.as_ref(), Type::U8 | Type::U16);
            }
        }
        false
    }

    pub(crate) fn is_simd_intrinsic(name: &str) -> bool {
        matches!(
            name,
            "splat"
                | "load"
                | "store"
                | "fma"
                | "sqrt"
                | "rsqrt"
                | "reduce_add"
                | "reduce_max"
                | "reduce_min"
                | "shuffle"
                | "select"
                | "widen_i8_f32x4"
                | "widen_u8_f32x4"
                | "narrow_f32x4_i8"
                | "maddubs_i16"
                | "maddubs_i32"
                | "to_f32"
                | "to_f64"
                | "to_i32"
                | "to_i64"
                | "prefetch"
        )
    }

    pub(crate) fn compile_simd_call(
        &mut self,
        name: &str,
        args: &[Expr],
        type_hint: Option<&Type>,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        match name {
            "splat" => {
                let elem_hint = if let Some(Type::Vector { elem, .. }) = type_hint {
                    Some(elem.as_ref())
                } else {
                    None
                };
                let width = match type_hint {
                    Some(Type::Vector { width, .. }) => *width as u32,
                    _ => 4,
                };
                let val = self.compile_expr_typed(&args[0], elem_hint, function)?;
                let vec = self.build_splat(val, width)?;
                Ok(BasicValueEnum::VectorValue(vec))
            }
            "load" => self.compile_load(args, type_hint, function),
            "store" => self.compile_store(args, function),
            "fma" => self.compile_fma(args, function),
            "sqrt" => self.compile_sqrt(args, function),
            "rsqrt" => self.compile_rsqrt(args, function),
            "reduce_add" | "reduce_max" | "reduce_min" => self.compile_reduce(args, name, function),
            "shuffle" => self.compile_shuffle(args, function),
            "select" => self.compile_select(args, function),
            "widen_i8_f32x4" => self.compile_widen_i8_f32x4(args, false, function),
            "widen_u8_f32x4" => self.compile_widen_i8_f32x4(args, true, function),
            "narrow_f32x4_i8" => self.compile_narrow_f32x4_i8(args, function),
            "maddubs_i16" => self.compile_maddubs_i16(args, function),
            "maddubs_i32" => self.compile_maddubs_i32(args, function),
            "to_f32" | "to_f64" | "to_i32" | "to_i64" => {
                self.compile_conversion(name, args, function)
            }
            "prefetch" => self.compile_prefetch(args, function),
            _ => Err(CompileError::codegen_error(format!(
                "unknown SIMD intrinsic '{name}'"
            ))),
        }
    }

    pub(crate) fn compile_vector_literal(
        &mut self,
        elements: &[Expr],
        ty: &TypeAnnotation,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let vec_type = Self::resolve_annotation(ty);
        let elem_hint = match &vec_type {
            Type::Vector { elem, .. } => Some(elem.as_ref()),
            _ => None,
        };
        let llvm_vec_type = self.llvm_type(&vec_type).into_vector_type();
        let mut vec_val = llvm_vec_type.get_undef();

        for (i, elem) in elements.iter().enumerate() {
            let elem_val = self.compile_expr_typed(elem, elem_hint, function)?;
            let idx = self.context.i32_type().const_int(i as u64, false);
            vec_val = self
                .builder
                .build_insert_element(vec_val, elem_val, idx, "vec_ins")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        }
        Ok(BasicValueEnum::VectorValue(vec_val))
    }

    fn compile_fma(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let c = self.compile_expr(&args[2], function)?.into_vector_value();

        let vec_ty = a.get_type();
        let intrinsic_name = self.llvm_vector_intrinsic_name("llvm.fma", vec_ty);

        let fn_type = vec_ty.fn_type(&[vec_ty.into(), vec_ty.into(), vec_ty.into()], false);
        let intrinsic = self
            .module
            .get_function(&intrinsic_name)
            .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));

        let result = self
            .builder
            .build_call(intrinsic, &[a.into(), b.into(), c.into()], "fma")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .left()
            .ok_or_else(|| CompileError::codegen_error("fma did not return a value"))?;
        Ok(result)
    }

    fn compile_reduce(
        &mut self,
        args: &[Expr],
        op: &str,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let is_unsigned_elem = self.infer_unsigned_elem_from_arg(&args[0]);
        let vec = self.compile_expr(&args[0], function)?.into_vector_value();
        let vec_ty = vec.get_type();
        let elem_ty = vec_ty.get_element_type();
        let is_float = elem_ty.is_float_type();

        let (intrinsic_base, needs_start_value) = match (op, is_float, is_unsigned_elem) {
            ("reduce_add", true, _) => ("llvm.vector.reduce.fadd", true),
            ("reduce_add", false, _) => ("llvm.vector.reduce.add", false),
            ("reduce_max", true, _) => ("llvm.vector.reduce.fmax", false),
            ("reduce_max", false, true) => ("llvm.vector.reduce.umax", false),
            ("reduce_max", false, false) => ("llvm.vector.reduce.smax", false),
            ("reduce_min", true, _) => ("llvm.vector.reduce.fmin", false),
            ("reduce_min", false, true) => ("llvm.vector.reduce.umin", false),
            ("reduce_min", false, false) => ("llvm.vector.reduce.smin", false),
            _ => {
                return Err(CompileError::codegen_error(format!(
                    "unknown reduction {op}"
                )))
            }
        };

        let intrinsic_name = self.llvm_vector_intrinsic_name(intrinsic_base, vec_ty);

        if needs_start_value {
            let zero = elem_ty.into_float_type().const_float(0.0);
            let fn_type = elem_ty
                .into_float_type()
                .fn_type(&[elem_ty.into_float_type().into(), vec_ty.into()], false);
            let intrinsic = self
                .module
                .get_function(&intrinsic_name)
                .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));
            let result = self
                .builder
                .build_call(intrinsic, &[zero.into(), vec.into()], "reduce")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .left()
                .ok_or_else(|| CompileError::codegen_error("reduce did not return a value"))?;
            Ok(result)
        } else {
            let fn_type = if is_float {
                elem_ty.into_float_type().fn_type(&[vec_ty.into()], false)
            } else {
                elem_ty.into_int_type().fn_type(&[vec_ty.into()], false)
            };
            let intrinsic = self
                .module
                .get_function(&intrinsic_name)
                .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));
            let result = self
                .builder
                .build_call(intrinsic, &[vec.into()], "reduce")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .left()
                .ok_or_else(|| CompileError::codegen_error("reduce did not return a value"))?;
            Ok(result)
        }
    }

    pub(crate) fn llvm_vector_intrinsic_name(
        &self,
        base: &str,
        vec_ty: VectorType<'ctx>,
    ) -> String {
        let width = vec_ty.get_size();
        let elem = vec_ty.get_element_type();
        let elem_name = if elem.is_float_type() {
            let ft = elem.into_float_type();
            if ft == self.context.f32_type() {
                "f32"
            } else {
                "f64"
            }
        } else {
            let it = elem.into_int_type();
            match it.get_bit_width() {
                8 => "i8",
                16 => "i16",
                32 => "i32",
                _ => "i64",
            }
        };
        format!("{base}.v{width}{elem_name}")
    }
}
