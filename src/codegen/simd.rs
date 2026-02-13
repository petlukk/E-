use inkwell::types::{BasicTypeEnum, VectorType};
use inkwell::values::{BasicValueEnum, FunctionValue, VectorValue};

use crate::ast::{BinaryOp, Expr, TypeAnnotation};
use crate::error::CompileError;
use crate::typeck::Type;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn is_simd_intrinsic(name: &str) -> bool {
        matches!(name, "splat" | "load" | "store")
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
                let val = self.compile_expr_typed(&args[0], elem_hint, function)?;
                let vec = self.build_splat(val, 4)?;
                Ok(BasicValueEnum::VectorValue(vec))
            }
            "load" => self.compile_load(args, function),
            "store" => self.compile_store(args, function),
            _ => Err(CompileError::codegen_error(format!(
                "unknown SIMD intrinsic '{name}'"
            ))),
        }
    }

    fn infer_load_vector_type(&self, ptr_arg: &Expr) -> VectorType<'ctx> {
        if let Expr::Variable(name) = ptr_arg {
            if let Some((_, Type::Pointer { inner, .. })) = self.variables.get(name) {
                let elem_llvm = self.llvm_type(inner);
                match elem_llvm {
                    BasicTypeEnum::FloatType(ft) => return ft.vec_type(4),
                    BasicTypeEnum::IntType(it) => return it.vec_type(4),
                    _ => {}
                }
            }
        }
        self.context.f32_type().vec_type(4)
    }

    fn compile_load(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let ptr_val = self.compile_expr(&args[0], function)?.into_pointer_value();
        let idx_val = self.compile_expr(&args[1], function)?.into_int_value();

        let elem_ptr = unsafe { self.builder.build_gep(ptr_val, &[idx_val], "load_gep") }
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        let vec_ty = self.infer_load_vector_type(&args[0]);
        let vec_ptr_ty = vec_ty.ptr_type(inkwell::AddressSpace::default());
        let vec_ptr = self
            .builder
            .build_pointer_cast(elem_ptr, vec_ptr_ty, "vec_ptr")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let val = self
            .builder
            .build_load(vec_ptr, "vec_load")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(val)
    }

    fn compile_store(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let ptr_val = self.compile_expr(&args[0], function)?.into_pointer_value();
        let idx_val = self.compile_expr(&args[1], function)?.into_int_value();
        let vec_val = self.compile_expr(&args[2], function)?.into_vector_value();

        let elem_ptr = unsafe { self.builder.build_gep(ptr_val, &[idx_val], "store_gep") }
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        let vec_ty = vec_val.get_type();
        let vec_ptr_ty = vec_ty.ptr_type(inkwell::AddressSpace::default());
        let vec_ptr = self
            .builder
            .build_pointer_cast(elem_ptr, vec_ptr_ty, "vec_ptr")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        self.builder
            .build_store(vec_ptr, vec_val)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(BasicValueEnum::IntValue(
            self.context.i32_type().const_int(0, false),
        ))
    }

    pub(crate) fn compile_vector_binary(
        &mut self,
        l: VectorValue<'ctx>,
        r: VectorValue<'ctx>,
        op: &BinaryOp,
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
                } else {
                    self.builder.build_int_signed_div(l, r, "vdiv")
                }
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

    pub(crate) fn compile_vector_literal(
        &mut self,
        elements: &[Expr],
        ty: &TypeAnnotation,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let vec_type = self.resolve_annotation(ty);
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

    pub(crate) fn build_splat(
        &self,
        val: BasicValueEnum<'ctx>,
        width: u32,
    ) -> crate::error::Result<VectorValue<'ctx>> {
        let first_ty = val.get_type();
        let vec_ty: BasicTypeEnum = match first_ty {
            BasicTypeEnum::IntType(t) => t.vec_type(width).into(),
            BasicTypeEnum::FloatType(t) => t.vec_type(width).into(),
            BasicTypeEnum::PointerType(t) => t.vec_type(width).into(),
            _ => {
                return Err(CompileError::codegen_error(format!(
                    "cannot splat type {first_ty:?}"
                )))
            }
        };

        let vec_ty_vector = vec_ty.into_vector_type();
        let undef = vec_ty_vector.get_undef();

        let vec0 = self
            .builder
            .build_insert_element(
                undef,
                val,
                self.context.i32_type().const_int(0, false),
                "splat_ins",
            )
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        let zero = self.context.i32_type().const_int(0, false);
        let mask: Vec<_> = (0..width).map(|_| zero).collect();
        let mask_val = VectorType::const_vector(&mask);

        let res = self
            .builder
            .build_shuffle_vector(vec0, vec0, mask_val, "splat_shuf")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        Ok(res)
    }
}
