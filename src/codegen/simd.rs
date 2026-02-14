use inkwell::types::{BasicTypeEnum, VectorType};
use inkwell::values::{BasicValueEnum, FunctionValue, VectorValue};

use crate::ast::{BinaryOp, Expr, TypeAnnotation};
use crate::error::CompileError;
use crate::typeck::Type;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn is_simd_intrinsic(name: &str) -> bool {
        matches!(
            name,
            "splat"
                | "load"
                | "store"
                | "fma"
                | "reduce_add"
                | "reduce_max"
                | "reduce_min"
                | "shuffle"
                | "select"
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
            "reduce_add" | "reduce_max" | "reduce_min" => self.compile_reduce(args, name, function),
            "shuffle" => self.compile_shuffle(args, function),
            "select" => self.compile_select(args, function),
            _ => Err(CompileError::codegen_error(format!(
                "unknown SIMD intrinsic '{name}'"
            ))),
        }
    }

    fn infer_load_vector_type(&self, ptr_arg: &Expr, type_hint: Option<&Type>) -> VectorType<'ctx> {
        if let Some(Type::Vector { elem, width }) = type_hint {
            let elem_llvm = self.llvm_type(elem);
            match elem_llvm {
                BasicTypeEnum::FloatType(ft) => return ft.vec_type(*width as u32),
                BasicTypeEnum::IntType(it) => return it.vec_type(*width as u32),
                _ => {}
            }
        }
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
        type_hint: Option<&Type>,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let ptr_val = self.compile_expr(&args[0], function)?.into_pointer_value();
        let idx_val = self.compile_expr(&args[1], function)?.into_int_value();

        let elem_ptr = unsafe { self.builder.build_gep(ptr_val, &[idx_val], "load_gep") }
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        let vec_ty = self.infer_load_vector_type(&args[0], type_hint);
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
            BinaryOp::LessDot
            | BinaryOp::GreaterDot
            | BinaryOp::LessEqualDot
            | BinaryOp::GreaterEqualDot
            | BinaryOp::EqualDot
            | BinaryOp::NotEqualDot => {
                return self.compile_vector_compare(l, r, op);
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
                BinaryOp::LessDot => IntPredicate::SLT,
                BinaryOp::GreaterDot => IntPredicate::SGT,
                BinaryOp::LessEqualDot => IntPredicate::SLE,
                BinaryOp::GreaterEqualDot => IntPredicate::SGE,
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

    pub(crate) fn compile_select(
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

    fn compile_shuffle(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let vec = self.compile_expr(&args[0], function)?.into_vector_value();

        let indices = match &args[1] {
            Expr::ArrayLiteral(elems) => elems
                .iter()
                .map(|e| match e {
                    Expr::Literal(crate::ast::Literal::Integer(n)) => {
                        self.context.i32_type().const_int(*n as u64, false)
                    }
                    _ => self.context.i32_type().const_int(0, false),
                })
                .collect::<Vec<_>>(),
            _ => {
                return Err(CompileError::codegen_error(
                    "shuffle requires array literal",
                ))
            }
        };

        let mask = VectorType::const_vector(&indices);
        let result = self
            .builder
            .build_shuffle_vector(vec, vec, mask, "shuffle")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(BasicValueEnum::VectorValue(result))
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
        let intrinsic = self.module.add_function(&intrinsic_name, fn_type, None);

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
        let vec = self.compile_expr(&args[0], function)?.into_vector_value();
        let vec_ty = vec.get_type();
        let elem_ty = vec_ty.get_element_type();
        let is_float = elem_ty.is_float_type();

        let (intrinsic_base, needs_start_value) = match (op, is_float) {
            ("reduce_add", true) => ("llvm.vector.reduce.fadd", true),
            ("reduce_add", false) => ("llvm.vector.reduce.add", false),
            ("reduce_max", true) => ("llvm.vector.reduce.fmax", false),
            ("reduce_max", false) => ("llvm.vector.reduce.smax", false),
            ("reduce_min", true) => ("llvm.vector.reduce.fmin", false),
            ("reduce_min", false) => ("llvm.vector.reduce.smin", false),
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
            let intrinsic = self.module.add_function(&intrinsic_name, fn_type, None);
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
            let intrinsic = self.module.add_function(&intrinsic_name, fn_type, None);
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
            if it == self.context.i32_type() {
                "i32"
            } else {
                "i64"
            }
        };
        format!("{base}.v{width}{elem_name}")
    }
}
