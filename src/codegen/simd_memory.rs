use inkwell::types::{BasicTypeEnum, VectorType};
use inkwell::values::{BasicValue, BasicValueEnum, FunctionValue, VectorValue};
use inkwell::IntPredicate;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::typeck::Type;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// Returns the natural alignment in bytes for an element type.
    fn element_alignment(&self, elem_ty: BasicTypeEnum<'ctx>) -> u32 {
        match elem_ty {
            BasicTypeEnum::FloatType(ft) => {
                if ft == self.context.f64_type() {
                    8
                } else {
                    4
                }
            }
            BasicTypeEnum::IntType(it) if it.get_bit_width() == 8 => 1,
            BasicTypeEnum::IntType(it) if it.get_bit_width() == 16 => 2,
            BasicTypeEnum::IntType(it) if it.get_bit_width() == 32 => 4,
            BasicTypeEnum::IntType(it) if it.get_bit_width() == 64 => 8,
            _ => 1,
        }
    }

    /// Builds a `<width x i1>` mask where lane `i` is true iff `i < count_val`.
    fn build_count_mask(
        &mut self,
        count_val: inkwell::values::IntValue<'ctx>,
        width: u32,
    ) -> crate::error::Result<VectorValue<'ctx>> {
        let i32_ty = self.context.i32_type();
        let lane_indices: Vec<_> = (0..width)
            .map(|i| i32_ty.const_int(i as u64, false))
            .collect();
        let indices_vec = VectorType::const_vector(&lane_indices);

        let count_i32 = self
            .builder
            .build_int_z_extend_or_bit_cast(count_val, i32_ty, "count_i32")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let count_splat = self.build_splat(count_i32.into(), width)?;

        let mask = self
            .builder
            .build_int_compare(IntPredicate::ULT, indices_vec, count_splat, "mask")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        Ok(mask)
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
        if let Expr::Variable(name, _) = ptr_arg {
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

    pub(super) fn compile_load(
        &mut self,
        args: &[Expr],
        type_hint: Option<&Type>,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let ptr_val = self.compile_expr(&args[0], function)?.into_pointer_value();
        let idx_val = self.compile_expr(&args[1], function)?.into_int_value();

        let vec_ty = self.infer_load_vector_type(&args[0], type_hint);
        let elem_ty = vec_ty.get_element_type();
        let elem_ptr = unsafe {
            self.builder
                .build_gep(elem_ty, ptr_val, &[idx_val], "load_gep")
        }
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        let val = self
            .builder
            .build_load(vec_ty, elem_ptr, "vec_load")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        let element_alignment = self.element_alignment(vec_ty.get_element_type());

        if let BasicValueEnum::VectorValue(vec_val) = val {
            let load_inst = vec_val.as_instruction_value().ok_or_else(|| {
                CompileError::codegen_error("vector load did not produce an instruction")
            })?;
            load_inst.set_alignment(element_alignment).map_err(|e| {
                CompileError::codegen_error(format!("failed to set alignment: {e}"))
            })?;
        }

        Ok(val)
    }

    pub(super) fn compile_store(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let ptr_val = self.compile_expr(&args[0], function)?.into_pointer_value();
        let idx_val = self.compile_expr(&args[1], function)?.into_int_value();
        let vec_val = self.compile_expr(&args[2], function)?.into_vector_value();

        let vec_ty = vec_val.get_type();
        let elem_ty = vec_ty.get_element_type();
        let elem_ptr = unsafe {
            self.builder
                .build_gep(elem_ty, ptr_val, &[idx_val], "store_gep")
        }
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        let store_inst = self
            .builder
            .build_store(elem_ptr, vec_val)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        let element_alignment = self.element_alignment(vec_ty.get_element_type());

        store_inst.set_alignment(element_alignment).map_err(|e| {
            CompileError::codegen_error(format!("failed to set store alignment: {e}"))
        })?;

        Ok(BasicValueEnum::IntValue(
            self.context.i32_type().const_int(0, false),
        ))
    }

    pub(super) fn compile_load_masked(
        &mut self,
        args: &[Expr],
        type_hint: Option<&Type>,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let ptr_val = self.compile_expr(&args[0], function)?.into_pointer_value();
        let idx_val = self.compile_expr(&args[1], function)?.into_int_value();
        let count_val = self.compile_expr(&args[2], function)?.into_int_value();

        let vec_ty = self.infer_load_vector_type(&args[0], type_hint);
        let elem_ty = vec_ty.get_element_type();
        let width = vec_ty.get_size();

        // GEP to offset position (same as compile_load)
        let elem_ptr = unsafe {
            self.builder
                .build_gep(elem_ty, ptr_val, &[idx_val], "masked_load_gep")
        }
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // Build mask: lane_indices < count_splat → <N x i1>
        let mask = self.build_count_mask(count_val, width)?;

        // Build zero passthrough vector
        let passthrough = vec_ty.const_zero();

        // Element alignment
        let element_alignment = self.element_alignment(elem_ty);

        // Declare @llvm.masked.load.vNTy.p0
        let i32_ty = self.context.i32_type();
        let suffix = self.llvm_vector_type_suffix(vec_ty);
        let intrinsic_name = format!("llvm.masked.load.{suffix}");
        let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
        let mask_ty = self.context.bool_type().vec_type(width);
        let fn_type = vec_ty.fn_type(
            &[
                ptr_type.into(),
                i32_ty.into(),
                mask_ty.into(),
                vec_ty.into(),
            ],
            false,
        );
        let intrinsic = self
            .module
            .get_function(&intrinsic_name)
            .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));

        let align_val = i32_ty.const_int(element_alignment as u64, false);

        let result = self
            .builder
            .build_call(
                intrinsic,
                &[
                    elem_ptr.into(),
                    align_val.into(),
                    mask.into(),
                    passthrough.into(),
                ],
                "masked_load",
            )
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .left()
            .ok_or_else(|| CompileError::codegen_error("masked load did not return a value"))?;

        Ok(result)
    }

    pub(super) fn compile_store_masked(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let ptr_val = self.compile_expr(&args[0], function)?.into_pointer_value();
        let idx_val = self.compile_expr(&args[1], function)?.into_int_value();
        let vec_val = self.compile_expr(&args[2], function)?.into_vector_value();
        let count_val = self.compile_expr(&args[3], function)?.into_int_value();

        let vec_ty = vec_val.get_type();
        let elem_ty = vec_ty.get_element_type();
        let width = vec_ty.get_size();

        // GEP to offset position (same as compile_store)
        let elem_ptr = unsafe {
            self.builder
                .build_gep(elem_ty, ptr_val, &[idx_val], "masked_store_gep")
        }
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // Build mask: lane_indices < count_splat → <N x i1>
        let mask = self.build_count_mask(count_val, width)?;

        // Element alignment
        let element_alignment = self.element_alignment(elem_ty);

        // Declare @llvm.masked.store.vNTy.p0
        let i32_ty = self.context.i32_type();
        let suffix = self.llvm_vector_type_suffix(vec_ty);
        let intrinsic_name = format!("llvm.masked.store.{suffix}");
        let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
        let mask_ty = self.context.bool_type().vec_type(width);
        let fn_type = self.context.void_type().fn_type(
            &[
                vec_ty.into(),
                ptr_type.into(),
                i32_ty.into(),
                mask_ty.into(),
            ],
            false,
        );
        let intrinsic = self
            .module
            .get_function(&intrinsic_name)
            .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));

        let align_val = i32_ty.const_int(element_alignment as u64, false);

        self.builder
            .build_call(
                intrinsic,
                &[
                    vec_val.into(),
                    elem_ptr.into(),
                    align_val.into(),
                    mask.into(),
                ],
                "",
            )
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        Ok(BasicValueEnum::IntValue(
            self.context.i32_type().const_int(0, false),
        ))
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

    fn build_gather_ptrs(
        &mut self,
        base: inkwell::values::PointerValue<'ctx>,
        indices: VectorValue<'ctx>,
        elem_ty: BasicTypeEnum<'ctx>,
    ) -> crate::error::Result<VectorValue<'ctx>> {
        let width = indices.get_type().get_size();
        let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
        let mut vec = ptr_ty.vec_type(width).get_undef();
        for i in 0..width {
            let lane = self.context.i32_type().const_int(i as u64, false);
            let idx = self
                .builder
                .build_extract_element(indices, lane, "idx")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            let gep = unsafe {
                self.builder
                    .build_gep(elem_ty, base, &[idx.into_int_value()], "gep")
            }
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            vec = self
                .builder
                .build_insert_element(vec, gep, lane, "ins")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        }
        Ok(vec)
    }

    pub(super) fn compile_gather(
        &mut self,
        args: &[Expr],
        type_hint: Option<&Type>,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let base = self.compile_expr(&args[0], function)?.into_pointer_value();
        let indices = self.compile_expr(&args[1], function)?.into_vector_value();
        let vec_ty = self.infer_load_vector_type(&args[0], type_hint);
        let elem_ty = vec_ty.get_element_type();
        let width = indices.get_type().get_size();
        let vec_ty = match elem_ty {
            BasicTypeEnum::FloatType(ft) => ft.vec_type(width),
            BasicTypeEnum::IntType(it) => it.vec_type(width),
            _ => {
                return Err(CompileError::codegen_error(
                    "unsupported gather element type",
                ))
            }
        };
        let ptr_vec = self.build_gather_ptrs(base, indices, elem_ty)?;
        let align = self.element_alignment(elem_ty);
        let i32_ty = self.context.i32_type();
        let true_val = self.context.bool_type().const_int(1, false);
        let mask = VectorType::const_vector(&vec![true_val; width as usize]);
        let passthrough = vec_ty.const_zero();
        let (w, en) = self.vector_type_parts(vec_ty);
        let name = format!("llvm.masked.gather.v{w}{en}.v{w}p0");
        let ptr_vec_ty = self
            .context
            .ptr_type(inkwell::AddressSpace::default())
            .vec_type(width);
        let mask_ty = self.context.bool_type().vec_type(width);
        let fn_type = vec_ty.fn_type(
            &[
                ptr_vec_ty.into(),
                i32_ty.into(),
                mask_ty.into(),
                vec_ty.into(),
            ],
            false,
        );
        let intrinsic = self
            .module
            .get_function(&name)
            .unwrap_or_else(|| self.module.add_function(&name, fn_type, None));
        let result = self
            .builder
            .build_call(
                intrinsic,
                &[
                    ptr_vec.into(),
                    i32_ty.const_int(align as u64, false).into(),
                    mask.into(),
                    passthrough.into(),
                ],
                "gather",
            )
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .left()
            .ok_or_else(|| CompileError::codegen_error("gather did not return a value"))?;
        Ok(result)
    }

    pub(super) fn compile_scatter(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if !self.avx512 {
            return Err(CompileError::codegen_error(
                "scatter requires AVX-512 for hardware support; compile with --avx512 or write a scalar loop explicitly",
            ));
        }
        let base = self.compile_expr(&args[0], function)?.into_pointer_value();
        let indices = self.compile_expr(&args[1], function)?.into_vector_value();
        let values = self.compile_expr(&args[2], function)?.into_vector_value();
        let vec_ty = values.get_type();
        let elem_ty = vec_ty.get_element_type();
        let width = vec_ty.get_size();
        let ptr_vec = self.build_gather_ptrs(base, indices, elem_ty)?;
        let align = self.element_alignment(elem_ty);
        let i32_ty = self.context.i32_type();
        let true_val = self.context.bool_type().const_int(1, false);
        let mask = VectorType::const_vector(&vec![true_val; width as usize]);
        let (w, en) = self.vector_type_parts(vec_ty);
        let name = format!("llvm.masked.scatter.v{w}{en}.v{w}p0");
        let ptr_vec_ty = self
            .context
            .ptr_type(inkwell::AddressSpace::default())
            .vec_type(width);
        let mask_ty = self.context.bool_type().vec_type(width);
        let fn_type = self.context.void_type().fn_type(
            &[
                vec_ty.into(),
                ptr_vec_ty.into(),
                i32_ty.into(),
                mask_ty.into(),
            ],
            false,
        );
        let intrinsic = self
            .module
            .get_function(&name)
            .unwrap_or_else(|| self.module.add_function(&name, fn_type, None));
        self.builder
            .build_call(
                intrinsic,
                &[
                    values.into(),
                    ptr_vec.into(),
                    i32_ty.const_int(align as u64, false).into(),
                    mask.into(),
                ],
                "",
            )
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(BasicValueEnum::IntValue(i32_ty.const_int(0, false)))
    }
}
