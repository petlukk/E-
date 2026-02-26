use inkwell::values::{BasicMetadataValueEnum, BasicValueEnum, FunctionValue};
use inkwell::{FloatPredicate, IntPredicate};

use crate::ast::{BinaryOp, Expr, Literal};
use crate::error::CompileError;
use crate::typeck::types::is_unsigned;
use crate::typeck::Type;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn compile_expr(
        &mut self,
        expr: &Expr,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        self.compile_expr_typed(expr, None, function)
    }

    pub(crate) fn compile_expr_typed(
        &mut self,
        expr: &Expr,
        type_hint: Option<&Type>,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        match expr {
            Expr::Literal(Literal::Integer(n)) => {
                let ty = type_hint.unwrap_or(&Type::I32);
                match ty {
                    Type::I8 | Type::U8 => {
                        let val = self.context.i8_type().const_int(*n as u64, true);
                        Ok(BasicValueEnum::IntValue(val))
                    }
                    Type::I16 | Type::U16 => {
                        let val = self.context.i16_type().const_int(*n as u64, true);
                        Ok(BasicValueEnum::IntValue(val))
                    }
                    Type::I64 | Type::U64 => {
                        let val = self.context.i64_type().const_int(*n as u64, true);
                        Ok(BasicValueEnum::IntValue(val))
                    }
                    _ => {
                        let val = self.context.i32_type().const_int(*n as u64, true);
                        Ok(BasicValueEnum::IntValue(val))
                    }
                }
            }
            Expr::Literal(Literal::Float(n)) => {
                let ty = type_hint.unwrap_or(&Type::F64);
                match ty {
                    Type::F32 => {
                        let val = self.context.f32_type().const_float(*n);
                        Ok(BasicValueEnum::FloatValue(val))
                    }
                    _ => {
                        let val = self.context.f64_type().const_float(*n);
                        Ok(BasicValueEnum::FloatValue(val))
                    }
                }
            }
            Expr::Literal(Literal::StringLit(s)) => {
                let global = self
                    .builder
                    .build_global_string_ptr(s, "str")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(BasicValueEnum::PointerValue(global.as_pointer_value()))
            }
            Expr::Literal(Literal::Bool(b)) => {
                let val = self.context.bool_type().const_int(*b as u64, false);
                Ok(BasicValueEnum::IntValue(val))
            }
            Expr::Not(inner) => {
                let val = self.compile_expr(inner, function)?;
                let int_val = val.into_int_value();
                let result = self
                    .builder
                    .build_not(int_val, "not")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(BasicValueEnum::IntValue(result))
            }
            Expr::Negate(inner) => {
                let val = self.compile_expr_typed(inner, type_hint, function)?;
                match val {
                    BasicValueEnum::IntValue(iv) => {
                        let result = self
                            .builder
                            .build_int_neg(iv, "neg")
                            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                        Ok(BasicValueEnum::IntValue(result))
                    }
                    BasicValueEnum::FloatValue(fv) => {
                        let result = self
                            .builder
                            .build_float_neg(fv, "fneg")
                            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                        Ok(BasicValueEnum::FloatValue(result))
                    }
                    BasicValueEnum::VectorValue(vv) => {
                        let is_float = vv.get_type().get_element_type().is_float_type();
                        if is_float {
                            let result = self
                                .builder
                                .build_float_neg(vv, "vneg")
                                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::VectorValue(result))
                        } else {
                            let result = self
                                .builder
                                .build_int_neg(vv, "vneg")
                                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::VectorValue(result))
                        }
                    }
                    _ => Err(CompileError::codegen_error("unary '-' on unsupported type")),
                }
            }
            Expr::Variable(name) => {
                let (ptr, ty) = self.variables.get(name).ok_or_else(|| {
                    CompileError::codegen_error(format!("undefined variable '{name}'"))
                })?;
                let pointee_ty = self.llvm_type(ty);
                let val = self
                    .builder
                    .build_load(pointee_ty, *ptr, name)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(val)
            }
            Expr::Index { object, index } => {
                let obj_val = self.compile_expr(object, function)?;
                let idx = self.compile_expr(index, function)?.into_int_value();
                if let BasicValueEnum::VectorValue(vec) = obj_val {
                    let val = self
                        .builder
                        .build_extract_element(vec, idx, "elem")
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                    Ok(val)
                } else {
                    let ptr = obj_val.into_pointer_value();
                    let inner_type = if let Expr::Variable(name) = object.as_ref() {
                        if let Some((_, Type::Pointer { inner, .. })) = self.variables.get(name) {
                            self.llvm_type(inner)
                        } else {
                            self.context.i32_type().into()
                        }
                    } else {
                        self.context.i32_type().into()
                    };
                    let elem_ptr =
                        unsafe { self.builder.build_gep(inner_type, ptr, &[idx], "elemptr") }
                            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                    let val = self
                        .builder
                        .build_load(inner_type, elem_ptr, "elem")
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                    Ok(val)
                }
            }
            Expr::Binary(lhs, op, rhs) => self.compile_binary(lhs, op, rhs, type_hint, function),
            Expr::Call { name, args } => {
                if name == "println" {
                    return self.compile_println(&args[0], function);
                }
                if Self::is_simd_intrinsic(name) {
                    return self.compile_simd_call(name, args, type_hint, function);
                }

                let callee = *self.functions.get(name).ok_or_else(|| {
                    CompileError::codegen_error(format!("undefined function '{name}'"))
                })?;

                let param_hints: Option<Vec<Type>> =
                    self.func_signatures.get(name).map(|(pts, _)| pts.clone());
                let compiled_args: Vec<BasicMetadataValueEnum> = args
                    .iter()
                    .enumerate()
                    .map(|(i, a)| {
                        let hint = param_hints.as_ref().and_then(|pts| pts.get(i));
                        self.compile_expr_typed(a, hint, function).map(|v| v.into())
                    })
                    .collect::<Result<_, _>>()?;
                let result = self
                    .builder
                    .build_call(callee, &compiled_args, "call")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                match result.try_as_basic_value().left() {
                    Some(val) => Ok(val),
                    None => Ok(BasicValueEnum::IntValue(
                        self.context.i32_type().const_int(0, false),
                    )),
                }
            }
            Expr::Vector { elements, ty } => self.compile_vector_literal(elements, ty, function),
            Expr::ArrayLiteral(_) => Err(CompileError::codegen_error(
                "array literals can only be used as shuffle indices",
            )),
            Expr::FieldAccess { object, field } => {
                self.compile_field_access(object, field, function)
            }
            Expr::StructLiteral { name, fields } => {
                self.compile_struct_literal(name, fields, function)
            }
        }
    }

    fn compile_binary(
        &mut self,
        lhs: &Expr,
        op: &BinaryOp,
        rhs: &Expr,
        type_hint: Option<&Type>,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        match op {
            BinaryOp::And => return self.compile_short_circuit_and(lhs, rhs, function),
            BinaryOp::Or => return self.compile_short_circuit_or(lhs, rhs, function),
            _ => {}
        }

        let hint = self.infer_binary_hint(lhs, rhs, type_hint);
        let unsigned = hint.as_ref().map(is_unsigned).unwrap_or(false);
        let left = self.compile_expr_typed(lhs, hint.as_ref(), function)?;
        let right = self.compile_expr_typed(rhs, hint.as_ref(), function)?;

        if matches!(
            op,
            BinaryOp::Less
                | BinaryOp::Greater
                | BinaryOp::LessEqual
                | BinaryOp::GreaterEqual
                | BinaryOp::Equal
                | BinaryOp::NotEqual
        ) {
            return self.compile_comparison(&left, &right, op, unsigned);
        }

        match (&left, &right) {
            (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
                let result = match op {
                    BinaryOp::Add => self.builder.build_int_add(*l, *r, "add"),
                    BinaryOp::Subtract => self.builder.build_int_sub(*l, *r, "sub"),
                    BinaryOp::Multiply => self.builder.build_int_mul(*l, *r, "mul"),
                    BinaryOp::Divide if unsigned => {
                        self.builder.build_int_unsigned_div(*l, *r, "div")
                    }
                    BinaryOp::Divide => self.builder.build_int_signed_div(*l, *r, "div"),
                    BinaryOp::Modulo if unsigned => {
                        self.builder.build_int_unsigned_rem(*l, *r, "rem")
                    }
                    BinaryOp::Modulo => self.builder.build_int_signed_rem(*l, *r, "rem"),
                    _ => unreachable!(),
                }
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(BasicValueEnum::IntValue(result))
            }
            (BasicValueEnum::FloatValue(l), BasicValueEnum::FloatValue(r)) => {
                let result = match op {
                    BinaryOp::Add => self.builder.build_float_add(*l, *r, "fadd"),
                    BinaryOp::Subtract => self.builder.build_float_sub(*l, *r, "fsub"),
                    BinaryOp::Multiply => self.builder.build_float_mul(*l, *r, "fmul"),
                    BinaryOp::Divide => self.builder.build_float_div(*l, *r, "fdiv"),
                    BinaryOp::Modulo => self.builder.build_float_rem(*l, *r, "frem"),
                    _ => unreachable!(),
                }
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(BasicValueEnum::FloatValue(result))
            }
            (BasicValueEnum::VectorValue(l), BasicValueEnum::VectorValue(r)) => {
                let vec_unsigned = match hint.as_ref() {
                    Some(Type::Vector { elem, .. }) => is_unsigned(elem),
                    _ => false,
                };
                self.compile_vector_binary(*l, *r, op, vec_unsigned)
            }
            _ => Err(CompileError::codegen_error(
                "mismatched operand types in binary expression",
            )),
        }
    }

    fn infer_binary_hint(&self, lhs: &Expr, rhs: &Expr, outer_hint: Option<&Type>) -> Option<Type> {
        if let Expr::Variable(name) = lhs {
            if let Some((_, ty)) = self.variables.get(name) {
                return Some(ty.clone());
            }
        }
        if let Expr::Variable(name) = rhs {
            if let Some((_, ty)) = self.variables.get(name) {
                return Some(ty.clone());
            }
        }
        outer_hint.cloned()
    }

    fn compile_comparison(
        &mut self,
        left: &BasicValueEnum<'ctx>,
        right: &BasicValueEnum<'ctx>,
        op: &BinaryOp,
        unsigned: bool,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        match (left, right) {
            (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
                let pred = match op {
                    BinaryOp::Less => {
                        if unsigned {
                            IntPredicate::ULT
                        } else {
                            IntPredicate::SLT
                        }
                    }
                    BinaryOp::Greater => {
                        if unsigned {
                            IntPredicate::UGT
                        } else {
                            IntPredicate::SGT
                        }
                    }
                    BinaryOp::LessEqual => {
                        if unsigned {
                            IntPredicate::ULE
                        } else {
                            IntPredicate::SLE
                        }
                    }
                    BinaryOp::GreaterEqual => {
                        if unsigned {
                            IntPredicate::UGE
                        } else {
                            IntPredicate::SGE
                        }
                    }
                    BinaryOp::Equal => IntPredicate::EQ,
                    BinaryOp::NotEqual => IntPredicate::NE,
                    _ => unreachable!(),
                };
                let result = self
                    .builder
                    .build_int_compare(pred, *l, *r, "cmp")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(BasicValueEnum::IntValue(result))
            }
            (BasicValueEnum::FloatValue(l), BasicValueEnum::FloatValue(r)) => {
                let pred = match op {
                    BinaryOp::Less => FloatPredicate::OLT,
                    BinaryOp::Greater => FloatPredicate::OGT,
                    BinaryOp::LessEqual => FloatPredicate::OLE,
                    BinaryOp::GreaterEqual => FloatPredicate::OGE,
                    BinaryOp::Equal => FloatPredicate::OEQ,
                    BinaryOp::NotEqual => FloatPredicate::ONE,
                    _ => unreachable!(),
                };
                let result = self
                    .builder
                    .build_float_compare(pred, *l, *r, "fcmp")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(BasicValueEnum::IntValue(result))
            }
            _ => Err(CompileError::codegen_error(
                "mismatched types in comparison",
            )),
        }
    }

    /// Short-circuit AND: if left is false, result is false (skip right)
    fn compile_short_circuit_and(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let left = self.compile_expr(lhs, function)?.into_int_value();
        let lhs_bb = self.builder.get_insert_block().unwrap();

        let rhs_bb = self.context.append_basic_block(function, "and_rhs");
        let merge_bb = self.context.append_basic_block(function, "and_merge");

        self.builder
            .build_conditional_branch(left, rhs_bb, merge_bb)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        self.builder.position_at_end(rhs_bb);
        let right = self.compile_expr(rhs, function)?.into_int_value();
        let rhs_end_bb = self.builder.get_insert_block().unwrap();
        self.builder
            .build_unconditional_branch(merge_bb)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        self.builder.position_at_end(merge_bb);
        let bool_ty = self.context.bool_type();
        let phi = self
            .builder
            .build_phi(bool_ty, "and_result")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let false_val = bool_ty.const_int(0, false);
        phi.add_incoming(&[(&false_val, lhs_bb), (&right, rhs_end_bb)]);

        Ok(BasicValueEnum::IntValue(
            phi.as_basic_value().into_int_value(),
        ))
    }

    /// Short-circuit OR: if left is true, result is true (skip right)
    fn compile_short_circuit_or(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let left = self.compile_expr(lhs, function)?.into_int_value();
        let lhs_bb = self.builder.get_insert_block().unwrap();

        let rhs_bb = self.context.append_basic_block(function, "or_rhs");
        let merge_bb = self.context.append_basic_block(function, "or_merge");

        self.builder
            .build_conditional_branch(left, merge_bb, rhs_bb)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        self.builder.position_at_end(rhs_bb);
        let right = self.compile_expr(rhs, function)?.into_int_value();
        let rhs_end_bb = self.builder.get_insert_block().unwrap();
        self.builder
            .build_unconditional_branch(merge_bb)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        self.builder.position_at_end(merge_bb);
        let bool_ty = self.context.bool_type();
        let phi = self
            .builder
            .build_phi(bool_ty, "or_result")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let true_val = bool_ty.const_int(1, false);
        phi.add_incoming(&[(&true_val, lhs_bb), (&right, rhs_end_bb)]);

        Ok(BasicValueEnum::IntValue(
            phi.as_basic_value().into_int_value(),
        ))
    }

    pub(super) fn arg_is_unsigned(&self, expr: &Expr) -> bool {
        if let Expr::Variable(name) = expr {
            if let Some((_, ty)) = self.variables.get(name) {
                return ty.is_unsigned_integer();
            }
        }
        false
    }
}
