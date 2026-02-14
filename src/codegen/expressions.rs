use inkwell::values::{BasicMetadataValueEnum, BasicValueEnum, FunctionValue};
use inkwell::{FloatPredicate, IntPredicate};

use crate::ast::{BinaryOp, Expr, Literal};
use crate::error::CompileError;
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
                    Type::I64 => {
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
            Expr::Variable(name) => {
                let (ptr, _ty) = self.variables.get(name).ok_or_else(|| {
                    CompileError::codegen_error(format!("undefined variable '{name}'"))
                })?;
                let val = self
                    .builder
                    .build_load(*ptr, name)
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
                    let elem_ptr = unsafe { self.builder.build_gep(ptr, &[idx], "elemptr") }
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                    let val = self
                        .builder
                        .build_load(elem_ptr, "elem")
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

                let compiled_args: Vec<BasicMetadataValueEnum> = args
                    .iter()
                    .map(|a| self.compile_expr(a, function).map(|v| v.into()))
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
        let hint = self.infer_binary_hint(lhs, rhs, type_hint);
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
            return self.compile_comparison(&left, &right, op);
        }

        if matches!(op, BinaryOp::And | BinaryOp::Or) {
            let l = left.into_int_value();
            let r = right.into_int_value();
            let result = match op {
                BinaryOp::And => self.builder.build_and(l, r, "and"),
                BinaryOp::Or => self.builder.build_or(l, r, "or"),
                _ => unreachable!(),
            }
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            return Ok(BasicValueEnum::IntValue(result));
        }

        match (&left, &right) {
            (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
                let result = match op {
                    BinaryOp::Add => self.builder.build_int_add(*l, *r, "add"),
                    BinaryOp::Subtract => self.builder.build_int_sub(*l, *r, "sub"),
                    BinaryOp::Multiply => self.builder.build_int_mul(*l, *r, "mul"),
                    BinaryOp::Divide => self.builder.build_int_signed_div(*l, *r, "div"),
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
                self.compile_vector_binary(*l, *r, op)
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
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        match (left, right) {
            (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
                let pred = match op {
                    BinaryOp::Less => IntPredicate::SLT,
                    BinaryOp::Greater => IntPredicate::SGT,
                    BinaryOp::LessEqual => IntPredicate::SLE,
                    BinaryOp::GreaterEqual => IntPredicate::SGE,
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

    pub(crate) fn compile_println(
        &mut self,
        arg: &Expr,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let printf = *self
            .functions
            .get("printf")
            .ok_or_else(|| CompileError::codegen_error("printf not declared"))?;

        let val = self.compile_expr(arg, function)?;

        match val {
            BasicValueEnum::IntValue(iv) => {
                let fmt_str = if iv.get_type().get_bit_width() == 64 {
                    "%ld\n"
                } else {
                    "%d\n"
                };
                let fmt = self
                    .builder
                    .build_global_string_ptr(fmt_str, "fmt")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                self.builder
                    .build_call(
                        printf,
                        &[fmt.as_pointer_value().into(), val.into()],
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
