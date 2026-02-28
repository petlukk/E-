use std::collections::HashMap;

use crate::ast::{BinaryOp, Expr, Literal};
use crate::error::CompileError;

use super::types::{self, Type};
use super::TypeChecker;

impl TypeChecker {
    pub(super) fn check_expr(
        &self,
        expr: &Expr,
        locals: &HashMap<String, (Type, bool)>,
    ) -> crate::error::Result<Type> {
        match expr {
            Expr::Literal(Literal::Integer(_), _) => Ok(Type::IntLiteral),
            Expr::Literal(Literal::Float(_), _) => Ok(Type::FloatLiteral),
            Expr::Literal(Literal::Bool(_), _) => Ok(Type::Bool),
            Expr::Literal(Literal::StringLit(_), _) => Ok(Type::String),
            Expr::Variable(name, span) => {
                if let Some((ty, _)) = locals.get(name) {
                    return Ok(ty.clone());
                }
                if let Some((ty, _)) = self.constants.get(name) {
                    return Ok(ty.clone());
                }
                Err(CompileError::type_error(
                    format!("undefined variable '{name}'"),
                    span.clone(),
                ))
            }
            Expr::Not(inner, span) => {
                let inner_type = self.check_expr(inner, locals)?;
                if !inner_type.is_bool() {
                    return Err(CompileError::type_error(
                        format!("'!' requires bool operand, got {inner_type}"),
                        span.clone(),
                    ));
                }
                Ok(Type::Bool)
            }
            Expr::Negate(inner, span) => {
                let inner_type = self.check_expr(inner, locals)?;
                if inner_type.is_numeric() || inner_type.is_vector() {
                    Ok(inner_type)
                } else {
                    Err(CompileError::type_error(
                        format!("unary '-' requires numeric or vector operand, got {inner_type}"),
                        span.clone(),
                    ))
                }
            }
            Expr::Index {
                object,
                index,
                span,
            } => {
                let obj_type = self.check_expr(object, locals)?;
                let idx_type = self.check_expr(index, locals)?;
                if !idx_type.is_integer() {
                    return Err(CompileError::type_error(
                        format!("index must be integer, got {idx_type}"),
                        index.span().clone(),
                    ));
                }
                if let Type::Vector { elem, .. } = &obj_type {
                    return Ok(*elem.clone());
                }
                match obj_type.pointee() {
                    Some(inner) => Ok(inner.clone()),
                    None => Err(CompileError::type_error(
                        format!("cannot index type {obj_type}. Only pointers and vectors support indexing"),
                        span.clone(),
                    )),
                }
            }
            Expr::Binary(lhs, op, rhs, span) => {
                let lt = self.check_expr(lhs, locals)?;
                let rt = self.check_expr(rhs, locals)?;
                match op {
                    BinaryOp::Add
                    | BinaryOp::Subtract
                    | BinaryOp::Multiply
                    | BinaryOp::Divide
                    | BinaryOp::Modulo => types::unify_numeric(&lt, &rt, span.clone()),
                    BinaryOp::Less
                    | BinaryOp::Greater
                    | BinaryOp::LessEqual
                    | BinaryOp::GreaterEqual => {
                        types::unify_numeric(&lt, &rt, span.clone())?;
                        Ok(Type::Bool)
                    }
                    BinaryOp::Equal | BinaryOp::NotEqual => {
                        if lt.is_bool() && rt.is_bool() {
                            Ok(Type::Bool)
                        } else {
                            types::unify_numeric(&lt, &rt, span.clone())?;
                            Ok(Type::Bool)
                        }
                    }
                    BinaryOp::And | BinaryOp::Or => {
                        if !lt.is_bool() || !rt.is_bool() {
                            return Err(CompileError::type_error(
                                format!(
                                    "logical operators require bool operands, got {lt} and {rt}"
                                ),
                                span.clone(),
                            ));
                        }
                        Ok(Type::Bool)
                    }
                    BinaryOp::AddDot | BinaryOp::SubDot | BinaryOp::MulDot | BinaryOp::DivDot => {
                        types::unify_vector(&lt, &rt, span.clone())
                    }
                    BinaryOp::AndDot | BinaryOp::OrDot | BinaryOp::XorDot => {
                        let result = types::unify_vector(&lt, &rt, span.clone())?;
                        match &result {
                            Type::Vector { elem, .. } if elem.is_integer() => Ok(result),
                            Type::Vector { elem, .. } => Err(CompileError::type_error(
                                format!(
                                    "bitwise vector ops require integer element type, got {elem}"
                                ),
                                span.clone(),
                            )),
                            _ => Err(CompileError::type_error(
                                format!("bitwise vector ops require vector operands, got {result}"),
                                span.clone(),
                            )),
                        }
                    }
                    BinaryOp::LessDot
                    | BinaryOp::GreaterDot
                    | BinaryOp::LessEqualDot
                    | BinaryOp::GreaterEqualDot
                    | BinaryOp::EqualDot
                    | BinaryOp::NotEqualDot => {
                        types::unify_vector(&lt, &rt, span.clone())?;
                        match &lt {
                            Type::Vector { width, .. } => Ok(Type::Vector {
                                elem: Box::new(Type::Bool),
                                width: *width,
                            }),
                            _ => Err(CompileError::type_error(
                                format!("dotted comparison requires vectors, got {lt}"),
                                span.clone(),
                            )),
                        }
                    }
                }
            }

            Expr::Vector { elements, ty, span } => {
                let vec_type = types::resolve_type(ty)?;
                let (elem_type, width) = match &vec_type {
                    Type::Vector { elem, width } => (elem.as_ref(), *width),
                    _ => {
                        return Err(CompileError::type_error(
                            format!("expected vector type, got {vec_type}"),
                            span.clone(),
                        ))
                    }
                };

                if elements.len() != width {
                    return Err(CompileError::type_error(
                        format!("vector expects {width} elements, got {}", elements.len()),
                        span.clone(),
                    ));
                }

                for (i, el) in elements.iter().enumerate() {
                    let actual = self.check_expr(el, locals)?;
                    if !types::types_compatible(&actual, elem_type) {
                        return Err(CompileError::type_error(
                            format!("vector element {i} expected {elem_type}, got {actual}"),
                            el.span().clone(),
                        ));
                    }
                }
                Ok(vec_type)
            }
            Expr::ArrayLiteral(_, span) => Err(CompileError::type_error(
                "array literals can only be used as shuffle indices",
                span.clone(),
            )),
            Expr::FieldAccess {
                object,
                field,
                span,
            } => {
                let obj_type = self.check_expr(object, locals)?;
                let struct_name = match &obj_type {
                    Type::Struct(name) => name.clone(),
                    Type::Pointer { inner, .. } => match inner.as_ref() {
                        Type::Struct(name) => name.clone(),
                        _ => {
                            return Err(CompileError::type_error(
                                format!("field access on non-struct pointer type {obj_type}"),
                                span.clone(),
                            ))
                        }
                    },
                    _ => {
                        return Err(CompileError::type_error(
                            format!("field access on non-struct type {obj_type}"),
                            span.clone(),
                        ))
                    }
                };
                let fields = self.structs.get(&struct_name).ok_or_else(|| {
                    CompileError::type_error(
                        format!("unknown struct '{struct_name}'"),
                        span.clone(),
                    )
                })?;
                fields
                    .iter()
                    .find(|(n, _)| n == field)
                    .map(|(_, t)| t.clone())
                    .ok_or_else(|| {
                        CompileError::type_error(
                            format!("struct '{struct_name}' has no field '{field}'"),
                            span.clone(),
                        )
                    })
            }
            Expr::StructLiteral { name, fields, span } => {
                let def_fields = self.structs.get(name).ok_or_else(|| {
                    CompileError::type_error(format!("unknown struct '{name}'"), span.clone())
                })?;
                if fields.len() != def_fields.len() {
                    return Err(CompileError::type_error(
                        format!(
                            "struct '{name}' expects {} fields, got {}",
                            def_fields.len(),
                            fields.len()
                        ),
                        span.clone(),
                    ));
                }
                for (field_name, field_val) in fields {
                    let expected = def_fields
                        .iter()
                        .find(|(n, _)| n == field_name)
                        .map(|(_, t)| t.clone())
                        .ok_or_else(|| {
                            CompileError::type_error(
                                format!("struct '{name}' has no field '{field_name}'"),
                                field_val.span().clone(),
                            )
                        })?;
                    let actual = self.check_expr(field_val, locals)?;
                    if !types::types_compatible(&actual, &expected) {
                        return Err(CompileError::type_error(
                            format!("field '{field_name}': expected {expected}, got {actual}"),
                            field_val.span().clone(),
                        ));
                    }
                }
                Ok(Type::Struct(name.clone()))
            }
            Expr::Call { name, args, span } => {
                if let Some(result) = self.check_intrinsic_call(name, args, locals, None, span) {
                    return result;
                }
                let sig = self.functions.get(name).ok_or_else(|| {
                    CompileError::type_error(format!("undefined function '{name}'"), span.clone())
                })?;
                if args.len() != sig.params.len() {
                    return Err(CompileError::type_error(
                        format!(
                            "function '{}' expects {} arguments, got {}",
                            name,
                            sig.params.len(),
                            args.len()
                        ),
                        span.clone(),
                    ));
                }
                for (i, (arg, expected)) in args.iter().zip(&sig.params).enumerate() {
                    let actual = self.check_expr(arg, locals)?;
                    if !types::types_compatible(&actual, expected) {
                        return Err(CompileError::type_error(
                            format!(
                                "argument {} of '{}': expected {}, got {}",
                                i + 1,
                                name,
                                expected,
                                actual
                            ),
                            arg.span().clone(),
                        ));
                    }
                }
                Ok(sig.return_type.clone())
            }
        }
    }

    pub(super) fn check_expr_with_hint(
        &self,
        expr: &Expr,
        locals: &HashMap<String, (Type, bool)>,
        type_hint: Option<&Type>,
    ) -> crate::error::Result<Type> {
        if let Expr::Call { name, args, span } = expr {
            if let Some(result) = self.check_intrinsic_call(name, args, locals, type_hint, span) {
                return result;
            }
        }
        self.check_expr(expr, locals)
    }
}
