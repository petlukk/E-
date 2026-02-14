use std::collections::HashMap;

use crate::ast::{BinaryOp, Expr, Literal};
use crate::error::CompileError;
use crate::lexer::Position;

use super::types::{self, Type};
use super::TypeChecker;

impl TypeChecker {
    pub(super) fn check_expr(
        &self,
        expr: &Expr,
        locals: &HashMap<String, (Type, bool)>,
    ) -> crate::error::Result<Type> {
        match expr {
            Expr::Literal(Literal::Integer(_)) => Ok(Type::IntLiteral),
            Expr::Literal(Literal::Float(_)) => Ok(Type::FloatLiteral),
            Expr::Literal(Literal::Bool(_)) => Ok(Type::Bool),
            Expr::Literal(Literal::StringLit(_)) => Ok(Type::String),
            Expr::Variable(name) => locals.get(name).map(|(ty, _)| ty.clone()).ok_or_else(|| {
                CompileError::type_error(
                    format!("undefined variable '{name}'"),
                    Position::default(),
                )
            }),
            Expr::Not(inner) => {
                let inner_type = self.check_expr(inner, locals)?;
                if !inner_type.is_bool() {
                    return Err(CompileError::type_error(
                        format!("'!' requires bool operand, got {inner_type:?}"),
                        Position::default(),
                    ));
                }
                Ok(Type::Bool)
            }
            Expr::Index { object, index } => {
                let obj_type = self.check_expr(object, locals)?;
                let idx_type = self.check_expr(index, locals)?;
                if !idx_type.is_integer() {
                    return Err(CompileError::type_error(
                        format!("index must be integer, got {idx_type:?}"),
                        Position::default(),
                    ));
                }
                if let Type::Vector { elem, .. } = &obj_type {
                    return Ok(*elem.clone());
                }
                match obj_type.pointee() {
                    Some(inner) => Ok(inner.clone()),
                    None => Err(CompileError::type_error(
                        format!("cannot index type {obj_type:?}"),
                        Position::default(),
                    )),
                }
            }
            Expr::Binary(lhs, op, rhs) => {
                let lt = self.check_expr(lhs, locals)?;
                let rt = self.check_expr(rhs, locals)?;
                match op {
                    BinaryOp::Add
                    | BinaryOp::Subtract
                    | BinaryOp::Multiply
                    | BinaryOp::Divide
                    | BinaryOp::Modulo => types::unify_numeric(&lt, &rt),
                    BinaryOp::Less
                    | BinaryOp::Greater
                    | BinaryOp::LessEqual
                    | BinaryOp::GreaterEqual => {
                        types::unify_numeric(&lt, &rt)?;
                        Ok(Type::Bool)
                    }
                    BinaryOp::Equal | BinaryOp::NotEqual => {
                        if lt.is_bool() && rt.is_bool() {
                            Ok(Type::Bool)
                        } else {
                            types::unify_numeric(&lt, &rt)?;
                            Ok(Type::Bool)
                        }
                    }
                    BinaryOp::And | BinaryOp::Or => {
                        if !lt.is_bool() || !rt.is_bool() {
                            return Err(CompileError::type_error(
                                format!("logical operators require bool operands, got {lt:?} and {rt:?}"),
                                Position::default(),
                            ));
                        }
                        Ok(Type::Bool)
                    }
                    BinaryOp::AddDot | BinaryOp::SubDot | BinaryOp::MulDot | BinaryOp::DivDot => {
                        types::unify_vector(&lt, &rt)
                    }
                    BinaryOp::LessDot
                    | BinaryOp::GreaterDot
                    | BinaryOp::LessEqualDot
                    | BinaryOp::GreaterEqualDot
                    | BinaryOp::EqualDot
                    | BinaryOp::NotEqualDot => {
                        types::unify_vector(&lt, &rt)?;
                        match &lt {
                            Type::Vector { width, .. } => Ok(Type::Vector {
                                elem: Box::new(Type::Bool),
                                width: *width,
                            }),
                            _ => Err(CompileError::type_error(
                                format!("dotted comparison requires vectors, got {lt:?}"),
                                Position::default(),
                            )),
                        }
                    }
                }
            }

            Expr::Vector { elements, ty } => {
                let vec_type = types::resolve_type(ty)?;
                let (elem_type, width) = match &vec_type {
                    Type::Vector { elem, width } => (elem.as_ref(), *width),
                    _ => {
                        return Err(CompileError::type_error(
                            format!("expected vector type, got {vec_type:?}"),
                            Position::default(),
                        ))
                    }
                };

                if elements.len() != width {
                    return Err(CompileError::type_error(
                        format!("vector expects {width} elements, got {}", elements.len()),
                        Position::default(),
                    ));
                }

                for (i, el) in elements.iter().enumerate() {
                    let actual = self.check_expr(el, locals)?;
                    if !types::types_compatible(&actual, elem_type) {
                        return Err(CompileError::type_error(
                            format!("vector element {i} expected {elem_type:?}, got {actual:?}"),
                            Position::default(),
                        ));
                    }
                }
                Ok(vec_type)
            }
            Expr::ArrayLiteral(_) => Err(CompileError::type_error(
                "array literals can only be used as shuffle indices",
                Position::default(),
            )),
            Expr::FieldAccess { object, field } => {
                let obj_type = self.check_expr(object, locals)?;
                let struct_name = match &obj_type {
                    Type::Struct(name) => name.clone(),
                    Type::Pointer { inner, .. } => match inner.as_ref() {
                        Type::Struct(name) => name.clone(),
                        _ => {
                            return Err(CompileError::type_error(
                                format!("field access on non-struct pointer type {obj_type:?}"),
                                Position::default(),
                            ))
                        }
                    },
                    _ => {
                        return Err(CompileError::type_error(
                            format!("field access on non-struct type {obj_type:?}"),
                            Position::default(),
                        ))
                    }
                };
                let fields = self.structs.get(&struct_name).ok_or_else(|| {
                    CompileError::type_error(
                        format!("unknown struct '{struct_name}'"),
                        Position::default(),
                    )
                })?;
                fields
                    .iter()
                    .find(|(n, _)| n == field)
                    .map(|(_, t)| t.clone())
                    .ok_or_else(|| {
                        CompileError::type_error(
                            format!("struct '{struct_name}' has no field '{field}'"),
                            Position::default(),
                        )
                    })
            }
            Expr::StructLiteral { name, fields } => {
                let def_fields = self.structs.get(name).ok_or_else(|| {
                    CompileError::type_error(
                        format!("unknown struct '{name}'"),
                        Position::default(),
                    )
                })?;
                if fields.len() != def_fields.len() {
                    return Err(CompileError::type_error(
                        format!(
                            "struct '{name}' expects {} fields, got {}",
                            def_fields.len(),
                            fields.len()
                        ),
                        Position::default(),
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
                                Position::default(),
                            )
                        })?;
                    let actual = self.check_expr(field_val, locals)?;
                    if !types::types_compatible(&actual, &expected) {
                        return Err(CompileError::type_error(
                            format!("field '{field_name}': expected {expected:?}, got {actual:?}"),
                            Position::default(),
                        ));
                    }
                }
                Ok(Type::Struct(name.clone()))
            }
            Expr::Call { name, args } => {
                if let Some(result) = self.check_intrinsic_call(name, args, locals, None) {
                    return result;
                }
                let sig = self.functions.get(name).ok_or_else(|| {
                    CompileError::type_error(
                        format!("undefined function '{name}'"),
                        Position::default(),
                    )
                })?;
                if args.len() != sig.params.len() {
                    return Err(CompileError::type_error(
                        format!(
                            "function '{}' expects {} arguments, got {}",
                            name,
                            sig.params.len(),
                            args.len()
                        ),
                        Position::default(),
                    ));
                }
                for (i, (arg, expected)) in args.iter().zip(&sig.params).enumerate() {
                    let actual = self.check_expr(arg, locals)?;
                    if !types::types_compatible(&actual, expected) {
                        return Err(CompileError::type_error(
                            format!(
                                "argument {} of '{}': expected {:?}, got {:?}",
                                i + 1,
                                name,
                                expected,
                                actual
                            ),
                            Position::default(),
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
        if let Expr::Call { name, args } = expr {
            if let Some(result) = self.check_intrinsic_call(name, args, locals, type_hint) {
                return result;
            }
        }
        self.check_expr(expr, locals)
    }
}
