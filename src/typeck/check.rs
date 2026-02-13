use std::collections::HashMap;

use crate::ast::{BinaryOp, Expr, Literal, Stmt};
use crate::error::CompileError;
use crate::lexer::Position;

use super::types::{self, Type};
use super::TypeChecker;

impl TypeChecker {
    pub(super) fn check_body(
        &self,
        body: &[Stmt],
        locals: &mut HashMap<String, (Type, bool)>,
        expected_return: &Type,
        func_name: &str,
    ) -> crate::error::Result<()> {
        for stmt in body {
            match stmt {
                Stmt::Let {
                    name,
                    ty,
                    value,
                    mutable,
                } => {
                    let declared = types::resolve_type(ty)?;
                    let init_type = self.check_expr(value, locals)?;
                    if !types::types_compatible(&init_type, &declared) {
                        return Err(CompileError::type_error(
                            format!(
                                "cannot initialize '{name}' of type {declared:?} with {init_type:?}"
                            ),
                            Position::default(),
                        ));
                    }
                    locals.insert(name.clone(), (declared, *mutable));
                }
                Stmt::Assign { target, value } => {
                    let (var_type, mutable) = locals.get(target).cloned().ok_or_else(|| {
                        CompileError::type_error(
                            format!("undefined variable '{target}'"),
                            Position::default(),
                        )
                    })?;
                    if !mutable {
                        return Err(CompileError::type_error(
                            format!("cannot assign to immutable variable '{target}'"),
                            Position::default(),
                        ));
                    }
                    let val_type = self.check_expr(value, locals)?;
                    if !types::types_compatible(&val_type, &var_type) {
                        return Err(CompileError::type_error(
                            format!(
                                "cannot assign {val_type:?} to '{target}' of type {var_type:?}"
                            ),
                            Position::default(),
                        ));
                    }
                }
                Stmt::IndexAssign {
                    object,
                    index,
                    value,
                } => {
                    let (var_type, _) = locals.get(object).cloned().ok_or_else(|| {
                        CompileError::type_error(
                            format!("undefined variable '{object}'"),
                            Position::default(),
                        )
                    })?;
                    match &var_type {
                        Type::Pointer { mutable, inner } => {
                            if !mutable {
                                return Err(CompileError::type_error(
                                    format!("cannot write through immutable pointer '{object}'"),
                                    Position::default(),
                                ));
                            }
                            let idx_type = self.check_expr(index, locals)?;
                            if !idx_type.is_integer() {
                                return Err(CompileError::type_error(
                                    format!("index must be integer, got {idx_type:?}"),
                                    Position::default(),
                                ));
                            }
                            let val_type = self.check_expr(value, locals)?;
                            if !types::types_compatible(&val_type, inner) {
                                return Err(CompileError::type_error(
                                    format!(
                                        "cannot assign {val_type:?} to element of {var_type:?}"
                                    ),
                                    Position::default(),
                                ));
                            }
                        }
                        _ => {
                            return Err(CompileError::type_error(
                                format!("cannot index-assign to non-pointer '{object}'"),
                                Position::default(),
                            ));
                        }
                    }
                }
                Stmt::Return(Some(expr)) => {
                    let actual = self.check_expr(expr, locals)?;
                    if *expected_return == Type::Void {
                        return Err(CompileError::type_error(
                            format!(
                                "function '{func_name}' has no return type but returns a value"
                            ),
                            Position::default(),
                        ));
                    }
                    if !types::types_compatible(&actual, expected_return) {
                        return Err(CompileError::type_error(
                            format!(
                                "function '{func_name}' returns {actual:?} but expected {expected_return:?}"
                            ),
                            Position::default(),
                        ));
                    }
                }
                Stmt::Return(None) => {
                    if *expected_return != Type::Void {
                        return Err(CompileError::type_error(
                            format!("function '{func_name}' must return {expected_return:?}"),
                            Position::default(),
                        ));
                    }
                }
                Stmt::ExprStmt(expr) => {
                    self.check_expr(expr, locals)?;
                }
                Stmt::If {
                    condition,
                    then_body,
                    else_body,
                } => {
                    let cond_type = self.check_expr(condition, locals)?;
                    if !cond_type.is_bool() {
                        return Err(CompileError::type_error(
                            format!("if condition must be bool, got {cond_type:?}"),
                            Position::default(),
                        ));
                    }
                    self.check_body(then_body, locals, expected_return, func_name)?;
                    if let Some(else_stmts) = else_body {
                        self.check_body(else_stmts, locals, expected_return, func_name)?;
                    }
                }
                Stmt::While {
                    condition,
                    body: while_body,
                } => {
                    let cond_type = self.check_expr(condition, locals)?;
                    if !cond_type.is_bool() {
                        return Err(CompileError::type_error(
                            format!("while condition must be bool, got {cond_type:?}"),
                            Position::default(),
                        ));
                    }
                    self.check_body(while_body, locals, expected_return, func_name)?;
                }
                Stmt::Function { .. } => {
                    return Err(CompileError::type_error(
                        "nested functions are not supported",
                        Position::default(),
                    ));
                }
            }
        }
        Ok(())
    }

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
            Expr::Call { name, args } => {
                // Intrinsics
                if name == "splat" {
                    if args.len() != 1 {
                        return Err(CompileError::type_error(
                            "splat expects 1 argument",
                            Position::default(),
                        ));
                    }
                    let arg_type = self.check_expr(&args[0], locals)?;
                    // splat(f32) -> f32x4 (default width 4 for now)
                    // Or should we infer? The AST for splat doesn't have type info.
                    // The return type depends on the arg type.
                    // f32 -> f32x4. i32 -> i32x4.
                    return match arg_type {
                        Type::F32 | Type::FloatLiteral => Ok(Type::Vector {
                            elem: Box::new(Type::F32),
                            width: 4,
                        }),
                        Type::I32 | Type::IntLiteral => Ok(Type::Vector {
                            elem: Box::new(Type::I32),
                            width: 4,
                        }),
                        _ => Err(CompileError::type_error(
                            format!("splat expects numeric, got {arg_type:?}"),
                            Position::default(),
                        )),
                    };
                }
                if name == "load" {
                    if args.len() != 2 {
                        return Err(CompileError::type_error(
                            "load expects 2 arguments (ptr, index)",
                            Position::default(),
                        ));
                    }
                    let ptr_type = self.check_expr(&args[0], locals)?;
                    let idx_type = self.check_expr(&args[1], locals)?;

                    if !idx_type.is_integer() {
                        return Err(CompileError::type_error(
                            "load index must be integer",
                            Position::default(),
                        ));
                    }

                    match ptr_type {
                        Type::Pointer { inner, .. } => {
                            if inner.is_numeric() {
                                return Ok(Type::Vector {
                                    elem: inner,
                                    width: 4,
                                });
                            } else {
                                return Err(CompileError::type_error(
                                    "load expects pointer to numeric type",
                                    Position::default(),
                                ));
                            }
                        }
                        _ => {
                            return Err(CompileError::type_error(
                                "load expects pointer",
                                Position::default(),
                            ))
                        }
                    }
                }
                if name == "store" {
                    // store(ptr, index, val)
                    if args.len() != 3 {
                        return Err(CompileError::type_error(
                            "store expects 3 arguments",
                            Position::default(),
                        ));
                    }
                    let ptr_type = self.check_expr(&args[0], locals)?;
                    let idx_type = self.check_expr(&args[1], locals)?;
                    let val_type = self.check_expr(&args[2], locals)?;

                    if !idx_type.is_integer() {
                        return Err(CompileError::type_error(
                            "store index must be integer",
                            Position::default(),
                        ));
                    }
                    match (ptr_type, val_type) {
                        (
                            Type::Pointer {
                                mutable: true,
                                inner,
                            },
                            Type::Vector { elem, .. },
                        ) => {
                            if !types::types_compatible(&elem, &inner) {
                                return Err(CompileError::type_error(
                                    format!("store mismatch: ptr to {inner:?}, val {elem:?}"),
                                    Position::default(),
                                ));
                            }
                            return Ok(Type::Void);
                        }
                        (Type::Pointer { mutable: false, .. }, _) => {
                            return Err(CompileError::type_error(
                                "store requires mutable pointer",
                                Position::default(),
                            ));
                        }
                        (_, _) => {
                            return Err(CompileError::type_error(
                                "store expects (mut ptr, index, vector)",
                                Position::default(),
                            ))
                        }
                    }
                }
                if name == "println" {
                    if args.len() != 1 {
                        return Err(CompileError::type_error(
                            "println expects exactly 1 argument",
                            Position::default(),
                        ));
                    }
                    let arg_type = self.check_expr(&args[0], locals)?;
                    if !arg_type.is_numeric() && arg_type != Type::String && !arg_type.is_vector() {
                        return Err(CompileError::type_error(
                            format!("println expects numeric, string, or vector result, got {arg_type:?}"),
                            Position::default(),
                        ));
                    }
                    return Ok(Type::Void);
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
}
