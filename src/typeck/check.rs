use std::collections::HashMap;

use crate::ast::Stmt;
use crate::error::CompileError;

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
                    span,
                } => {
                    if self.constants.contains_key(name) {
                        return Err(CompileError::type_error(
                            format!("'{name}' is a constant, cannot be used as variable name"),
                            span.clone(),
                        ));
                    }
                    let declared = types::resolve_type(ty)?;
                    let init_type = self.check_expr_with_hint(value, locals, Some(&declared))?;
                    if !types::types_compatible(&init_type, &declared) {
                        let hint = types::conversion_hint(&init_type, &declared)
                            .map(|h| format!(". {h}"))
                            .unwrap_or_default();
                        return Err(CompileError::type_error(
                            format!(
                                "cannot initialize '{name}' of type {declared} with {init_type}{hint}"
                            ),
                            value.span().clone(),
                        ));
                    }
                    locals.insert(name.clone(), (declared, *mutable));
                }
                Stmt::Assign {
                    target,
                    value,
                    span,
                } => {
                    if self.constants.contains_key(target) {
                        return Err(CompileError::type_error(
                            format!("cannot assign to constant '{target}'"),
                            span.clone(),
                        ));
                    }
                    let (var_type, mutable) = locals.get(target).cloned().ok_or_else(|| {
                        CompileError::type_error(
                            format!("undefined variable '{target}'"),
                            span.clone(),
                        )
                    })?;
                    if !mutable {
                        return Err(CompileError::type_error(
                            format!("cannot assign to immutable variable '{target}'"),
                            span.clone(),
                        ));
                    }
                    let val_type = self.check_expr(value, locals)?;
                    if !types::types_compatible(&val_type, &var_type) {
                        let hint = types::conversion_hint(&val_type, &var_type)
                            .map(|h| format!(". {h}"))
                            .unwrap_or_default();
                        return Err(CompileError::type_error(
                            format!(
                                "cannot assign {val_type} to '{target}' of type {var_type}{hint}"
                            ),
                            value.span().clone(),
                        ));
                    }
                }
                Stmt::IndexAssign {
                    object,
                    index,
                    value,
                    span,
                } => {
                    let (var_type, _) = locals.get(object).cloned().ok_or_else(|| {
                        CompileError::type_error(
                            format!("undefined variable '{object}'"),
                            span.clone(),
                        )
                    })?;
                    match &var_type {
                        Type::Pointer { mutable, inner, .. } => {
                            if !mutable {
                                return Err(CompileError::type_error(
                                    format!("cannot write through immutable pointer '{object}'. Declare as *mut to allow writes"),
                                    span.clone(),
                                ));
                            }
                            let idx_type = self.check_expr(index, locals)?;
                            if !idx_type.is_integer() {
                                return Err(CompileError::type_error(
                                    format!("index must be integer, got {idx_type}"),
                                    index.span().clone(),
                                ));
                            }
                            let val_type = self.check_expr(value, locals)?;
                            if !types::types_compatible(&val_type, inner) {
                                return Err(CompileError::type_error(
                                    format!("cannot assign {val_type} to element of {var_type}"),
                                    value.span().clone(),
                                ));
                            }
                        }
                        _ => {
                            return Err(CompileError::type_error(
                                format!("cannot index-assign to non-pointer '{object}'"),
                                span.clone(),
                            ));
                        }
                    }
                }
                Stmt::Return(Some(expr), _) => {
                    let actual = self.check_expr(expr, locals)?;
                    if *expected_return == Type::Void {
                        return Err(CompileError::type_error(
                            format!(
                                "function '{func_name}' has no return type but returns a value"
                            ),
                            expr.span().clone(),
                        ));
                    }
                    if !types::types_compatible(&actual, expected_return) {
                        return Err(CompileError::type_error(
                            format!(
                                "function '{func_name}' returns {actual} but expected {expected_return}"
                            ),
                            expr.span().clone(),
                        ));
                    }
                }
                Stmt::Return(None, span) => {
                    if *expected_return != Type::Void {
                        return Err(CompileError::type_error(
                            format!("function '{func_name}' must return {expected_return}"),
                            span.clone(),
                        ));
                    }
                }
                Stmt::ExprStmt(expr, _) => {
                    self.check_expr(expr, locals)?;
                }
                Stmt::If {
                    condition,
                    then_body,
                    else_body,
                    ..
                } => {
                    let cond_type = self.check_expr(condition, locals)?;
                    if !cond_type.is_bool() {
                        return Err(CompileError::type_error(
                            format!("if condition must be bool, got {cond_type}"),
                            condition.span().clone(),
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
                    ..
                } => {
                    let cond_type = self.check_expr(condition, locals)?;
                    if !cond_type.is_bool() {
                        return Err(CompileError::type_error(
                            format!("while condition must be bool, got {cond_type}"),
                            condition.span().clone(),
                        ));
                    }
                    self.check_body(while_body, locals, expected_return, func_name)?;
                }
                Stmt::ForEach {
                    var,
                    start,
                    end,
                    body: foreach_body,
                    ..
                } => {
                    let start_type = self.check_expr(start, locals)?;
                    if !start_type.is_integer() {
                        return Err(CompileError::type_error(
                            format!("foreach start must be integer, got {start_type}"),
                            start.span().clone(),
                        ));
                    }
                    let end_type = self.check_expr(end, locals)?;
                    if !end_type.is_integer() {
                        return Err(CompileError::type_error(
                            format!("foreach end must be integer, got {end_type}"),
                            end.span().clone(),
                        ));
                    }
                    let mut inner_locals = locals.clone();
                    inner_locals.insert(var.clone(), (Type::I32, false));
                    self.check_body(foreach_body, &mut inner_locals, expected_return, func_name)?;
                }
                Stmt::Unroll { count, span, .. } => {
                    if *count == 0 {
                        return Err(CompileError::type_error(
                            "unroll count must be greater than 0",
                            span.clone(),
                        ));
                    }
                    // Note: body check still uses the stmt's body
                    if let Stmt::Unroll { body, .. } = stmt {
                        self.check_body(&[*body.clone()], locals, expected_return, func_name)?;
                    }
                }
                Stmt::Function { span, .. } => {
                    return Err(CompileError::type_error(
                        "nested functions are not supported",
                        span.clone(),
                    ));
                }
                Stmt::Struct { .. } => {
                    // Registered in check_program first pass
                }
                Stmt::Const { .. } => {
                    // Registered in check_program first pass
                }
                Stmt::Kernel { span, .. } => {
                    return Err(CompileError::type_error(
                        "kernel should have been desugared before type checking",
                        span.clone(),
                    ));
                }
                Stmt::FieldAssign {
                    object,
                    field,
                    value,
                    span,
                } => {
                    let obj_type = self.check_expr(object, locals)?;
                    let struct_name = match &obj_type {
                        Type::Struct(name) => name.clone(),
                        Type::Pointer {
                            mutable: true,
                            inner,
                            ..
                        } => match inner.as_ref() {
                            Type::Struct(name) => name.clone(),
                            _ => {
                                return Err(CompileError::type_error(
                                    format!("field assign on non-struct pointer type {obj_type}"),
                                    span.clone(),
                                ))
                            }
                        },
                        Type::Pointer {
                            mutable: false,
                            inner,
                            ..
                        } if matches!(inner.as_ref(), Type::Struct(_)) => {
                            return Err(CompileError::type_error(
                                "cannot assign field through immutable pointer. Declare as *mut to allow writes",
                                span.clone(),
                            ))
                        }
                        _ => {
                            return Err(CompileError::type_error(
                                format!("field assign on non-struct type {obj_type}"),
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
                    let field_type = fields
                        .iter()
                        .find(|(n, _)| n == field)
                        .map(|(_, t)| t.clone())
                        .ok_or_else(|| {
                            CompileError::type_error(
                                format!("struct '{struct_name}' has no field '{field}'"),
                                span.clone(),
                            )
                        })?;
                    let val_type = self.check_expr(value, locals)?;
                    if !types::types_compatible(&val_type, &field_type) {
                        return Err(CompileError::type_error(
                            format!(
                                "cannot assign {val_type} to field '{field}' of type {field_type}"
                            ),
                            value.span().clone(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}
