use std::collections::HashMap;

use crate::ast::Stmt;
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
                    let init_type = self.check_expr_with_hint(value, locals, Some(&declared))?;
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
                        Type::Pointer { mutable, inner, .. } => {
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
                Stmt::Struct { .. } => {
                    // Registered in check_program first pass
                }
                Stmt::FieldAssign {
                    object,
                    field,
                    value,
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
                                    format!("field assign on non-struct pointer type {obj_type:?}"),
                                    Position::default(),
                                ))
                            }
                        },
                        Type::Pointer {
                            mutable: false,
                            inner,
                            ..
                        } if matches!(inner.as_ref(), Type::Struct(_)) => {
                            return Err(CompileError::type_error(
                                "cannot assign field through immutable pointer",
                                Position::default(),
                            ))
                        }
                        _ => {
                            return Err(CompileError::type_error(
                                format!("field assign on non-struct type {obj_type:?}"),
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
                    let field_type = fields
                        .iter()
                        .find(|(n, _)| n == field)
                        .map(|(_, t)| t.clone())
                        .ok_or_else(|| {
                            CompileError::type_error(
                                format!("struct '{struct_name}' has no field '{field}'"),
                                Position::default(),
                            )
                        })?;
                    let val_type = self.check_expr(value, locals)?;
                    if !types::types_compatible(&val_type, &field_type) {
                        return Err(CompileError::type_error(
                            format!(
                                "cannot assign {val_type:?} to field '{field}' of type {field_type:?}"
                            ),
                            Position::default(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}
