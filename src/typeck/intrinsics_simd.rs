use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::types::{self, Type};
use super::TypeChecker;

impl TypeChecker {
    pub(super) fn check_load(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        type_hint: Option<&Type>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "load expects 2 arguments (ptr, index)",
                span.clone(),
            ));
        }
        let ptr_type = self.check_expr(&args[0], locals)?;
        let idx_type = self.check_expr(&args[1], locals)?;

        if !idx_type.is_integer() {
            return Err(CompileError::type_error(
                "load index must be integer",
                args[1].span().clone(),
            ));
        }

        let width = match type_hint {
            Some(Type::Vector { width, .. }) => *width,
            _ => 4,
        };

        match ptr_type {
            Type::Pointer { inner, .. } => {
                if inner.is_numeric() {
                    Ok(Type::Vector { elem: inner, width })
                } else {
                    Err(CompileError::type_error(
                        "load expects pointer to numeric type",
                        args[0].span().clone(),
                    ))
                }
            }
            _ => Err(CompileError::type_error(
                format!("load expects a pointer, got {ptr_type}"),
                args[0].span().clone(),
            )),
        }
    }

    pub(super) fn check_store(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 3 {
            return Err(CompileError::type_error(
                "store expects 3 arguments",
                span.clone(),
            ));
        }
        let ptr_type = self.check_expr(&args[0], locals)?;
        let idx_type = self.check_expr(&args[1], locals)?;
        let val_type = self.check_expr(&args[2], locals)?;

        if !idx_type.is_integer() {
            return Err(CompileError::type_error(
                "store index must be integer",
                args[1].span().clone(),
            ));
        }
        match (ptr_type, val_type) {
            (
                Type::Pointer {
                    mutable: true,
                    inner,
                    ..
                },
                Type::Vector { elem, .. },
            ) => {
                if !types::types_compatible(&elem, &inner) {
                    return Err(CompileError::type_error(
                        format!("store mismatch: ptr to {inner}, val {elem}"),
                        span.clone(),
                    ));
                }
                Ok(Type::Void)
            }
            (Type::Pointer { mutable: false, .. }, _) => Err(CompileError::type_error(
                "store requires mutable pointer. Declare as *mut to allow writes",
                args[0].span().clone(),
            )),
            (_, _) => Err(CompileError::type_error(
                "store expects (mut ptr, index, vector)",
                span.clone(),
            )),
        }
    }

    pub(super) fn check_reduction(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                format!("{name} expects 1 argument"),
                span.clone(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;
        match &arg_type {
            Type::Vector { elem, .. } => Ok(*elem.clone()),
            _ => Err(CompileError::type_error(
                format!("{name} expects vector argument, got {arg_type}"),
                args[0].span().clone(),
            )),
        }
    }

    pub(super) fn check_shuffle(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "shuffle expects 2 arguments (vector, indices)",
                span.clone(),
            ));
        }
        let vec_type = self.check_expr(&args[0], locals)?;
        let width = match &vec_type {
            Type::Vector { width, .. } => *width,
            _ => {
                return Err(CompileError::type_error(
                    format!("shuffle first argument must be vector, got {vec_type}"),
                    args[0].span().clone(),
                ))
            }
        };

        match &args[1] {
            Expr::ArrayLiteral(indices, arr_span) => {
                if indices.len() != width {
                    return Err(CompileError::type_error(
                        format!(
                            "shuffle indices length {} != vector width {width}",
                            indices.len()
                        ),
                        arr_span.clone(),
                    ));
                }
                for (i, idx) in indices.iter().enumerate() {
                    match idx {
                        Expr::Literal(crate::ast::Literal::Integer(n), idx_span) => {
                            if *n < 0 || *n >= width as i64 {
                                return Err(CompileError::type_error(
                                    format!("shuffle index {i} out of range: {n} (width {width})"),
                                    idx_span.clone(),
                                ));
                            }
                        }
                        _ => {
                            return Err(CompileError::type_error(
                                format!("shuffle index {i} must be integer literal"),
                                idx.span().clone(),
                            ))
                        }
                    }
                }
            }
            _ => {
                return Err(CompileError::type_error(
                    "shuffle second argument must be [index, ...] array literal",
                    args[1].span().clone(),
                ))
            }
        }
        Ok(vec_type)
    }

    pub(super) fn check_select(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 3 {
            return Err(CompileError::type_error(
                "select expects 3 arguments (mask, a, b)",
                span.clone(),
            ));
        }
        let mask_type = self.check_expr(&args[0], locals)?;
        let a_type = self.check_expr(&args[1], locals)?;
        let b_type = self.check_expr(&args[2], locals)?;
        types::unify_vector(&a_type, &b_type, span.clone())?;

        match (&mask_type, &a_type) {
            (
                Type::Vector {
                    elem: mask_elem,
                    width: mask_w,
                },
                Type::Vector { width: val_w, .. },
            ) if mask_elem.is_bool() && mask_w == val_w => Ok(a_type),
            _ => Err(CompileError::type_error(
                format!(
                    "select mask must be bool vector matching operand width, got {mask_type}. Use comparison operators (.>, .==) to create a mask"
                ),
                args[0].span().clone(),
            )),
        }
    }

    pub(super) fn check_gather(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "gather expects 2 arguments (ptr, indices)",
                span.clone(),
            ));
        }
        let ptr_type = self.check_expr(&args[0], locals)?;
        let idx_type = self.check_expr(&args[1], locals)?;
        let inner = match &ptr_type {
            Type::Pointer { inner, .. } if inner.is_numeric() => inner.clone(),
            Type::Pointer { .. } => {
                return Err(CompileError::type_error(
                    "gather expects pointer to numeric type",
                    args[0].span().clone(),
                ))
            }
            _ => {
                return Err(CompileError::type_error(
                    format!("gather expects a pointer, got {ptr_type}"),
                    args[0].span().clone(),
                ))
            }
        };
        let width = match &idx_type {
            Type::Vector { elem, width } if matches!(elem.as_ref(), Type::I32) => *width,
            _ => {
                return Err(CompileError::type_error(
                    format!("gather indices must be i32 vector, got {idx_type}"),
                    args[1].span().clone(),
                ))
            }
        };
        Ok(Type::Vector { elem: inner, width })
    }

    pub(super) fn check_scatter(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 3 {
            return Err(CompileError::type_error(
                "scatter expects 3 arguments (ptr, indices, values)",
                span.clone(),
            ));
        }
        let ptr_type = self.check_expr(&args[0], locals)?;
        let idx_type = self.check_expr(&args[1], locals)?;
        let val_type = self.check_expr(&args[2], locals)?;
        match &idx_type {
            Type::Vector { elem, .. } if matches!(elem.as_ref(), Type::I32) => {}
            _ => {
                return Err(CompileError::type_error(
                    format!("scatter indices must be i32 vector, got {idx_type}"),
                    args[1].span().clone(),
                ))
            }
        }
        match (ptr_type, val_type) {
            (
                Type::Pointer {
                    mutable: true,
                    inner,
                    ..
                },
                Type::Vector { elem, .. },
            ) => {
                if !types::types_compatible(&elem, &inner) {
                    return Err(CompileError::type_error(
                        format!("scatter mismatch: ptr to {inner}, val {elem}"),
                        span.clone(),
                    ));
                }
                Ok(Type::Void)
            }
            (Type::Pointer { mutable: false, .. }, _) => Err(CompileError::type_error(
                "scatter requires mutable pointer. Declare as *mut to allow writes",
                args[0].span().clone(),
            )),
            (_, _) => Err(CompileError::type_error(
                "scatter expects (mut ptr, indices, values)",
                span.clone(),
            )),
        }
    }

    pub(super) fn check_load_masked(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        type_hint: Option<&Type>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 3 {
            return Err(CompileError::type_error(
                "load_masked expects 3 arguments (ptr, offset, count)",
                span.clone(),
            ));
        }
        let ptr_type = self.check_expr(&args[0], locals)?;
        let idx_type = self.check_expr(&args[1], locals)?;
        let count_type = self.check_expr(&args[2], locals)?;

        if !idx_type.is_integer() {
            return Err(CompileError::type_error(
                "load_masked offset must be integer",
                args[1].span().clone(),
            ));
        }
        if !count_type.is_integer() {
            return Err(CompileError::type_error(
                "load_masked count must be integer",
                args[2].span().clone(),
            ));
        }

        let width = match type_hint {
            Some(Type::Vector { width, .. }) => *width,
            _ => 4,
        };

        match ptr_type {
            Type::Pointer { inner, .. } => {
                if inner.is_numeric() {
                    Ok(Type::Vector { elem: inner, width })
                } else {
                    Err(CompileError::type_error(
                        "load_masked expects pointer to numeric type",
                        args[0].span().clone(),
                    ))
                }
            }
            _ => Err(CompileError::type_error(
                format!("load_masked expects a pointer, got {ptr_type}"),
                args[0].span().clone(),
            )),
        }
    }

    pub(super) fn check_store_masked(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 4 {
            return Err(CompileError::type_error(
                "store_masked expects 4 arguments (ptr, offset, vector, count)",
                span.clone(),
            ));
        }
        let ptr_type = self.check_expr(&args[0], locals)?;
        let idx_type = self.check_expr(&args[1], locals)?;
        let val_type = self.check_expr(&args[2], locals)?;
        let count_type = self.check_expr(&args[3], locals)?;

        if !idx_type.is_integer() {
            return Err(CompileError::type_error(
                "store_masked offset must be integer",
                args[1].span().clone(),
            ));
        }
        if !count_type.is_integer() {
            return Err(CompileError::type_error(
                "store_masked count must be integer",
                args[3].span().clone(),
            ));
        }
        match (ptr_type, val_type) {
            (
                Type::Pointer {
                    mutable: true,
                    inner,
                    ..
                },
                Type::Vector { elem, .. },
            ) => {
                if !types::types_compatible(&elem, &inner) {
                    return Err(CompileError::type_error(
                        format!("store_masked mismatch: ptr to {inner}, val {elem}"),
                        span.clone(),
                    ));
                }
                Ok(Type::Void)
            }
            (Type::Pointer { mutable: false, .. }, _) => Err(CompileError::type_error(
                "store_masked requires mutable pointer. Declare as *mut to allow writes",
                args[0].span().clone(),
            )),
            (_, _) => Err(CompileError::type_error(
                "store_masked expects (mut ptr, offset, vector, count)",
                span.clone(),
            )),
        }
    }
}
