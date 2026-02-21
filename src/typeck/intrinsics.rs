use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Position;

use super::types::{self, Type};
use super::TypeChecker;

impl TypeChecker {
    /// Returns Some(type) if the call is a known intrinsic, None if it's a user function.
    pub(super) fn check_intrinsic_call(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        type_hint: Option<&Type>,
    ) -> Option<crate::error::Result<Type>> {
        match name {
            "println" => Some(self.check_println(args, locals)),
            "splat" => Some(self.check_splat(args, locals, type_hint)),
            "load" => Some(self.check_load(args, locals, type_hint)),
            "store" => Some(self.check_store(args, locals)),
            "fma" => Some(self.check_fma(args, locals)),
            "reduce_add" | "reduce_max" | "reduce_min" => {
                Some(self.check_reduction(name, args, locals))
            }
            "shuffle" => Some(self.check_shuffle(args, locals)),
            "select" => Some(self.check_select(args, locals)),
            "widen_i8_f32x4" | "widen_u8_f32x4" => {
                Some(self.check_widen_i8_f32x4(name, args, locals))
            }
            "narrow_f32x4_i8" => Some(self.check_narrow_f32x4_i8(args, locals)),
            "maddubs_i16" => Some(self.check_maddubs_i16(args, locals)),
            "maddubs_i32" => Some(self.check_maddubs_i32(args, locals)),
            _ => None,
        }
    }

    fn check_println(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
    ) -> crate::error::Result<Type> {
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
        Ok(Type::Void)
    }

    fn check_splat(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        type_hint: Option<&Type>,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                "splat expects 1 argument",
                Position::default(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;
        let (width, hint_elem) = match type_hint {
            Some(Type::Vector { width, elem }) => (*width, Some(elem.as_ref())),
            _ => (4, None),
        };
        match arg_type {
            Type::FloatLiteral => {
                let elem = hint_elem
                    .filter(|e| e.is_float())
                    .cloned()
                    .unwrap_or(Type::F32);
                Ok(Type::Vector {
                    elem: Box::new(elem),
                    width,
                })
            }
            Type::IntLiteral => {
                let elem = hint_elem
                    .filter(|e| e.is_integer())
                    .cloned()
                    .unwrap_or(Type::I32);
                Ok(Type::Vector {
                    elem: Box::new(elem),
                    width,
                })
            }
            concrete if concrete.is_numeric() => Ok(Type::Vector {
                elem: Box::new(concrete),
                width,
            }),
            _ => Err(CompileError::type_error(
                format!("splat expects numeric, got {arg_type:?}"),
                Position::default(),
            )),
        }
    }

    fn check_load(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        type_hint: Option<&Type>,
    ) -> crate::error::Result<Type> {
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
                        Position::default(),
                    ))
                }
            }
            _ => Err(CompileError::type_error(
                "load expects pointer",
                Position::default(),
            )),
        }
    }

    fn check_store(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
    ) -> crate::error::Result<Type> {
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
                    ..
                },
                Type::Vector { elem, .. },
            ) => {
                if !types::types_compatible(&elem, &inner) {
                    return Err(CompileError::type_error(
                        format!("store mismatch: ptr to {inner:?}, val {elem:?}"),
                        Position::default(),
                    ));
                }
                Ok(Type::Void)
            }
            (Type::Pointer { mutable: false, .. }, _) => Err(CompileError::type_error(
                "store requires mutable pointer",
                Position::default(),
            )),
            (_, _) => Err(CompileError::type_error(
                "store expects (mut ptr, index, vector)",
                Position::default(),
            )),
        }
    }

    fn check_fma(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
    ) -> crate::error::Result<Type> {
        if args.len() != 3 {
            return Err(CompileError::type_error(
                "fma expects 3 arguments",
                Position::default(),
            ));
        }
        let t1 = self.check_expr(&args[0], locals)?;
        let t2 = self.check_expr(&args[1], locals)?;
        let t3 = self.check_expr(&args[2], locals)?;
        if !t1.is_vector() || !t2.is_vector() || !t3.is_vector() {
            return Err(CompileError::type_error(
                format!("fma expects vector arguments, got {t1:?}, {t2:?}, {t3:?}"),
                Position::default(),
            ));
        }
        types::unify_vector(&t1, &t2)?;
        types::unify_vector(&t1, &t3)?;
        Ok(t1)
    }

    fn check_reduction(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                format!("{name} expects 1 argument"),
                Position::default(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;
        match &arg_type {
            Type::Vector { elem, .. } => Ok(*elem.clone()),
            _ => Err(CompileError::type_error(
                format!("{name} expects vector argument, got {arg_type:?}"),
                Position::default(),
            )),
        }
    }

    fn check_shuffle(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "shuffle expects 2 arguments (vector, indices)",
                Position::default(),
            ));
        }
        let vec_type = self.check_expr(&args[0], locals)?;
        let width = match &vec_type {
            Type::Vector { width, .. } => *width,
            _ => {
                return Err(CompileError::type_error(
                    format!("shuffle first argument must be vector, got {vec_type:?}"),
                    Position::default(),
                ))
            }
        };

        match &args[1] {
            Expr::ArrayLiteral(indices) => {
                if indices.len() != width {
                    return Err(CompileError::type_error(
                        format!(
                            "shuffle indices length {} != vector width {width}",
                            indices.len()
                        ),
                        Position::default(),
                    ));
                }
                for (i, idx) in indices.iter().enumerate() {
                    match idx {
                        Expr::Literal(crate::ast::Literal::Integer(n)) => {
                            if *n < 0 || *n >= width as i64 {
                                return Err(CompileError::type_error(
                                    format!("shuffle index {i} out of range: {n} (width {width})"),
                                    Position::default(),
                                ));
                            }
                        }
                        _ => {
                            return Err(CompileError::type_error(
                                format!("shuffle index {i} must be integer literal"),
                                Position::default(),
                            ))
                        }
                    }
                }
            }
            _ => {
                return Err(CompileError::type_error(
                    "shuffle second argument must be [index, ...] array literal",
                    Position::default(),
                ))
            }
        }
        Ok(vec_type)
    }

    fn check_select(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
    ) -> crate::error::Result<Type> {
        if args.len() != 3 {
            return Err(CompileError::type_error(
                "select expects 3 arguments (mask, a, b)",
                Position::default(),
            ));
        }
        let mask_type = self.check_expr(&args[0], locals)?;
        let a_type = self.check_expr(&args[1], locals)?;
        let b_type = self.check_expr(&args[2], locals)?;
        types::unify_vector(&a_type, &b_type)?;

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
                    "select mask must be bool vector matching operand width, got {mask_type:?}"
                ),
                Position::default(),
            )),
        }
    }

    fn check_widen_i8_f32x4(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                format!("{name} expects 1 argument (i8x16 vector)"),
                Position::default(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;
        match &arg_type {
            Type::Vector { elem, width: 16 }
                if matches!(elem.as_ref(), Type::I8 | Type::U8) =>
            {
                Ok(Type::Vector {
                    elem: Box::new(Type::F32),
                    width: 4,
                })
            }
            _ => Err(CompileError::type_error(
                format!("{name} expects i8x16 or u8x16, got {arg_type:?}"),
                Position::default(),
            )),
        }
    }

    fn check_narrow_f32x4_i8(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                "narrow_f32x4_i8 expects 1 argument (f32x4 vector)",
                Position::default(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;
        match &arg_type {
            Type::Vector { elem, width: 4 } if matches!(elem.as_ref(), Type::F32) => {
                // Returns i8x16 with lower 4 elements set, upper 12 undefined
                Ok(Type::Vector {
                    elem: Box::new(Type::I8),
                    width: 16,
                })
            }
            _ => Err(CompileError::type_error(
                format!("narrow_f32x4_i8 expects f32x4, got {arg_type:?}"),
                Position::default(),
            )),
        }
    }

    fn check_maddubs_i16(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "maddubs_i16 expects 2 arguments: (u8x16, i8x16)",
                Position::default(),
            ));
        }
        let a = self.check_expr(&args[0], locals)?;
        let b = self.check_expr(&args[1], locals)?;
        match (&a, &b) {
            (
                Type::Vector { elem: ea, width: 16 },
                Type::Vector { elem: eb, width: 16 },
            ) if matches!(ea.as_ref(), Type::U8) && matches!(eb.as_ref(), Type::I8) => {
                Ok(Type::Vector { elem: Box::new(Type::I16), width: 8 })
            }
            _ => Err(CompileError::type_error(
                format!("maddubs_i16 expects (u8x16, i8x16), got ({a:?}, {b:?})"),
                Position::default(),
            )),
        }
    }

    fn check_maddubs_i32(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "maddubs_i32 expects 2 arguments: (u8x16, i8x16)",
                Position::default(),
            ));
        }
        let a = self.check_expr(&args[0], locals)?;
        let b = self.check_expr(&args[1], locals)?;
        match (&a, &b) {
            (
                Type::Vector { elem: ea, width: 16 },
                Type::Vector { elem: eb, width: 16 },
            ) if matches!(ea.as_ref(), Type::U8) && matches!(eb.as_ref(), Type::I8) => {
                Ok(Type::Vector { elem: Box::new(Type::I32), width: 4 })
            }
            _ => Err(CompileError::type_error(
                format!("maddubs_i32 expects (u8x16, i8x16), got ({a:?}, {b:?})"),
                Position::default(),
            )),
        }
    }
}
