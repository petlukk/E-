mod check;
pub mod const_eval;
mod expr_check;
mod intrinsics;
mod intrinsics_conv;
mod intrinsics_simd;
pub mod types;

use std::collections::HashMap;

use crate::ast::{Literal, Param, Stmt};

pub use types::Type;

#[derive(Debug, Clone)]
pub(crate) struct FuncSig {
    pub params: Vec<Type>,
    pub return_type: Type,
}

pub struct TypeChecker {
    pub(crate) functions: HashMap<String, FuncSig>,
    pub(crate) structs: HashMap<String, Vec<(String, Type)>>,
    pub(crate) constants: HashMap<String, (Type, Literal)>,
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            structs: HashMap::new(),
            constants: HashMap::new(),
        }
    }

    pub fn check_program(&mut self, stmts: &[Stmt]) -> crate::error::Result<()> {
        // Register constants first
        for stmt in stmts {
            if let Stmt::Const {
                name,
                ty,
                value,
                span,
            } = stmt
            {
                if self.constants.contains_key(name) {
                    return Err(crate::error::CompileError::type_error(
                        format!("duplicate constant '{name}'"),
                        span.clone(),
                    ));
                }
                let declared = types::resolve_type(ty)?;
                // Verify the type is a scalar numeric type
                match &declared {
                    Type::I8
                    | Type::U8
                    | Type::I16
                    | Type::U16
                    | Type::I32
                    | Type::U32
                    | Type::I64
                    | Type::U64
                    | Type::F32
                    | Type::F64 => {}
                    _ => {
                        return Err(crate::error::CompileError::type_error(
                            format!("const type must be a scalar numeric type, got {declared}"),
                            ty.span().clone(),
                        ));
                    }
                }
                // Verify literal matches declared type
                match (&declared, value) {
                    (
                        Type::I8
                        | Type::U8
                        | Type::I16
                        | Type::U16
                        | Type::I32
                        | Type::U32
                        | Type::I64
                        | Type::U64,
                        Literal::Integer(_),
                    ) => {}
                    (Type::F32 | Type::F64, Literal::Float(_)) => {}
                    (Type::F32 | Type::F64, Literal::Integer(_)) => {}
                    _ => {
                        return Err(crate::error::CompileError::type_error(
                            format!("const value does not match type {declared}"),
                            span.clone(),
                        ));
                    }
                }
                self.constants
                    .insert(name.clone(), (declared, value.clone()));
            }
        }

        // Evaluate static assertions after constants are registered
        for stmt in stmts {
            if let Stmt::StaticAssert {
                condition,
                message,
                span,
            } = stmt
            {
                let val = self.eval_const_expr(condition)?;
                match val {
                    const_eval::ConstValue::Bool(true) => {}
                    const_eval::ConstValue::Bool(false) => {
                        return Err(crate::error::CompileError::type_error(
                            format!("static assertion failed: {message}"),
                            span.clone(),
                        ));
                    }
                    _ => {
                        return Err(crate::error::CompileError::type_error(
                            "static_assert condition must evaluate to a boolean",
                            span.clone(),
                        ));
                    }
                }
            }
        }

        for stmt in stmts {
            if let Stmt::Struct { name, fields, .. } = stmt {
                let typed_fields = fields
                    .iter()
                    .map(|f| {
                        let ty = types::resolve_type(&f.ty)?;
                        Ok((f.name.clone(), ty))
                    })
                    .collect::<Result<_, _>>()?;
                self.structs.insert(name.clone(), typed_fields);
            }
        }

        for stmt in stmts {
            if let Stmt::Function {
                name,
                params,
                return_type,
                ..
            } = stmt
            {
                let param_types: Vec<Type> = params
                    .iter()
                    .map(|p| types::resolve_type(&p.ty))
                    .collect::<Result<_, _>>()?;
                let ret = match return_type {
                    Some(ty) => types::resolve_type(ty)?,
                    None => Type::Void,
                };
                self.functions.insert(
                    name.clone(),
                    FuncSig {
                        params: param_types,
                        return_type: ret,
                    },
                );
            }
        }

        // Validate output annotations
        for stmt in stmts {
            if let Stmt::Function { params, .. } = stmt {
                self.check_output_annotations(params)?;
            }
        }

        for stmt in stmts {
            if let Stmt::Function {
                name,
                params,
                return_type,
                body,
                ..
            } = stmt
            {
                let expected_return = match return_type {
                    Some(ty) => types::resolve_type(ty)?,
                    None => Type::Void,
                };
                let mut locals = self.build_locals(params)?;
                self.check_body(body, &mut locals, &expected_return, name)?;
            }
        }

        Ok(())
    }

    fn check_output_annotations(&self, params: &[Param]) -> crate::error::Result<()> {
        let input_names: Vec<&str> = params
            .iter()
            .filter(|p| !p.output)
            .map(|p| p.name.as_str())
            .collect();
        let const_names: Vec<&str> = self.constants.keys().map(|k| k.as_str()).collect();

        for p in params {
            if !p.output {
                continue;
            }
            // `out` requires *mut pointer type
            let resolved = types::resolve_type(&p.ty)?;
            match &resolved {
                Type::Pointer { mutable: true, .. } => {}
                Type::Pointer { mutable: false, .. } => {
                    return Err(crate::error::CompileError::type_error(
                        format!(
                            "'out' parameter '{}' must be *mut pointer, got immutable pointer",
                            p.name
                        ),
                        p.span.clone(),
                    ));
                }
                _ => {
                    return Err(crate::error::CompileError::type_error(
                        format!(
                            "'out' parameter '{}' must be a *mut pointer type, got {}",
                            p.name, resolved
                        ),
                        p.span.clone(),
                    ));
                }
            }

            // Validate cap references
            if let Some(cap) = &p.cap {
                // Cap is an expression string — extract identifiers and check
                // each word-like token against known params/constants
                for word in cap.split_whitespace() {
                    if word
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_alphabetic() || c == '_')
                    {
                        let ident = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
                        if !ident.is_empty()
                            && !input_names.contains(&ident)
                            && !const_names.contains(&ident)
                        {
                            return Err(crate::error::CompileError::type_error(
                                format!(
                                    "cap expression references unknown identifier '{ident}' \
                                     (must be a preceding input parameter or constant)"
                                ),
                                p.span.clone(),
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn build_locals(
        &self,
        params: &[Param],
    ) -> crate::error::Result<HashMap<String, (Type, bool)>> {
        let mut locals = HashMap::new();
        for p in params {
            locals.insert(p.name.clone(), (types::resolve_type(&p.ty)?, false));
        }
        Ok(locals)
    }
}
