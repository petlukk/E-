mod check;
pub mod types;

use std::collections::HashMap;

use crate::ast::{Param, Stmt};

pub use types::Type;

#[derive(Debug, Clone)]
pub(crate) struct FuncSig {
    pub params: Vec<Type>,
    pub return_type: Type,
}

pub struct TypeChecker {
    pub(crate) functions: HashMap<String, FuncSig>,
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
        }
    }

    pub fn check_program(&mut self, stmts: &[Stmt]) -> crate::error::Result<()> {
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
