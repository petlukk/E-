use crate::ast::TypeAnnotation;
use crate::error::CompileError;
use crate::lexer::Position;

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    I32,
    I64,
    F32,
    F64,
    Bool,
    IntLiteral,
    FloatLiteral,
    String,
    Void,
    Pointer {
        mutable: bool,
        inner: Box<Type>,
    },
}

impl Type {
    pub fn is_integer(&self) -> bool {
        matches!(self, Type::I32 | Type::I64 | Type::IntLiteral)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Type::F32 | Type::F64 | Type::FloatLiteral)
    }

    pub fn is_numeric(&self) -> bool {
        self.is_integer() || self.is_float()
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, Type::Bool)
    }

    pub fn is_pointer(&self) -> bool {
        matches!(self, Type::Pointer { .. })
    }

    pub fn pointee(&self) -> Option<&Type> {
        match self {
            Type::Pointer { inner, .. } => Some(inner),
            _ => None,
        }
    }
}

pub fn types_compatible(actual: &Type, expected: &Type) -> bool {
    if actual == expected {
        return true;
    }
    match (actual, expected) {
        (Type::IntLiteral, t) if t.is_integer() => true,
        (Type::FloatLiteral, t) if t.is_float() => true,
        _ => false,
    }
}

pub fn unify_numeric(left: &Type, right: &Type) -> crate::error::Result<Type> {
    if !left.is_numeric() || !right.is_numeric() {
        return Err(CompileError::type_error(
            format!("binary operations require numeric operands, got {left:?} and {right:?}"),
            Position::default(),
        ));
    }
    if left.is_integer() != right.is_integer() {
        return Err(CompileError::type_error(
            format!("cannot mix integer and float in binary operation: {left:?} and {right:?}"),
            Position::default(),
        ));
    }
    match (left, right) {
        (Type::IntLiteral, Type::IntLiteral) => Ok(Type::I32),
        (Type::FloatLiteral, Type::FloatLiteral) => Ok(Type::F64),
        (Type::IntLiteral, concrete) | (concrete, Type::IntLiteral) => Ok(concrete.clone()),
        (Type::FloatLiteral, concrete) | (concrete, Type::FloatLiteral) => Ok(concrete.clone()),
        (a, b) if a == b => Ok(a.clone()),
        _ => Err(CompileError::type_error(
            format!("mismatched types in binary operation: {left:?} and {right:?}"),
            Position::default(),
        )),
    }
}

pub fn resolve_type(ty: &TypeAnnotation) -> crate::error::Result<Type> {
    match ty {
        TypeAnnotation::Named(name) => match name.as_str() {
            "i32" => Ok(Type::I32),
            "i64" => Ok(Type::I64),
            "f32" => Ok(Type::F32),
            "f64" => Ok(Type::F64),
            "bool" => Ok(Type::Bool),
            other => Err(CompileError::type_error(
                format!("unknown type '{other}'"),
                Position::default(),
            )),
        },
        TypeAnnotation::Pointer { mutable, inner } => {
            let inner_type = resolve_type(inner)?;
            Ok(Type::Pointer {
                mutable: *mutable,
                inner: Box::new(inner_type),
            })
        }
    }
}
