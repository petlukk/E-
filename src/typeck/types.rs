use crate::ast::TypeAnnotation;
use crate::error::CompileError;
use crate::lexer::Position;

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    I8,
    U8,
    I16,
    U16,
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
        restrict: bool,
        inner: Box<Type>,
    },
    Vector {
        elem: Box<Type>,
        width: usize,
    },
    Struct(String),
}

impl Type {
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            Type::I8 | Type::U8 | Type::I16 | Type::U16 | Type::I32 | Type::I64 | Type::IntLiteral
        )
    }

    pub fn is_unsigned_integer(&self) -> bool {
        matches!(self, Type::U8 | Type::U16)
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

    pub fn is_vector(&self) -> bool {
        matches!(self, Type::Vector { .. })
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
        (
            Type::Vector {
                elem: a_elem,
                width: a_width,
            },
            Type::Vector {
                elem: e_elem,
                width: e_width,
            },
        ) => a_width == e_width && types_compatible(a_elem, e_elem),
        (
            Type::Pointer {
                mutable: a_mut,
                inner: a_inner,
                ..
            },
            Type::Pointer {
                mutable: e_mut,
                inner: e_inner,
                ..
            },
        ) => a_mut == e_mut && types_compatible(a_inner, e_inner),
        (Type::Struct(a), Type::Struct(b)) => a == b,
        _ => false,
    }
}

pub fn unify_vector(left: &Type, right: &Type) -> crate::error::Result<Type> {
    match (left, right) {
        (
            Type::Vector {
                elem: l_elem,
                width: l_width,
            },
            Type::Vector {
                elem: r_elem,
                width: r_width,
            },
        ) => {
            if l_width != r_width {
                return Err(CompileError::type_error(
                    format!("vector width mismatch: {l_width} vs {r_width}"),
                    Position::default(),
                ));
            }
            if !types_compatible(l_elem, r_elem) {
                return Err(CompileError::type_error(
                    format!("vector element type mismatch: {l_elem:?} vs {r_elem:?}"),
                    Position::default(),
                ));
            }
            Ok(left.clone())
        }
        _ => Err(CompileError::type_error(
            format!("binary vector operations require vector operands, got {left:?} and {right:?}"),
            Position::default(),
        )),
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

/// Returns true if the type is an unsigned integer.
pub fn is_unsigned(ty: &Type) -> bool {
    matches!(ty, Type::U8 | Type::U16)
}

pub fn resolve_type(ty: &TypeAnnotation) -> crate::error::Result<Type> {
    match ty {
        TypeAnnotation::Named(name) => match name.as_str() {
            "i8" => Ok(Type::I8),
            "u8" => Ok(Type::U8),
            "i16" => Ok(Type::I16),
            "u16" => Ok(Type::U16),
            "i32" => Ok(Type::I32),
            "i64" => Ok(Type::I64),
            "f32" => Ok(Type::F32),
            "f64" => Ok(Type::F64),
            "bool" => Ok(Type::Bool),
            other => Ok(Type::Struct(other.to_string())),
        },
        TypeAnnotation::Pointer {
            mutable,
            restrict,
            inner,
        } => {
            let inner_type = resolve_type(inner)?;
            Ok(Type::Pointer {
                mutable: *mutable,
                restrict: *restrict,
                inner: Box::new(inner_type),
            })
        }
        TypeAnnotation::Vector { elem, width } => {
            let elem_type = resolve_type(elem)?;
            Ok(Type::Vector {
                elem: Box::new(elem_type),
                width: *width,
            })
        }
    }
}
