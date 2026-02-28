use std::fmt;

use crate::lexer::Span;

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    Equal,
    NotEqual,
    And,
    Or,
    AddDot,
    SubDot,
    MulDot,
    DivDot,
    LessDot,
    GreaterDot,
    LessEqualDot,
    GreaterEqualDot,
    EqualDot,
    NotEqualDot,
    AndDot,
    OrDot,
    XorDot,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Subtract => write!(f, "-"),
            BinaryOp::Multiply => write!(f, "*"),
            BinaryOp::Divide => write!(f, "/"),
            BinaryOp::Modulo => write!(f, "%"),
            BinaryOp::Less => write!(f, "<"),
            BinaryOp::Greater => write!(f, ">"),
            BinaryOp::LessEqual => write!(f, "<="),
            BinaryOp::GreaterEqual => write!(f, ">="),
            BinaryOp::Equal => write!(f, "=="),
            BinaryOp::NotEqual => write!(f, "!="),
            BinaryOp::And => write!(f, "&&"),
            BinaryOp::Or => write!(f, "||"),
            BinaryOp::AddDot => write!(f, ".+"),
            BinaryOp::SubDot => write!(f, ".-"),
            BinaryOp::MulDot => write!(f, ".*"),
            BinaryOp::DivDot => write!(f, "./"),
            BinaryOp::LessDot => write!(f, ".<"),
            BinaryOp::GreaterDot => write!(f, ".>"),
            BinaryOp::LessEqualDot => write!(f, ".<="),
            BinaryOp::GreaterEqualDot => write!(f, ".>="),
            BinaryOp::EqualDot => write!(f, ".=="),
            BinaryOp::NotEqualDot => write!(f, ".!="),
            BinaryOp::AndDot => write!(f, ".&"),
            BinaryOp::OrDot => write!(f, ".|"),
            BinaryOp::XorDot => write!(f, ".^"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    StringLit(String),
    Bool(bool),
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Integer(n) => write!(f, "{n}"),
            Literal::Float(n) => write!(f, "{n}"),
            Literal::StringLit(s) => write!(f, "\"{s}\""),
            Literal::Bool(b) => write!(f, "{b}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Literal, Span),
    Variable(String, Span),
    Binary(Box<Expr>, BinaryOp, Box<Expr>, Span),
    Call {
        name: String,
        args: Vec<Expr>,
        span: Span,
    },
    Not(Box<Expr>, Span),
    Negate(Box<Expr>, Span),
    Index {
        object: Box<Expr>,
        index: Box<Expr>,
        span: Span,
    },
    Vector {
        elements: Vec<Expr>,
        ty: TypeAnnotation,
        span: Span,
    },
    ArrayLiteral(Vec<Expr>, Span),
    FieldAccess {
        object: Box<Expr>,
        field: String,
        span: Span,
    },
    StructLiteral {
        name: String,
        fields: Vec<(String, Expr)>,
        span: Span,
    },
}

impl Expr {
    pub fn span(&self) -> &Span {
        match self {
            Expr::Literal(_, span) => span,
            Expr::Variable(_, span) => span,
            Expr::Binary(_, _, _, span) => span,
            Expr::Call { span, .. } => span,
            Expr::Not(_, span) => span,
            Expr::Negate(_, span) => span,
            Expr::Index { span, .. } => span,
            Expr::Vector { span, .. } => span,
            Expr::ArrayLiteral(_, span) => span,
            Expr::FieldAccess { span, .. } => span,
            Expr::StructLiteral { span, .. } => span,
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Literal(lit, _) => write!(f, "{lit}"),
            Expr::Variable(name, _) => write!(f, "{name}"),
            Expr::Binary(lhs, op, rhs, _) => write!(f, "({lhs} {op} {rhs})"),
            Expr::Call { name, args, .. } => {
                write!(f, "{name}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
            Expr::Not(inner, _) => write!(f, "!{inner}"),
            Expr::Negate(inner, _) => write!(f, "-{inner}"),
            Expr::Index { object, index, .. } => write!(f, "{object}[{index}]"),
            Expr::Vector { elements, ty, .. } => {
                write!(f, "[")?;
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{elem}")?;
                }
                write!(f, "]{ty}")
            }
            Expr::ArrayLiteral(elements, _) => {
                write!(f, "[")?;
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{elem}")?;
                }
                write!(f, "]")
            }
            Expr::FieldAccess { object, field, .. } => write!(f, "{object}.{field}"),
            Expr::StructLiteral { name, fields, .. } => {
                write!(f, "{name} {{ ")?;
                for (i, (fname, fval)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{fname}: {fval}")?;
                }
                write!(f, " }}")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeAnnotation {
    Named(String, Span),
    Pointer {
        mutable: bool,
        restrict: bool,
        inner: Box<TypeAnnotation>,
        span: Span,
    },
    Vector {
        elem: Box<TypeAnnotation>,
        width: usize,
        span: Span,
    },
}

impl TypeAnnotation {
    pub fn span(&self) -> &Span {
        match self {
            TypeAnnotation::Named(_, span) => span,
            TypeAnnotation::Pointer { span, .. } => span,
            TypeAnnotation::Vector { span, .. } => span,
        }
    }
}

impl fmt::Display for TypeAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeAnnotation::Named(name, _) => write!(f, "{name}"),
            TypeAnnotation::Pointer {
                mutable,
                restrict,
                inner,
                ..
            } => {
                if *restrict && *mutable {
                    write!(f, "*restrict mut {inner}")
                } else if *restrict {
                    write!(f, "*restrict {inner}")
                } else if *mutable {
                    write!(f, "*mut {inner}")
                } else {
                    write!(f, "*{inner}")
                }
            }
            TypeAnnotation::Vector { elem, width, .. } => write!(f, "{elem}x{width}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: TypeAnnotation,
    pub span: Span,
}

impl fmt::Display for Param {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.ty)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructField {
    pub name: String,
    pub ty: TypeAnnotation,
}

impl fmt::Display for StructField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.ty)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Function {
        name: String,
        params: Vec<Param>,
        return_type: Option<TypeAnnotation>,
        body: Vec<Stmt>,
        export: bool,
        span: Span,
    },
    Let {
        name: String,
        ty: TypeAnnotation,
        value: Expr,
        mutable: bool,
        span: Span,
    },
    Assign {
        target: String,
        value: Expr,
        span: Span,
    },
    IndexAssign {
        object: String,
        index: Expr,
        value: Expr,
        span: Span,
    },
    Return(Option<Expr>, Span),
    ExprStmt(Expr, Span),
    If {
        condition: Expr,
        then_body: Vec<Stmt>,
        else_body: Option<Vec<Stmt>>,
        span: Span,
    },
    While {
        condition: Expr,
        body: Vec<Stmt>,
        span: Span,
    },
    Unroll {
        count: u32,
        body: Box<Stmt>,
        span: Span,
    },
    ForEach {
        var: String,
        start: Expr,
        end: Expr,
        body: Vec<Stmt>,
        span: Span,
    },
    Struct {
        name: String,
        fields: Vec<StructField>,
        span: Span,
    },
    FieldAssign {
        object: Expr,
        field: String,
        value: Expr,
        span: Span,
    },
    Const {
        name: String,
        ty: TypeAnnotation,
        value: Literal,
        span: Span,
    },
    Kernel {
        name: String,
        params: Vec<Param>,
        range_var: String,
        range_bound: String,
        step: u32,
        tail: Option<TailStrategy>,
        tail_body: Option<Vec<Stmt>>,
        body: Vec<Stmt>,
        export: bool,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TailStrategy {
    Scalar,
    Mask,
    Pad,
}

impl Stmt {
    pub fn span(&self) -> &Span {
        match self {
            Stmt::Function { span, .. } => span,
            Stmt::Let { span, .. } => span,
            Stmt::Assign { span, .. } => span,
            Stmt::IndexAssign { span, .. } => span,
            Stmt::Return(_, span) => span,
            Stmt::ExprStmt(_, span) => span,
            Stmt::If { span, .. } => span,
            Stmt::While { span, .. } => span,
            Stmt::Unroll { span, .. } => span,
            Stmt::ForEach { span, .. } => span,
            Stmt::Struct { span, .. } => span,
            Stmt::FieldAssign { span, .. } => span,
            Stmt::Const { span, .. } => span,
            Stmt::Kernel { span, .. } => span,
        }
    }
}

impl fmt::Display for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Stmt::Function {
                name,
                params,
                return_type,
                export,
                ..
            } => {
                if *export {
                    write!(f, "export ")?;
                }
                write!(f, "func {name}(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{p}")?;
                }
                write!(f, ")")?;
                if let Some(ret) = return_type {
                    write!(f, " -> {ret}")?;
                }
                write!(f, " {{ ... }}")
            }
            Stmt::Let {
                name, ty, mutable, ..
            } => {
                if *mutable {
                    write!(f, "let mut {name}: {ty} = ...")
                } else {
                    write!(f, "let {name}: {ty} = ...")
                }
            }
            Stmt::Assign { target, .. } => write!(f, "{target} = ..."),
            Stmt::IndexAssign { object, index, .. } => write!(f, "{object}[{index}] = ..."),
            Stmt::Return(Some(expr), _) => write!(f, "return {expr}"),
            Stmt::Return(None, _) => write!(f, "return"),
            Stmt::ExprStmt(expr, _) => write!(f, "{expr}"),
            Stmt::If { else_body, .. } => {
                if else_body.is_some() {
                    write!(f, "if ... {{ ... }} else {{ ... }}")
                } else {
                    write!(f, "if ... {{ ... }}")
                }
            }
            Stmt::While { .. } => write!(f, "while ... {{ ... }}"),
            Stmt::Unroll { count, .. } => write!(f, "unroll({count}) {{ ... }}"),
            Stmt::ForEach { var, .. } => write!(f, "foreach ({var} in ...) {{ ... }}"),
            Stmt::Struct { name, fields, .. } => {
                write!(f, "struct {name} {{ ")?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{field}")?;
                }
                write!(f, " }}")
            }
            Stmt::FieldAssign { object, field, .. } => write!(f, "{object}.{field} = ..."),
            Stmt::Const {
                name, ty, value, ..
            } => write!(f, "const {name}: {ty} = {value}"),
            Stmt::Kernel {
                name,
                range_var,
                range_bound,
                step,
                ..
            } => write!(
                f,
                "kernel {name}(...) over {range_var} in {range_bound} step {step}"
            ),
        }
    }
}

/// Returns the names of all exported functions in the given statements.
pub fn exported_function_names(stmts: &[Stmt]) -> Vec<&str> {
    stmts
        .iter()
        .filter_map(|s| match s {
            Stmt::Function {
                name, export: true, ..
            } => Some(name.as_str()),
            _ => None,
        })
        .collect()
}
