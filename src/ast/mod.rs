use std::fmt;

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
    Literal(Literal),
    Variable(String),
    Binary(Box<Expr>, BinaryOp, Box<Expr>),
    Call {
        name: String,
        args: Vec<Expr>,
    },
    Not(Box<Expr>),
    Negate(Box<Expr>),
    Index {
        object: Box<Expr>,
        index: Box<Expr>,
    },
    Vector {
        elements: Vec<Expr>,
        ty: TypeAnnotation,
    },
    ArrayLiteral(Vec<Expr>),
    FieldAccess {
        object: Box<Expr>,
        field: String,
    },
    StructLiteral {
        name: String,
        fields: Vec<(String, Expr)>,
    },
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Literal(lit) => write!(f, "{lit}"),
            Expr::Variable(name) => write!(f, "{name}"),
            Expr::Binary(lhs, op, rhs) => write!(f, "({lhs} {op} {rhs})"),
            Expr::Call { name, args } => {
                write!(f, "{name}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
            Expr::Not(inner) => write!(f, "!{inner}"),
            Expr::Negate(inner) => write!(f, "-{inner}"),
            Expr::Index { object, index } => write!(f, "{object}[{index}]"),
            Expr::Vector { elements, ty } => {
                write!(f, "[")?;
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{elem}")?;
                }
                write!(f, "]{ty}")
            }
            Expr::ArrayLiteral(elements) => {
                write!(f, "[")?;
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{elem}")?;
                }
                write!(f, "]")
            }
            Expr::FieldAccess { object, field } => write!(f, "{object}.{field}"),
            Expr::StructLiteral { name, fields } => {
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
    Named(String),
    Pointer {
        mutable: bool,
        restrict: bool,
        inner: Box<TypeAnnotation>,
    },
    Vector {
        elem: Box<TypeAnnotation>,
        width: usize,
    },
}

impl fmt::Display for TypeAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeAnnotation::Named(name) => write!(f, "{name}"),
            TypeAnnotation::Pointer {
                mutable,
                restrict,
                inner,
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
            TypeAnnotation::Vector { elem, width } => write!(f, "{elem}x{width}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: TypeAnnotation,
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
    },
    Let {
        name: String,
        ty: TypeAnnotation,
        value: Expr,
        mutable: bool,
    },
    Assign {
        target: String,
        value: Expr,
    },
    IndexAssign {
        object: String,
        index: Expr,
        value: Expr,
    },
    Return(Option<Expr>),
    ExprStmt(Expr),
    If {
        condition: Expr,
        then_body: Vec<Stmt>,
        else_body: Option<Vec<Stmt>>,
    },
    While {
        condition: Expr,
        body: Vec<Stmt>,
    },
    Unroll {
        count: u32,
        body: Box<Stmt>,
    },
    ForEach {
        var: String,
        start: Expr,
        end: Expr,
        body: Vec<Stmt>,
    },
    Struct {
        name: String,
        fields: Vec<StructField>,
    },
    FieldAssign {
        object: Expr,
        field: String,
        value: Expr,
    },
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
            Stmt::Return(Some(expr)) => write!(f, "return {expr}"),
            Stmt::Return(None) => write!(f, "return"),
            Stmt::ExprStmt(expr) => write!(f, "{expr}"),
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
            Stmt::Struct { name, fields } => {
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
        }
    }
}
