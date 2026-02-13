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
    Call { name: String, args: Vec<Expr> },
    Not(Box<Expr>),
    Index {
        object: Box<Expr>,
        index: Box<Expr>,
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
            Expr::Index { object, index } => write!(f, "{object}[{index}]"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeAnnotation {
    Named(String),
    Pointer {
        mutable: bool,
        inner: Box<TypeAnnotation>,
    },
}

impl fmt::Display for TypeAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeAnnotation::Named(name) => write!(f, "{name}"),
            TypeAnnotation::Pointer { mutable, inner } => {
                if *mutable {
                    write!(f, "*mut {inner}")
                } else {
                    write!(f, "*{inner}")
                }
            }
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
        }
    }
}
