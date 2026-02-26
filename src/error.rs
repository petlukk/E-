use crate::lexer::{Position, Span};
use std::fmt;

pub type Result<T> = std::result::Result<T, CompileError>;

#[derive(Debug, Clone)]
pub enum CompileError {
    LexError {
        message: String,
        position: Position,
    },
    ParseError {
        message: String,
        position: Position,
    },
    TypeError {
        message: String,
        span: Span,
    },
    CodeGenError {
        message: String,
        position: Option<Position>,
    },
}

impl CompileError {
    pub fn lex_error(message: impl Into<String>, position: Position) -> Self {
        Self::LexError {
            message: message.into(),
            position,
        }
    }

    pub fn parse_error(message: impl Into<String>, position: Position) -> Self {
        Self::ParseError {
            message: message.into(),
            position,
        }
    }

    pub fn type_error(message: impl Into<String>, span: Span) -> Self {
        Self::TypeError {
            message: message.into(),
            span,
        }
    }

    pub fn codegen_error(message: impl Into<String>) -> Self {
        Self::CodeGenError {
            message: message.into(),
            position: None,
        }
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompileError::LexError { message, position } => {
                write!(
                    f,
                    "error[lex] {}:{}: {}",
                    position.line, position.column, message
                )
            }
            CompileError::ParseError { message, position } => {
                write!(
                    f,
                    "error[parse] {}:{}: {}",
                    position.line, position.column, message
                )
            }
            CompileError::TypeError { message, span } => {
                write!(
                    f,
                    "error[type] {}:{}: {}",
                    span.start.line, span.start.column, message
                )
            }
            CompileError::CodeGenError { message, position } => {
                if let Some(pos) = position {
                    write!(f, "error[codegen] {}:{}: {}", pos.line, pos.column, message)
                } else {
                    write!(f, "error[codegen]: {message}")
                }
            }
        }
    }
}

impl std::error::Error for CompileError {}
