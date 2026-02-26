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

/// Format a compile error with source context showing the relevant line and caret.
///
/// Output format:
/// ```text
/// kernel.ea:14:23  error[type]: cannot assign f32 to 'x' of type i32
///     let y: f32 = x + 1
///                  ^
/// ```
pub fn format_with_source(error: &CompileError, filename: &str, source: &str) -> String {
    let (line, col, kind, message) = match error {
        CompileError::LexError { message, position } => {
            (position.line, position.column, "lex", message.as_str())
        }
        CompileError::ParseError { message, position } => {
            (position.line, position.column, "parse", message.as_str())
        }
        CompileError::TypeError { message, span } => {
            (span.start.line, span.start.column, "type", message.as_str())
        }
        CompileError::CodeGenError { message, position } => {
            if let Some(pos) = position {
                (pos.line, pos.column, "codegen", message.as_str())
            } else {
                return format!("error[codegen]: {message}");
            }
        }
    };

    let header = format!("{filename}:{line}:{col}  error[{kind}]: {message}");

    let lines: Vec<&str> = source.lines().collect();
    if line == 0 || line > lines.len() {
        return header;
    }

    let source_line = lines[line - 1];
    let caret_col = if col > 0 { col - 1 } else { 0 };
    let caret = format!("{:>width$}^", "", width = caret_col + 4);

    format!("{header}\n    {source_line}\n{caret}")
}
