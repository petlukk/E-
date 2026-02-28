use crate::ast::{BinaryOp, Expr, Literal};
use crate::error::CompileError;

use super::TypeChecker;

#[derive(Debug, Clone)]
pub enum ConstValue {
    Integer(i64),
    Float(f64),
    Bool(bool),
}

impl TypeChecker {
    pub fn eval_const_expr(&self, expr: &Expr) -> crate::error::Result<ConstValue> {
        match expr {
            Expr::Literal(Literal::Integer(n), _) => Ok(ConstValue::Integer(*n)),
            Expr::Literal(Literal::Float(n), _) => Ok(ConstValue::Float(*n)),
            Expr::Literal(Literal::Bool(b), _) => Ok(ConstValue::Bool(*b)),
            Expr::Literal(Literal::StringLit(_), span) => Err(CompileError::type_error(
                "string literals cannot appear in static_assert conditions",
                span.clone(),
            )),
            Expr::Variable(name, span) => {
                if let Some((_ty, lit)) = self.constants.get(name) {
                    match lit {
                        Literal::Integer(n) => Ok(ConstValue::Integer(*n)),
                        Literal::Float(n) => Ok(ConstValue::Float(*n)),
                        Literal::Bool(b) => Ok(ConstValue::Bool(*b)),
                        Literal::StringLit(_) => Err(CompileError::type_error(
                            format!("constant '{name}' is a string, not usable in static_assert"),
                            span.clone(),
                        )),
                    }
                } else {
                    Err(CompileError::type_error(
                        format!("'{name}' is not a compile-time constant"),
                        span.clone(),
                    ))
                }
            }
            Expr::Negate(inner, span) => {
                let val = self.eval_const_expr(inner)?;
                match val {
                    ConstValue::Integer(n) => Ok(ConstValue::Integer(-n)),
                    ConstValue::Float(n) => Ok(ConstValue::Float(-n)),
                    ConstValue::Bool(_) => Err(CompileError::type_error(
                        "cannot negate a boolean in constant expression",
                        span.clone(),
                    )),
                }
            }
            Expr::Not(inner, span) => {
                let val = self.eval_const_expr(inner)?;
                match val {
                    ConstValue::Bool(b) => Ok(ConstValue::Bool(!b)),
                    _ => Err(CompileError::type_error(
                        "logical not (!) requires a boolean in constant expression",
                        span.clone(),
                    )),
                }
            }
            Expr::Binary(lhs, op, rhs, span) => {
                let l = self.eval_const_expr(lhs)?;
                let r = self.eval_const_expr(rhs)?;
                eval_binary(l, op, r, span)
            }
            _ => Err(CompileError::type_error(
                "expression is not a compile-time constant",
                expr.span().clone(),
            )),
        }
    }
}

fn eval_binary(
    l: ConstValue,
    op: &BinaryOp,
    r: ConstValue,
    span: &crate::lexer::Span,
) -> crate::error::Result<ConstValue> {
    match (l, r) {
        (ConstValue::Integer(a), ConstValue::Integer(b)) => eval_int_binary(a, op, b, span),
        (ConstValue::Float(a), ConstValue::Float(b)) => eval_float_binary(a, op, b, span),
        (ConstValue::Integer(a), ConstValue::Float(b)) => eval_float_binary(a as f64, op, b, span),
        (ConstValue::Float(a), ConstValue::Integer(b)) => eval_float_binary(a, op, b as f64, span),
        (ConstValue::Bool(a), ConstValue::Bool(b)) => match op {
            BinaryOp::And => Ok(ConstValue::Bool(a && b)),
            BinaryOp::Or => Ok(ConstValue::Bool(a || b)),
            BinaryOp::Equal => Ok(ConstValue::Bool(a == b)),
            BinaryOp::NotEqual => Ok(ConstValue::Bool(a != b)),
            _ => Err(CompileError::type_error(
                format!("operator {op} not supported on booleans in constant expression"),
                span.clone(),
            )),
        },
        _ => Err(CompileError::type_error(
            "type mismatch in constant expression",
            span.clone(),
        )),
    }
}

fn eval_int_binary(
    a: i64,
    op: &BinaryOp,
    b: i64,
    span: &crate::lexer::Span,
) -> crate::error::Result<ConstValue> {
    match op {
        BinaryOp::Add => Ok(ConstValue::Integer(a + b)),
        BinaryOp::Subtract => Ok(ConstValue::Integer(a - b)),
        BinaryOp::Multiply => Ok(ConstValue::Integer(a * b)),
        BinaryOp::Divide => {
            if b == 0 {
                return Err(CompileError::type_error(
                    "division by zero in constant expression",
                    span.clone(),
                ));
            }
            Ok(ConstValue::Integer(a / b))
        }
        BinaryOp::Modulo => {
            if b == 0 {
                return Err(CompileError::type_error(
                    "modulo by zero in constant expression",
                    span.clone(),
                ));
            }
            Ok(ConstValue::Integer(a % b))
        }
        BinaryOp::Equal => Ok(ConstValue::Bool(a == b)),
        BinaryOp::NotEqual => Ok(ConstValue::Bool(a != b)),
        BinaryOp::Less => Ok(ConstValue::Bool(a < b)),
        BinaryOp::Greater => Ok(ConstValue::Bool(a > b)),
        BinaryOp::LessEqual => Ok(ConstValue::Bool(a <= b)),
        BinaryOp::GreaterEqual => Ok(ConstValue::Bool(a >= b)),
        BinaryOp::And => Ok(ConstValue::Bool(a != 0 && b != 0)),
        BinaryOp::Or => Ok(ConstValue::Bool(a != 0 || b != 0)),
        _ => Err(CompileError::type_error(
            format!("operator {op} not supported in constant expression"),
            span.clone(),
        )),
    }
}

fn eval_float_binary(
    a: f64,
    op: &BinaryOp,
    b: f64,
    span: &crate::lexer::Span,
) -> crate::error::Result<ConstValue> {
    match op {
        BinaryOp::Add => Ok(ConstValue::Float(a + b)),
        BinaryOp::Subtract => Ok(ConstValue::Float(a - b)),
        BinaryOp::Multiply => Ok(ConstValue::Float(a * b)),
        BinaryOp::Divide => Ok(ConstValue::Float(a / b)),
        BinaryOp::Equal => Ok(ConstValue::Bool(a == b)),
        BinaryOp::NotEqual => Ok(ConstValue::Bool(a != b)),
        BinaryOp::Less => Ok(ConstValue::Bool(a < b)),
        BinaryOp::Greater => Ok(ConstValue::Bool(a > b)),
        BinaryOp::LessEqual => Ok(ConstValue::Bool(a <= b)),
        BinaryOp::GreaterEqual => Ok(ConstValue::Bool(a >= b)),
        _ => Err(CompileError::type_error(
            format!("operator {op} not supported on floats in constant expression"),
            span.clone(),
        )),
    }
}
