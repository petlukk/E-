use crate::ast::{BinaryOp, Expr, Literal};
use crate::error::CompileError;
use crate::lexer::TokenKind;

use super::Parser;

impl Parser {
    pub(super) fn expression(&mut self) -> crate::error::Result<Expr> {
        self.logical_or()
    }

    fn logical_or(&mut self) -> crate::error::Result<Expr> {
        let mut left = self.logical_and()?;
        while self.check(TokenKind::PipePipe) {
            self.advance();
            let right = self.logical_and()?;
            left = Expr::Binary(Box::new(left), BinaryOp::Or, Box::new(right));
        }
        Ok(left)
    }

    fn logical_and(&mut self) -> crate::error::Result<Expr> {
        let mut left = self.comparison()?;
        while self.check(TokenKind::AmpAmp) {
            self.advance();
            let right = self.comparison()?;
            left = Expr::Binary(Box::new(left), BinaryOp::And, Box::new(right));
        }
        Ok(left)
    }

    fn comparison(&mut self) -> crate::error::Result<Expr> {
        let left = self.additive()?;
        let op = if self.check(TokenKind::LessEqual) {
            Some(BinaryOp::LessEqual)
        } else if self.check(TokenKind::GreaterEqual) {
            Some(BinaryOp::GreaterEqual)
        } else if self.check(TokenKind::Less) {
            Some(BinaryOp::Less)
        } else if self.check(TokenKind::Greater) {
            Some(BinaryOp::Greater)
        } else if self.check(TokenKind::EqualEqual) {
            Some(BinaryOp::Equal)
        } else if self.check(TokenKind::BangEqual) {
            Some(BinaryOp::NotEqual)
        } else if self.check(TokenKind::LessEqualDot) {
            Some(BinaryOp::LessEqualDot)
        } else if self.check(TokenKind::GreaterEqualDot) {
            Some(BinaryOp::GreaterEqualDot)
        } else if self.check(TokenKind::LessDot) {
            Some(BinaryOp::LessDot)
        } else if self.check(TokenKind::GreaterDot) {
            Some(BinaryOp::GreaterDot)
        } else if self.check(TokenKind::EqualEqualDot) {
            Some(BinaryOp::EqualDot)
        } else if self.check(TokenKind::BangEqualDot) {
            Some(BinaryOp::NotEqualDot)
        } else {
            None
        };
        if let Some(op) = op {
            self.advance();
            let right = self.additive()?;
            Ok(Expr::Binary(Box::new(left), op, Box::new(right)))
        } else {
            Ok(left)
        }
    }

    fn additive(&mut self) -> crate::error::Result<Expr> {
        let mut left = self.multiplicative()?;
        while self.check(TokenKind::Plus)
            || self.check(TokenKind::Minus)
            || self.check(TokenKind::PlusDot)
            || self.check(TokenKind::MinusDot)
        {
            let op = if self.check(TokenKind::Plus) {
                BinaryOp::Add
            } else if self.check(TokenKind::Minus) {
                BinaryOp::Subtract
            } else if self.check(TokenKind::PlusDot) {
                BinaryOp::AddDot
            } else {
                BinaryOp::SubDot
            };
            self.advance();
            let right = self.multiplicative()?;
            left = Expr::Binary(Box::new(left), op, Box::new(right));
        }
        Ok(left)
    }

    fn multiplicative(&mut self) -> crate::error::Result<Expr> {
        let mut left = self.unary()?;
        while self.check(TokenKind::Star)
            || self.check(TokenKind::Slash)
            || self.check(TokenKind::Percent)
            || self.check(TokenKind::StarDot)
            || self.check(TokenKind::SlashDot)
        {
            let op = if self.check(TokenKind::Star) {
                BinaryOp::Multiply
            } else if self.check(TokenKind::Slash) {
                BinaryOp::Divide
            } else if self.check(TokenKind::Percent) {
                BinaryOp::Modulo
            } else if self.check(TokenKind::StarDot) {
                BinaryOp::MulDot
            } else {
                BinaryOp::DivDot
            };
            self.advance();
            let right = self.unary()?;
            left = Expr::Binary(Box::new(left), op, Box::new(right));
        }
        Ok(left)
    }

    fn unary(&mut self) -> crate::error::Result<Expr> {
        if self.check(TokenKind::Bang) {
            self.advance();
            let inner = self.unary()?;
            return Ok(Expr::Not(Box::new(inner)));
        }
        self.primary()
    }

    fn primary(&mut self) -> crate::error::Result<Expr> {
        // Unary minus for numeric literals
        if self.check(TokenKind::Minus) {
            if self.peek_next_kind() == Some(&TokenKind::IntLiteral) {
                self.advance(); // consume '-'
                let token = self.advance().clone();
                let value: i64 = token.lexeme.parse::<i64>().map_err(|_| {
                    CompileError::parse_error(
                        format!("invalid integer literal: {}", token.lexeme),
                        token.position.clone(),
                    )
                })?;
                return Ok(Expr::Literal(Literal::Integer(-value)));
            }
            if self.peek_next_kind() == Some(&TokenKind::FloatLiteral) {
                self.advance(); // consume '-'
                let token = self.advance().clone();
                let value: f64 = token.lexeme.parse().map_err(|_| {
                    CompileError::parse_error(
                        format!("invalid float literal: {}", token.lexeme),
                        token.position.clone(),
                    )
                })?;
                return Ok(Expr::Literal(Literal::Float(-value)));
            }
        }

        if self.check(TokenKind::IntLiteral) {
            let token = self.advance().clone();
            let value: i64 = token.lexeme.parse().map_err(|_| {
                CompileError::parse_error(
                    format!("invalid integer literal: {}", token.lexeme),
                    token.position.clone(),
                )
            })?;
            return Ok(Expr::Literal(Literal::Integer(value)));
        }

        if self.check(TokenKind::FloatLiteral) {
            let token = self.advance().clone();
            let value: f64 = token.lexeme.parse().map_err(|_| {
                CompileError::parse_error(
                    format!("invalid float literal: {}", token.lexeme),
                    token.position.clone(),
                )
            })?;
            return Ok(Expr::Literal(Literal::Float(value)));
        }

        if self.check(TokenKind::StringLiteral) {
            let token = self.advance().clone();
            let content = token.lexeme[1..token.lexeme.len() - 1].to_string();
            return Ok(Expr::Literal(Literal::StringLit(content)));
        }

        if self.check(TokenKind::True) {
            self.advance();
            return Ok(Expr::Literal(Literal::Bool(true)));
        }

        if self.check(TokenKind::False) {
            self.advance();
            return Ok(Expr::Literal(Literal::Bool(false)));
        }

        if self.check(TokenKind::Splat) {
            self.advance();
            self.expect_kind(TokenKind::LeftParen, "expected '(' after 'splat'")?;
            let args = self.parse_args()?;
            self.expect_kind(TokenKind::RightParen, "expected ')' after arguments")?;
            return Ok(Expr::Call {
                name: "splat".to_string(),
                args,
            });
        }

        if self.check(TokenKind::Identifier) {
            let token = self.advance().clone();
            let name = token.lexeme.clone();
            if self.check(TokenKind::LeftParen) {
                self.advance();
                let args = self.parse_args()?;
                self.expect_kind(TokenKind::RightParen, "expected ')' after arguments")?;
                return Ok(Expr::Call { name, args });
            }
            // Struct literal: Name { field: val, ... } — name starts with uppercase
            if self.check(TokenKind::LeftBrace)
                && name.starts_with(|c: char| c.is_ascii_uppercase())
            {
                self.advance(); // consume {
                let mut fields = Vec::new();
                while !self.check(TokenKind::RightBrace) && !self.is_at_end() {
                    let field_name = self
                        .expect_kind(TokenKind::Identifier, "expected field name")?
                        .lexeme
                        .clone();
                    self.expect_kind(TokenKind::Colon, "expected ':' after field name")?;
                    let value = self.expression()?;
                    fields.push((field_name, value));
                    if self.check(TokenKind::Comma) {
                        self.advance();
                    }
                }
                self.expect_kind(TokenKind::RightBrace, "expected '}' after struct literal")?;
                return Ok(Expr::StructLiteral { name, fields });
            }
            let mut expr = Expr::Variable(name);
            // Postfix indexing and field access: name[expr] or name.field
            loop {
                if self.check(TokenKind::LeftBracket) {
                    self.advance(); // consume [
                    let index = self.expression()?;
                    self.expect_kind(TokenKind::RightBracket, "expected ']' after index")?;
                    expr = Expr::Index {
                        object: Box::new(expr),
                        index: Box::new(index),
                    };
                } else if self.check(TokenKind::Dot) {
                    self.advance(); // consume .
                    let field_token =
                        self.expect_kind(TokenKind::Identifier, "expected field name after '.'")?;
                    let field = field_token.lexeme.clone();
                    expr = Expr::FieldAccess {
                        object: Box::new(expr),
                        field,
                    };
                } else {
                    break;
                }
            }
            return Ok(expr);
        }

        if self.check(TokenKind::LeftBracket) {
            self.advance(); // consume [
            let mut elements = Vec::new();
            if !self.check(TokenKind::RightBracket) {
                loop {
                    elements.push(self.expression()?);
                    if !self.check(TokenKind::Comma) {
                        break;
                    }
                    self.advance();
                }
            }
            self.expect_kind(
                TokenKind::RightBracket,
                "expected ']' after vector elements",
            )?;

            // Check for vector type suffix
            if self.check(TokenKind::F32x4) {
                self.advance();
                return Ok(Expr::Vector {
                    elements,
                    ty: crate::ast::TypeAnnotation::Vector {
                        elem: Box::new(crate::ast::TypeAnnotation::Named("f32".to_string())),
                        width: 4,
                    },
                });
            }
            if self.check(TokenKind::I32x4) {
                self.advance();
                return Ok(Expr::Vector {
                    elements,
                    ty: crate::ast::TypeAnnotation::Vector {
                        elem: Box::new(crate::ast::TypeAnnotation::Named("i32".to_string())),
                        width: 4,
                    },
                });
            }
            if self.check(TokenKind::F32x8) {
                self.advance();
                return Ok(Expr::Vector {
                    elements,
                    ty: crate::ast::TypeAnnotation::Vector {
                        elem: Box::new(crate::ast::TypeAnnotation::Named("f32".to_string())),
                        width: 8,
                    },
                });
            }
            if self.check(TokenKind::I32x8) {
                self.advance();
                return Ok(Expr::Vector {
                    elements,
                    ty: crate::ast::TypeAnnotation::Vector {
                        elem: Box::new(crate::ast::TypeAnnotation::Named("i32".to_string())),
                        width: 8,
                    },
                });
            }
            // No type suffix — it's an array literal (used for shuffle masks etc.)
            return Ok(Expr::ArrayLiteral(elements));
        }

        if self.check(TokenKind::LeftParen) {
            self.advance();
            let expr = self.expression()?;
            self.expect_kind(TokenKind::RightParen, "expected ')' after expression")?;
            return Ok(expr);
        }

        Err(CompileError::parse_error(
            format!("expected expression, found {:?}", self.peek_kind()),
            self.current_position(),
        ))
    }
}
