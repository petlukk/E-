use crate::ast::{BinaryOp, Expr, Literal};
use crate::error::CompileError;
use crate::lexer::{Span, TokenKind};

use super::Parser;

impl Parser {
    pub(super) fn expression(&mut self) -> crate::error::Result<Expr> {
        self.logical_or()
    }

    fn logical_or(&mut self) -> crate::error::Result<Expr> {
        let mut left = self.logical_and()?;
        while self.check(TokenKind::PipePipe) {
            let start = left.span().start.clone();
            self.advance();
            let right = self.logical_and()?;
            let end = right.span().end.clone();
            left = Expr::Binary(
                Box::new(left),
                BinaryOp::Or,
                Box::new(right),
                Span::new(start, end),
            );
        }
        Ok(left)
    }

    fn logical_and(&mut self) -> crate::error::Result<Expr> {
        let mut left = self.comparison()?;
        while self.check(TokenKind::AmpAmp) {
            let start = left.span().start.clone();
            self.advance();
            let right = self.comparison()?;
            let end = right.span().end.clone();
            left = Expr::Binary(
                Box::new(left),
                BinaryOp::And,
                Box::new(right),
                Span::new(start, end),
            );
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
            let start = left.span().start.clone();
            self.advance();
            let right = self.additive()?;
            let end = right.span().end.clone();
            Ok(Expr::Binary(
                Box::new(left),
                op,
                Box::new(right),
                Span::new(start, end),
            ))
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
            || self.check(TokenKind::AmpDot)
            || self.check(TokenKind::PipeDot)
            || self.check(TokenKind::CaretDot)
        {
            let op = if self.check(TokenKind::Plus) {
                BinaryOp::Add
            } else if self.check(TokenKind::Minus) {
                BinaryOp::Subtract
            } else if self.check(TokenKind::PlusDot) {
                BinaryOp::AddDot
            } else if self.check(TokenKind::MinusDot) {
                BinaryOp::SubDot
            } else if self.check(TokenKind::AmpDot) {
                BinaryOp::AndDot
            } else if self.check(TokenKind::PipeDot) {
                BinaryOp::OrDot
            } else {
                BinaryOp::XorDot
            };
            let start = left.span().start.clone();
            self.advance();
            let right = self.multiplicative()?;
            let end = right.span().end.clone();
            left = Expr::Binary(Box::new(left), op, Box::new(right), Span::new(start, end));
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
            let start = left.span().start.clone();
            self.advance();
            let right = self.unary()?;
            let end = right.span().end.clone();
            left = Expr::Binary(Box::new(left), op, Box::new(right), Span::new(start, end));
        }
        Ok(left)
    }

    fn unary(&mut self) -> crate::error::Result<Expr> {
        if self.check(TokenKind::Bang) {
            let start = self.current_position();
            self.advance();
            let inner = self.unary()?;
            let end = inner.span().end.clone();
            return Ok(Expr::Not(Box::new(inner), Span::new(start, end)));
        }
        if self.check(TokenKind::Minus) {
            let is_literal = matches!(
                self.peek_next_kind(),
                Some(
                    TokenKind::IntLiteral
                        | TokenKind::FloatLiteral
                        | TokenKind::HexLiteral
                        | TokenKind::BinLiteral
                )
            );
            if !is_literal {
                let start = self.current_position();
                self.advance();
                let inner = self.unary()?;
                let end = inner.span().end.clone();
                return Ok(Expr::Negate(Box::new(inner), Span::new(start, end)));
            }
        }
        self.primary()
    }

    fn primary(&mut self) -> crate::error::Result<Expr> {
        // Unary minus for numeric literals
        if self.check(TokenKind::Minus) {
            if self.peek_next_kind() == Some(&TokenKind::IntLiteral) {
                let start = self.current_position();
                self.advance(); // consume '-'
                let token = self.advance().clone();
                let end = token.position.clone();
                let value: i64 = token.lexeme.parse::<i64>().map_err(|_| {
                    CompileError::parse_error(
                        format!("invalid integer literal: {}", token.lexeme),
                        token.position.clone(),
                    )
                })?;
                return Ok(Expr::Literal(
                    Literal::Integer(-value),
                    Span::new(start, end),
                ));
            }
            if self.peek_next_kind() == Some(&TokenKind::FloatLiteral) {
                let start = self.current_position();
                self.advance(); // consume '-'
                let token = self.advance().clone();
                let end = token.position.clone();
                let value: f64 = token.lexeme.parse().map_err(|_| {
                    CompileError::parse_error(
                        format!("invalid float literal: {}", token.lexeme),
                        token.position.clone(),
                    )
                })?;
                return Ok(Expr::Literal(Literal::Float(-value), Span::new(start, end)));
            }
            if self.peek_next_kind() == Some(&TokenKind::HexLiteral) {
                let start = self.current_position();
                self.advance(); // consume '-'
                let token = self.advance().clone();
                let end = token.position.clone();
                let hex_str = &token.lexeme[2..]; // strip "0x"
                let value = i64::from_str_radix(hex_str, 16).map_err(|_| {
                    CompileError::parse_error(
                        format!("invalid hex literal: {}", token.lexeme),
                        token.position.clone(),
                    )
                })?;
                return Ok(Expr::Literal(
                    Literal::Integer(-value),
                    Span::new(start, end),
                ));
            }
            if self.peek_next_kind() == Some(&TokenKind::BinLiteral) {
                let start = self.current_position();
                self.advance(); // consume '-'
                let token = self.advance().clone();
                let end = token.position.clone();
                let bin_str = &token.lexeme[2..]; // strip "0b"
                let value = i64::from_str_radix(bin_str, 2).map_err(|_| {
                    CompileError::parse_error(
                        format!("invalid binary literal: {}", token.lexeme),
                        token.position.clone(),
                    )
                })?;
                return Ok(Expr::Literal(
                    Literal::Integer(-value),
                    Span::new(start, end),
                ));
            }
        }

        if self.check(TokenKind::IntLiteral) {
            let token = self.advance().clone();
            let span = Span::new(token.position.clone(), token.position.clone());
            let value: i64 = token.lexeme.parse().map_err(|_| {
                CompileError::parse_error(
                    format!("invalid integer literal: {}", token.lexeme),
                    token.position.clone(),
                )
            })?;
            return Ok(Expr::Literal(Literal::Integer(value), span));
        }

        if self.check(TokenKind::HexLiteral) {
            let token = self.advance().clone();
            let span = Span::new(token.position.clone(), token.position.clone());
            let hex_str = &token.lexeme[2..]; // strip "0x"
            let value = i64::from_str_radix(hex_str, 16).map_err(|_| {
                CompileError::parse_error(
                    format!("invalid hex literal: {}", token.lexeme),
                    token.position.clone(),
                )
            })?;
            return Ok(Expr::Literal(Literal::Integer(value), span));
        }

        if self.check(TokenKind::BinLiteral) {
            let token = self.advance().clone();
            let span = Span::new(token.position.clone(), token.position.clone());
            let bin_str = &token.lexeme[2..]; // strip "0b"
            let value = i64::from_str_radix(bin_str, 2).map_err(|_| {
                CompileError::parse_error(
                    format!("invalid binary literal: {}", token.lexeme),
                    token.position.clone(),
                )
            })?;
            return Ok(Expr::Literal(Literal::Integer(value), span));
        }

        if self.check(TokenKind::FloatLiteral) {
            let token = self.advance().clone();
            let span = Span::new(token.position.clone(), token.position.clone());
            let value: f64 = token.lexeme.parse().map_err(|_| {
                CompileError::parse_error(
                    format!("invalid float literal: {}", token.lexeme),
                    token.position.clone(),
                )
            })?;
            return Ok(Expr::Literal(Literal::Float(value), span));
        }

        if self.check(TokenKind::StringLiteral) {
            let token = self.advance().clone();
            let span = Span::new(token.position.clone(), token.position.clone());
            let content = token.lexeme[1..token.lexeme.len() - 1].to_string();
            return Ok(Expr::Literal(Literal::StringLit(content), span));
        }

        if self.check(TokenKind::True) {
            let pos = self.current_position();
            self.advance();
            let span = Span::new(pos.clone(), pos);
            return Ok(Expr::Literal(Literal::Bool(true), span));
        }

        if self.check(TokenKind::False) {
            let pos = self.current_position();
            self.advance();
            let span = Span::new(pos.clone(), pos);
            return Ok(Expr::Literal(Literal::Bool(false), span));
        }

        if self.check(TokenKind::Splat) {
            let start = self.current_position();
            self.advance();
            self.expect_kind(TokenKind::LeftParen, "expected '(' after 'splat'")?;
            let args = self.parse_args()?;
            self.expect_kind(TokenKind::RightParen, "expected ')' after arguments")?;
            let end = self.previous_position();
            return Ok(Expr::Call {
                name: "splat".to_string(),
                args,
                span: Span::new(start, end),
            });
        }

        if self.check(TokenKind::Identifier) {
            let token = self.advance().clone();
            let start = token.position.clone();
            let name = token.lexeme.clone();
            if self.check(TokenKind::LeftParen) {
                self.advance();
                let args = self.parse_args()?;
                self.expect_kind(TokenKind::RightParen, "expected ')' after arguments")?;
                let end = self.previous_position();
                return Ok(Expr::Call {
                    name,
                    args,
                    span: Span::new(start, end),
                });
            }
            // Struct literal: Name { field: val, ... } — name starts with uppercase.
            // Disambiguate from control-flow body: struct literals always have
            // { Identifier : ... } so peek two tokens past the { to confirm.
            if self.check(TokenKind::LeftBrace)
                && name.starts_with(|c: char| c.is_ascii_uppercase())
                && self.peek_at(1) == Some(&TokenKind::Identifier)
                && self.peek_at(2) == Some(&TokenKind::Colon)
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
                let end = self.previous_position();
                return Ok(Expr::StructLiteral {
                    name,
                    fields,
                    span: Span::new(start, end),
                });
            }
            let mut expr = Expr::Variable(name, Span::new(start.clone(), start));
            // Postfix indexing and field access: name[expr] or name.field
            loop {
                if self.check(TokenKind::LeftBracket) {
                    let idx_start = expr.span().start.clone();
                    self.advance(); // consume [
                    let index = self.expression()?;
                    self.expect_kind(TokenKind::RightBracket, "expected ']' after index")?;
                    let end = self.previous_position();
                    expr = Expr::Index {
                        object: Box::new(expr),
                        index: Box::new(index),
                        span: Span::new(idx_start, end),
                    };
                } else if self.check(TokenKind::Dot) {
                    let fa_start = expr.span().start.clone();
                    self.advance(); // consume .
                    let field_token =
                        self.expect_kind(TokenKind::Identifier, "expected field name after '.'")?;
                    let field = field_token.lexeme.clone();
                    let end = self.previous_position();
                    expr = Expr::FieldAccess {
                        object: Box::new(expr),
                        field,
                        span: Span::new(fa_start, end),
                    };
                } else {
                    break;
                }
            }
            return Ok(expr);
        }

        if self.check(TokenKind::LeftBracket) {
            let start = self.current_position();
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
            let vec_suffixes: &[(TokenKind, &str, usize)] = &[
                (TokenKind::I8x16, "i8", 16),
                (TokenKind::I8x32, "i8", 32),
                (TokenKind::U8x16, "u8", 16),
                (TokenKind::I16x8, "i16", 8),
                (TokenKind::I16x16, "i16", 16),
                (TokenKind::F32x4, "f32", 4),
                (TokenKind::I32x4, "i32", 4),
                (TokenKind::F32x8, "f32", 8),
                (TokenKind::I32x8, "i32", 8),
                (TokenKind::F32x16, "f32", 16),
            ];
            for (tk, elem_name, width) in vec_suffixes {
                if self.check(tk.clone()) {
                    let ty_pos = self.current_position();
                    self.advance();
                    let end = self.previous_position();
                    let ty_span = Span::new(ty_pos.clone(), ty_pos.clone());
                    return Ok(Expr::Vector {
                        elements,
                        ty: crate::ast::TypeAnnotation::Vector {
                            elem: Box::new(crate::ast::TypeAnnotation::Named(
                                elem_name.to_string(),
                                ty_span.clone(),
                            )),
                            width: *width,
                            span: ty_span,
                        },
                        span: Span::new(start, end),
                    });
                }
            }
            // No type suffix — it's an array literal (used for shuffle masks etc.)
            let end = self.previous_position();
            return Ok(Expr::ArrayLiteral(elements, Span::new(start, end)));
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
