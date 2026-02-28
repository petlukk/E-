use crate::ast::{Expr, Literal, Param, Stmt};
use crate::error::CompileError;
use crate::lexer::{Position, Span, TokenKind};

use super::Parser;

impl Parser {
    pub(super) fn statement(&mut self) -> crate::error::Result<Stmt> {
        if self.check(TokenKind::Return) {
            let start = self.current_position();
            self.advance();
            if self.check(TokenKind::RightBrace) {
                return Ok(Stmt::Return(None, Span::new(start.clone(), start)));
            }
            let expr = self.expression()?;
            let end = expr.span().end.clone();
            return Ok(Stmt::Return(Some(expr), Span::new(start, end)));
        }

        if self.check(TokenKind::Let) {
            return self.parse_let();
        }

        if self.check(TokenKind::If) {
            return self.parse_if();
        }

        if self.check(TokenKind::Unroll) {
            return self.parse_unroll();
        }

        if self.check(TokenKind::ForEach) {
            return self.parse_foreach();
        }

        if self.check(TokenKind::While) {
            return self.parse_while();
        }

        // Assignment: name = value
        if self.check(TokenKind::Identifier) && self.peek_next_kind() == Some(&TokenKind::Equals) {
            let start = self.current_position();
            let name = self.advance().lexeme.clone();
            self.advance(); // consume '='
            let value = self.expression()?;
            let end = value.span().end.clone();
            return Ok(Stmt::Assign {
                target: name,
                value,
                span: Span::new(start, end),
            });
        }

        // Parse expression (could be index expr, field access, function call, etc.)
        let expr = self.expression()?;
        let expr_start = expr.span().start.clone();

        // Check for assignment: expr = value
        if self.check(TokenKind::Equals) {
            self.advance(); // consume '='
            let value = self.expression()?;
            let end = value.span().end.clone();
            if let Expr::FieldAccess { object, field, .. } = expr {
                return Ok(Stmt::FieldAssign {
                    object: *object,
                    field,
                    value,
                    span: Span::new(expr_start, end),
                });
            }
            if let Expr::Index { object, index, .. } = expr {
                if let Expr::Variable(name, _) = *object {
                    return Ok(Stmt::IndexAssign {
                        object: name,
                        index: *index,
                        value,
                        span: Span::new(expr_start, end),
                    });
                }
            }
            return Err(crate::error::CompileError::parse_error(
                "invalid assignment target",
                self.current_position(),
            ));
        }

        let end = expr.span().end.clone();
        Ok(Stmt::ExprStmt(expr, Span::new(expr_start, end)))
    }

    fn parse_let(&mut self) -> crate::error::Result<Stmt> {
        let start = self.current_position();
        self.advance(); // consume 'let'
        let mutable = if self.check(TokenKind::Mut) {
            self.advance();
            true
        } else {
            false
        };
        let name_token =
            self.expect_kind(TokenKind::Identifier, "expected variable name after 'let'")?;
        let name = name_token.lexeme.clone();
        self.expect_kind(
            TokenKind::Colon,
            "expected ':' after variable name (type annotation required)",
        )?;
        let ty = self.parse_type()?;
        self.expect_kind(TokenKind::Equals, "expected '=' after type annotation")?;
        let value = self.expression()?;
        let end = value.span().end.clone();
        Ok(Stmt::Let {
            name,
            ty,
            value,
            mutable,
            span: Span::new(start, end),
        })
    }

    fn parse_if(&mut self) -> crate::error::Result<Stmt> {
        let start = self.current_position();
        self.advance(); // consume 'if'
        let condition = self.expression()?;
        self.expect_kind(TokenKind::LeftBrace, "expected '{' after if condition")?;
        let then_body = self.parse_block()?;
        self.expect_kind(TokenKind::RightBrace, "expected '}' after if body")?;

        let else_body = if self.check(TokenKind::Else) {
            self.advance();
            if self.check(TokenKind::If) {
                let nested_if = self.parse_if()?;
                Some(vec![nested_if])
            } else {
                self.expect_kind(TokenKind::LeftBrace, "expected '{' after else")?;
                let body = self.parse_block()?;
                self.expect_kind(TokenKind::RightBrace, "expected '}' after else body")?;
                Some(body)
            }
        } else {
            None
        };
        let end = self.previous_position();

        Ok(Stmt::If {
            condition,
            then_body,
            else_body,
            span: Span::new(start, end),
        })
    }

    fn parse_unroll(&mut self) -> crate::error::Result<Stmt> {
        let start = self.current_position();
        self.advance(); // consume 'unroll'
        self.expect_kind(TokenKind::LeftParen, "expected '(' after 'unroll'")?;
        let count_token =
            self.expect_kind(TokenKind::IntLiteral, "expected integer in unroll(N)")?;
        let count: u32 = count_token.lexeme.parse().map_err(|_| {
            crate::error::CompileError::parse_error(
                "unroll count must be a positive integer",
                self.current_position(),
            )
        })?;
        self.expect_kind(TokenKind::RightParen, "expected ')' after unroll count")?;

        let inner = self.statement()?;
        match &inner {
            Stmt::While { .. } | Stmt::ForEach { .. } => {}
            _ => {
                return Err(crate::error::CompileError::parse_error(
                    "unroll must be followed by a loop (while or foreach)",
                    self.current_position(),
                ));
            }
        }
        let end = inner.span().end.clone();
        Ok(Stmt::Unroll {
            count,
            body: Box::new(inner),
            span: Span::new(start, end),
        })
    }

    fn parse_foreach(&mut self) -> crate::error::Result<Stmt> {
        let start = self.current_position();
        self.advance(); // consume 'foreach'
        self.expect_kind(TokenKind::LeftParen, "expected '(' after 'foreach'")?;
        let var_token = self.expect_kind(TokenKind::Identifier, "expected loop variable name")?;
        let var = var_token.lexeme.clone();
        self.expect_kind(TokenKind::In, "expected 'in' after loop variable")?;
        let start_expr = self.expression()?;
        self.expect_kind(TokenKind::DotDot, "expected '..' in range")?;
        let end_expr = self.expression()?;
        self.expect_kind(TokenKind::RightParen, "expected ')' after range")?;
        self.expect_kind(TokenKind::LeftBrace, "expected '{' after foreach header")?;
        let body = self.parse_block()?;
        self.expect_kind(TokenKind::RightBrace, "expected '}' after foreach body")?;
        let end = self.previous_position();
        Ok(Stmt::ForEach {
            var,
            start: start_expr,
            end: end_expr,
            body,
            span: Span::new(start, end),
        })
    }

    fn parse_while(&mut self) -> crate::error::Result<Stmt> {
        let start = self.current_position();
        self.advance(); // consume 'while'
        let condition = self.expression()?;
        self.expect_kind(TokenKind::LeftBrace, "expected '{' after while condition")?;
        let body = self.parse_block()?;
        self.expect_kind(TokenKind::RightBrace, "expected '}' after while body")?;
        let end = self.previous_position();
        Ok(Stmt::While {
            condition,
            body,
            span: Span::new(start, end),
        })
    }

    pub(super) fn parse_static_assert(&mut self, start: Position) -> crate::error::Result<Stmt> {
        self.expect_kind(TokenKind::LeftParen, "expected '(' after 'static_assert'")?;
        let condition = self.expression()?;
        self.expect_kind(
            TokenKind::Comma,
            "expected ',' after static_assert condition",
        )?;
        let msg_token = self.expect_kind(
            TokenKind::StringLiteral,
            "expected string literal as static_assert message",
        )?;
        // Strip surrounding quotes from lexeme
        let raw = &msg_token.lexeme;
        let message = raw[1..raw.len() - 1].to_string();
        self.expect_kind(
            TokenKind::RightParen,
            "expected ')' after static_assert message",
        )?;
        let end = self.previous_position();
        Ok(Stmt::StaticAssert {
            condition,
            message,
            span: Span::new(start, end),
        })
    }

    pub(super) fn parse_const(&mut self, start: Position) -> crate::error::Result<Stmt> {
        let name_token = self.expect_kind(
            TokenKind::Identifier,
            "expected constant name after 'const'",
        )?;
        let name = name_token.lexeme.clone();
        self.expect_kind(TokenKind::Colon, "expected ':' after constant name")?;
        let ty = self.parse_type()?;
        self.expect_kind(TokenKind::Equals, "expected '=' after constant type")?;

        // Parse optional leading minus for negative literals
        let negative = if self.check(TokenKind::Minus) {
            self.advance();
            true
        } else {
            false
        };

        let value = match self.peek_kind() {
            Some(TokenKind::IntLiteral) => {
                let tok = self.advance().clone();
                let n: i64 = tok.lexeme.parse().map_err(|_| {
                    CompileError::parse_error("invalid integer literal", tok.position.clone())
                })?;
                Literal::Integer(if negative { -n } else { n })
            }
            Some(TokenKind::HexLiteral) => {
                let tok = self.advance().clone();
                let n = i64::from_str_radix(&tok.lexeme[2..], 16).map_err(|_| {
                    CompileError::parse_error("invalid hex literal", tok.position.clone())
                })?;
                if negative {
                    Literal::Integer(-n)
                } else {
                    Literal::Integer(n)
                }
            }
            Some(TokenKind::BinLiteral) => {
                let tok = self.advance().clone();
                let n = i64::from_str_radix(&tok.lexeme[2..], 2).map_err(|_| {
                    CompileError::parse_error("invalid binary literal", tok.position.clone())
                })?;
                if negative {
                    Literal::Integer(-n)
                } else {
                    Literal::Integer(n)
                }
            }
            Some(TokenKind::FloatLiteral) => {
                let tok = self.advance().clone();
                let n: f64 = tok.lexeme.parse().map_err(|_| {
                    CompileError::parse_error("invalid float literal", tok.position.clone())
                })?;
                Literal::Float(if negative { -n } else { n })
            }
            _ => {
                return Err(CompileError::parse_error(
                    "const value must be a literal (integer or float)",
                    self.current_position(),
                ));
            }
        };

        let end = self.previous_position();
        Ok(Stmt::Const {
            name,
            ty,
            value,
            span: Span::new(start, end),
        })
    }

    pub(super) fn parse_kernel(
        &mut self,
        export: bool,
        start: Position,
    ) -> crate::error::Result<Stmt> {
        let name_token = self.expect_kind(TokenKind::Identifier, "expected kernel name")?;
        let name = name_token.lexeme.clone();

        self.expect_kind(TokenKind::LeftParen, "expected '(' after kernel name")?;
        let params = self.parse_params()?;
        self.expect_kind(TokenKind::RightParen, "expected ')' after parameters")?;

        // Parse: over VAR in RANGE step STEP
        self.expect_identifier("over", "expected 'over' after kernel parameters")?;
        let var_token =
            self.expect_kind(TokenKind::Identifier, "expected loop variable after 'over'")?;
        let range_var = var_token.lexeme.clone();
        self.expect_kind(TokenKind::In, "expected 'in' after loop variable")?;
        let bound_token =
            self.expect_kind(TokenKind::Identifier, "expected range bound after 'in'")?;
        let range_bound = bound_token.lexeme.clone();
        self.expect_identifier("step", "expected 'step' after range bound")?;

        let step = if self.check(TokenKind::IntLiteral) {
            let tok = self.advance().clone();
            let n: u32 = tok.lexeme.parse().map_err(|_| {
                CompileError::parse_error("invalid step value", tok.position.clone())
            })?;
            if n == 0 {
                return Err(CompileError::parse_error(
                    "step must be a positive integer",
                    tok.position,
                ));
            }
            n
        } else {
            return Err(CompileError::parse_error(
                "step must be a positive integer literal",
                self.current_position(),
            ));
        };

        // Parse optional tail clause
        let (tail, tail_body) = if self.check_identifier("tail") {
            self.advance();
            let strategy = if self.check_identifier("scalar") {
                self.advance();
                crate::ast::TailStrategy::Scalar
            } else if self.check_identifier("mask") {
                self.advance();
                crate::ast::TailStrategy::Mask
            } else if self.check_identifier("pad") {
                self.advance();
                crate::ast::TailStrategy::Pad
            } else {
                return Err(CompileError::parse_error(
                    "expected tail strategy: 'scalar', 'mask', or 'pad'",
                    self.current_position(),
                ));
            };

            let tb = match strategy {
                crate::ast::TailStrategy::Scalar | crate::ast::TailStrategy::Mask => {
                    self.expect_kind(TokenKind::LeftBrace, "expected '{' for tail body")?;
                    let body = self.parse_block()?;
                    self.expect_kind(TokenKind::RightBrace, "expected '}' after tail body")?;
                    Some(body)
                }
                crate::ast::TailStrategy::Pad => None,
            };
            (Some(strategy), tb)
        } else {
            (None, None)
        };

        // Parse main body
        self.expect_kind(TokenKind::LeftBrace, "expected '{' before kernel body")?;
        let body = self.parse_block()?;
        self.expect_kind(TokenKind::RightBrace, "expected '}' after kernel body")?;
        let end = self.previous_position();

        Ok(Stmt::Kernel {
            name,
            params,
            range_var,
            range_bound,
            step,
            tail,
            tail_body,
            body,
            export,
            span: Span::new(start, end),
        })
    }

    pub(super) fn parse_params(&mut self) -> crate::error::Result<Vec<Param>> {
        let mut params = Vec::new();
        if self.check(TokenKind::RightParen) {
            return Ok(params);
        }
        loop {
            let start = self.current_position();

            // Detect `out` contextual keyword via lookahead:
            //   out IDENT : → output param
            //   out :       → regular param named "out"
            let output = self.check_identifier("out")
                && matches!(self.peek_at(1), Some(TokenKind::Identifier))
                && matches!(self.peek_at(2), Some(TokenKind::Colon));
            if output {
                self.advance(); // consume `out`
            }

            let name_token = self.expect_kind(TokenKind::Identifier, "expected parameter name")?;
            let name = name_token.lexeme.clone();
            self.expect_kind(TokenKind::Colon, "expected ':' after parameter name")?;
            let ty = self.parse_type()?;
            let mut end = ty.span().end.clone();

            // Parse optional annotation: [cap: EXPR] or [cap: EXPR, count: PATH]
            let (cap, count) = if self.check(TokenKind::LeftBracket) {
                self.advance(); // consume [
                self.expect_identifier("cap", "expected 'cap' in output annotation")?;
                self.expect_kind(TokenKind::Colon, "expected ':' after 'cap'")?;
                let cap_str = self.collect_annotation_expr()?;
                let count_str = if self.check(TokenKind::Comma) {
                    self.advance(); // consume ,
                    self.expect_identifier("count", "expected 'count' after ','")?;
                    self.expect_kind(TokenKind::Colon, "expected ':' after 'count'")?;
                    let c = self.collect_annotation_expr()?;
                    Some(c)
                } else {
                    None
                };
                end = self.current_position();
                self.expect_kind(TokenKind::RightBracket, "expected ']' after annotation")?;
                (Some(cap_str), count_str)
            } else {
                (None, None)
            };

            params.push(Param {
                name,
                ty,
                output,
                cap,
                count,
                span: Span::new(start, end),
            });
            if !self.check(TokenKind::Comma) {
                break;
            }
            self.advance();
        }
        Ok(params)
    }

    /// Collect tokens as a string until `,`, `]`, or end-of-input.
    fn collect_annotation_expr(&mut self) -> crate::error::Result<String> {
        let mut parts = Vec::new();
        while !self.is_at_end()
            && !self.check(TokenKind::Comma)
            && !self.check(TokenKind::RightBracket)
        {
            let tok = self.advance();
            parts.push(tok.lexeme.clone());
        }
        if parts.is_empty() {
            return Err(CompileError::parse_error(
                "expected expression in annotation",
                self.current_position(),
            ));
        }
        Ok(parts.join(" "))
    }
}
