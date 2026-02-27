use crate::ast::{Expr, Stmt};
use crate::lexer::{Span, TokenKind};

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
}
