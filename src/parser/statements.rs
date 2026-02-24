use crate::ast::{Expr, Stmt};
use crate::lexer::TokenKind;

use super::Parser;

impl Parser {
    pub(super) fn statement(&mut self) -> crate::error::Result<Stmt> {
        if self.check(TokenKind::Return) {
            self.advance();
            if self.check(TokenKind::RightBrace) {
                return Ok(Stmt::Return(None));
            }
            let expr = self.expression()?;
            return Ok(Stmt::Return(Some(expr)));
        }

        if self.check(TokenKind::Let) {
            return self.parse_let();
        }

        if self.check(TokenKind::If) {
            return self.parse_if();
        }

        if self.check(TokenKind::While) {
            return self.parse_while();
        }

        // Assignment: name = value
        if self.check(TokenKind::Identifier) && self.peek_next_kind() == Some(&TokenKind::Equals) {
            let name = self.advance().lexeme.clone();
            self.advance(); // consume '='
            let value = self.expression()?;
            return Ok(Stmt::Assign {
                target: name,
                value,
            });
        }

        // Parse expression (could be index expr, field access, function call, etc.)
        let expr = self.expression()?;

        // Check for assignment: expr = value
        if self.check(TokenKind::Equals) {
            self.advance(); // consume '='
            let value = self.expression()?;
            if let Expr::FieldAccess { object, field } = expr {
                return Ok(Stmt::FieldAssign {
                    object: *object,
                    field,
                    value,
                });
            }
            if let Expr::Index { object, index } = expr {
                if let Expr::Variable(name) = *object {
                    return Ok(Stmt::IndexAssign {
                        object: name,
                        index: *index,
                        value,
                    });
                }
            }
            return Err(crate::error::CompileError::parse_error(
                "invalid assignment target",
                self.current_position(),
            ));
        }

        Ok(Stmt::ExprStmt(expr))
    }

    fn parse_let(&mut self) -> crate::error::Result<Stmt> {
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
        Ok(Stmt::Let {
            name,
            ty,
            value,
            mutable,
        })
    }

    fn parse_if(&mut self) -> crate::error::Result<Stmt> {
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

        Ok(Stmt::If {
            condition,
            then_body,
            else_body,
        })
    }

    fn parse_while(&mut self) -> crate::error::Result<Stmt> {
        self.advance(); // consume 'while'
        let condition = self.expression()?;
        self.expect_kind(TokenKind::LeftBrace, "expected '{' after while condition")?;
        let body = self.parse_block()?;
        self.expect_kind(TokenKind::RightBrace, "expected '}' after while body")?;
        Ok(Stmt::While { condition, body })
    }
}
