mod expressions;
mod statements;

use crate::ast::{Stmt, TypeAnnotation};
use crate::error::CompileError;
use crate::lexer::{Position, Span, Token, TokenKind};

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, current: 0 }
    }

    pub fn parse_program(&mut self) -> crate::error::Result<Vec<Stmt>> {
        let mut stmts = Vec::new();
        while !self.is_at_end() {
            stmts.push(self.declaration()?);
        }
        Ok(stmts)
    }

    fn declaration(&mut self) -> crate::error::Result<Stmt> {
        if self.check(TokenKind::Export) {
            let start = self.current_position();
            self.advance();
            if self.check_identifier("kernel") {
                self.advance();
                return self.parse_kernel(true, start);
            }
            self.expect_kind(
                TokenKind::Func,
                "expected 'func' or 'kernel' after 'export'",
            )?;
            return self.function(true, start);
        }
        if self.check(TokenKind::Func) {
            let start = self.current_position();
            self.advance();
            return self.function(false, start);
        }
        if self.check_identifier("kernel") {
            let start = self.current_position();
            self.advance();
            return self.parse_kernel(false, start);
        }
        if self.check(TokenKind::Struct) {
            let start = self.current_position();
            self.advance();
            return self.parse_struct(start);
        }
        if self.check(TokenKind::Const) {
            let start = self.current_position();
            self.advance();
            return self.parse_const(start);
        }
        if self.check_identifier("static_assert") {
            let start = self.current_position();
            self.advance();
            return self.parse_static_assert(start);
        }
        Err(CompileError::parse_error(
            format!("expected declaration, found {:?}", self.peek_kind()),
            self.current_position(),
        ))
    }

    fn function(&mut self, export: bool, start: Position) -> crate::error::Result<Stmt> {
        let name_token = self.expect_kind(TokenKind::Identifier, "expected function name")?;
        let name = name_token.lexeme.clone();

        self.expect_kind(TokenKind::LeftParen, "expected '(' after function name")?;
        let params = self.parse_params()?;
        self.expect_kind(TokenKind::RightParen, "expected ')' after parameters")?;

        let return_type = if self.check(TokenKind::Arrow) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect_kind(TokenKind::LeftBrace, "expected '{' before function body")?;
        let body = self.parse_block()?;
        self.expect_kind(TokenKind::RightBrace, "expected '}' after function body")?;
        let end = self.previous_position();

        Ok(Stmt::Function {
            name,
            params,
            return_type,
            body,
            export,
            span: Span::new(start, end),
        })
    }

    fn parse_struct(&mut self, start: Position) -> crate::error::Result<Stmt> {
        let name_token =
            self.expect_kind(TokenKind::Identifier, "expected struct name after 'struct'")?;
        let name = name_token.lexeme.clone();
        self.expect_kind(TokenKind::LeftBrace, "expected '{' after struct name")?;
        let mut fields = Vec::new();
        while !self.check(TokenKind::RightBrace) && !self.is_at_end() {
            let field_name = self.expect_kind(TokenKind::Identifier, "expected field name")?;
            let field_name = field_name.lexeme.clone();
            self.expect_kind(TokenKind::Colon, "expected ':' after field name")?;
            let ty = self.parse_type()?;
            fields.push(crate::ast::StructField {
                name: field_name,
                ty,
            });
            if self.check(TokenKind::Comma) {
                self.advance();
            }
        }
        self.expect_kind(TokenKind::RightBrace, "expected '}' after struct fields")?;
        let end = self.previous_position();
        Ok(Stmt::Struct {
            name,
            fields,
            span: Span::new(start, end),
        })
    }

    pub(super) fn parse_type(&mut self) -> crate::error::Result<TypeAnnotation> {
        // Pointer types: *T, *mut T, *restrict T, *restrict mut T
        if self.check(TokenKind::Star) {
            let start = self.current_position();
            self.advance(); // consume *
            let restrict = if self.check(TokenKind::Restrict) {
                self.advance();
                true
            } else {
                false
            };
            let mutable = if self.check(TokenKind::Mut) {
                self.advance();
                true
            } else {
                false
            };
            let inner = self.parse_type()?;
            let end = inner.span().end.clone();
            return Ok(TypeAnnotation::Pointer {
                mutable,
                restrict,
                inner: Box::new(inner),
                span: Span::new(start, end),
            });
        }

        let type_tokens = [
            TokenKind::I8,
            TokenKind::U8,
            TokenKind::I16,
            TokenKind::U16,
            TokenKind::I32,
            TokenKind::I64,
            TokenKind::F32,
            TokenKind::F64,
            TokenKind::Bool,
        ];

        // Vector type tokens — single token like f32x4 gets one span
        let vec_types: &[(TokenKind, &str, usize)] = &[
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
        for (tk, elem_name, width) in vec_types {
            if self.check(tk.clone()) {
                let pos = self.current_position();
                self.advance();
                let span = Span::new(pos.clone(), pos.clone());
                return Ok(TypeAnnotation::Vector {
                    elem: Box::new(TypeAnnotation::Named(elem_name.to_string(), span.clone())),
                    width: *width,
                    span,
                });
            }
        }

        for tk in &type_tokens {
            if self.check(tk.clone()) {
                let token = self.advance().clone();
                let span = Span::new(token.position.clone(), token.position);
                return Ok(TypeAnnotation::Named(token.lexeme.clone(), span));
            }
        }
        if self.check(TokenKind::Identifier) {
            let token = self.advance().clone();
            let span = Span::new(token.position.clone(), token.position);
            return Ok(TypeAnnotation::Named(token.lexeme.clone(), span));
        }
        Err(CompileError::parse_error(
            format!("expected type, found {:?}", self.peek_kind()),
            self.current_position(),
        ))
    }

    pub(super) fn parse_block(&mut self) -> crate::error::Result<Vec<Stmt>> {
        let mut stmts = Vec::new();
        while !self.check(TokenKind::RightBrace) && !self.is_at_end() {
            stmts.push(self.statement()?);
        }
        Ok(stmts)
    }

    pub(super) fn parse_args(&mut self) -> crate::error::Result<Vec<crate::ast::Expr>> {
        let mut args = Vec::new();
        if self.check(TokenKind::RightParen) {
            return Ok(args);
        }
        loop {
            args.push(self.expression()?);
            if !self.check(TokenKind::Comma) {
                break;
            }
            self.advance();
        }
        Ok(args)
    }

    // --- Helpers ---

    pub(super) fn peek_kind(&self) -> Option<&TokenKind> {
        self.tokens.get(self.current).map(|t| &t.kind)
    }

    pub(super) fn peek_next_kind(&self) -> Option<&TokenKind> {
        self.tokens.get(self.current + 1).map(|t| &t.kind)
    }

    pub(super) fn peek_at(&self, offset: usize) -> Option<&TokenKind> {
        self.tokens.get(self.current + offset).map(|t| &t.kind)
    }

    pub(super) fn check(&self, kind: TokenKind) -> bool {
        self.peek_kind() == Some(&kind)
    }

    pub(super) fn check_identifier(&self, name: &str) -> bool {
        self.peek_kind() == Some(&TokenKind::Identifier)
            && self.tokens.get(self.current).map(|t| t.lexeme.as_str()) == Some(name)
    }

    pub(super) fn expect_identifier(
        &mut self,
        name: &str,
        msg: &str,
    ) -> crate::error::Result<&Token> {
        if self.check_identifier(name) {
            Ok(self.advance())
        } else {
            Err(CompileError::parse_error(msg, self.current_position()))
        }
    }

    pub(super) fn advance(&mut self) -> &Token {
        let token = &self.tokens[self.current];
        self.current += 1;
        token
    }

    pub(super) fn expect_kind(
        &mut self,
        kind: TokenKind,
        msg: &str,
    ) -> crate::error::Result<&Token> {
        if self.check(kind) {
            Ok(self.advance())
        } else {
            Err(CompileError::parse_error(msg, self.current_position()))
        }
    }

    pub(super) fn is_at_end(&self) -> bool {
        self.current >= self.tokens.len()
    }

    pub(super) fn current_position(&self) -> Position {
        self.tokens
            .get(self.current)
            .map(|t| t.position.clone())
            .unwrap_or_else(|| {
                self.tokens
                    .last()
                    .map(|t| t.position.clone())
                    .unwrap_or_default()
            })
    }

    pub(super) fn previous_position(&self) -> Position {
        if self.current > 0 {
            self.tokens[self.current - 1].position.clone()
        } else {
            Position {
                line: 1,
                column: 1,
                offset: 0,
            }
        }
    }
}
