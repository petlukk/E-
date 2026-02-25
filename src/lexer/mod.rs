pub mod tokens;

use logos::Logos;
use std::fmt;

use crate::error::CompileError;

#[derive(Debug, Clone, PartialEq)]
pub struct Position {
    pub line: usize,
    pub column: usize,
    pub offset: usize,
}

impl Position {
    pub fn new(line: usize, column: usize, offset: usize) -> Self {
        Self {
            line,
            column,
            offset,
        }
    }
}

impl Default for Position {
    fn default() -> Self {
        Self {
            line: 1,
            column: 1,
            offset: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub lexeme: String,
    pub position: Position,
}

impl Token {
    pub fn new(kind: TokenKind, lexeme: String, position: Position) -> Self {
        Self {
            kind,
            lexeme,
            position,
        }
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}({:?}) at {}:{}",
            self.kind, self.lexeme, self.position.line, self.position.column
        )
    }
}

#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r]+")]
pub enum TokenKind {
    #[token("func")]
    Func,
    #[token("export")]
    Export,
    #[token("return")]
    Return,
    #[token("let")]
    Let,
    #[token("mut")]
    Mut,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("while")]
    While,
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("struct")]
    Struct,
    #[token("restrict")]
    Restrict,
    #[token("unroll")]
    Unroll,
    #[token("foreach")]
    ForEach,
    #[token("in")]
    In,

    #[token("i8")]
    I8,
    #[token("u8")]
    U8,
    #[token("i16")]
    I16,
    #[token("u16")]
    U16,
    #[token("i32")]
    I32,
    #[token("i64")]
    I64,
    #[token("f32")]
    F32,
    #[token("f64")]
    F64,
    #[token("bool")]
    Bool,
    #[token("i8x16")]
    I8x16,
    #[token("i8x32")]
    I8x32,
    #[token("u8x16")]
    U8x16,
    #[token("i16x8")]
    I16x8,
    #[token("i16x16")]
    I16x16,
    #[token("f32x4")]
    F32x4,
    #[token("i32x4")]
    I32x4,
    #[token("f32x8")]
    F32x8,
    #[token("i32x8")]
    I32x8,
    #[token("f32x16")]
    F32x16,
    #[token("splat")]
    Splat,

    #[regex("[a-zA-Z_][a-zA-Z0-9_]*")]
    Identifier,
    #[regex("0x[0-9a-fA-F]+")]
    HexLiteral,
    #[regex("0b[01]+")]
    BinLiteral,
    #[regex("[0-9]+\\.[0-9]+|[0-9]*\\.[0-9]+")]
    FloatLiteral,
    #[regex("[0-9]+")]
    IntLiteral,
    #[regex(r#""[^"]*""#)]
    StringLiteral,

    #[token("(")]
    LeftParen,
    #[token(")")]
    RightParen,
    #[token("{")]
    LeftBrace,
    #[token("}")]
    RightBrace,
    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token("->")]
    Arrow,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("=")]
    Equals,
    #[token("<=")]
    LessEqual,
    #[token(">=")]
    GreaterEqual,
    #[token("==")]
    EqualEqual,
    #[token("!=")]
    BangEqual,
    #[token("<")]
    Less,
    #[token(">")]
    Greater,
    #[token(".+")]
    PlusDot,
    #[token(".-")]
    MinusDot,
    #[token(".*")]
    StarDot,
    #[token("./")]
    SlashDot,
    #[token(".<=")]
    LessEqualDot,
    #[token(".>=")]
    GreaterEqualDot,
    #[token(".==")]
    EqualEqualDot,
    #[token(".!=")]
    BangEqualDot,
    #[token(".<")]
    LessDot,
    #[token(".>")]
    GreaterDot,
    #[token(".&")]
    AmpDot,
    #[token(".|")]
    PipeDot,
    #[token(".^")]
    CaretDot,
    #[token("..")]
    DotDot,
    #[token(".")]
    Dot,
    #[token("&&")]
    AmpAmp,
    #[token("||")]
    PipePipe,
    #[token("!")]
    Bang,
    #[token("[")]
    LeftBracket,
    #[token("]")]
    RightBracket,

    #[token("\n")]
    Newline,

    #[regex(r"//[^\n]*")]
    LineComment,
}

pub struct Lexer<'src> {
    source: &'src str,
}

impl<'src> Lexer<'src> {
    pub fn new(source: &'src str) -> Self {
        Self { source }
    }

    pub fn tokenize(&self) -> crate::error::Result<Vec<Token>> {
        let mut tokens = Vec::new();

        let line_starts = self.compute_line_starts();
        let mut lex = TokenKind::lexer(self.source);

        while let Some(result) = lex.next() {
            let span = lex.span();
            let lexeme = lex.slice().to_string();

            match result {
                Ok(kind) => {
                    if matches!(kind, TokenKind::LineComment | TokenKind::Newline) {
                        continue;
                    }

                    let position = self.offset_to_position(span.start, &line_starts);
                    tokens.push(Token::new(kind, lexeme, position));
                }
                Err(()) => {
                    let position = self.offset_to_position(span.start, &line_starts);
                    return Err(CompileError::lex_error(
                        format!("unexpected character: {lexeme:?}"),
                        position,
                    ));
                }
            }
        }

        Ok(tokens)
    }

    fn compute_line_starts(&self) -> Vec<usize> {
        let mut starts = vec![0];
        for (i, ch) in self.source.bytes().enumerate() {
            if ch == b'\n' {
                starts.push(i + 1);
            }
        }
        starts
    }

    fn offset_to_position(&self, offset: usize, line_starts: &[usize]) -> Position {
        let line = match line_starts.binary_search(&offset) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let column = offset - line_starts[line] + 1;
        Position::new(line + 1, column, offset)
    }
}
