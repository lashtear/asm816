#![warn(trivial_casts)]
#![warn(trivial_numeric_casts)]
pub mod cpu816;

use ariadne::{Color, ColorGenerator, Fmt, Label, Report, ReportKind, Source};
use bit_vec::BitVec;
use chumsky::error::SimpleReason;
use chumsky::prelude::*;
use clap::Parser as _;
use log::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::Debug;
use std::fs::{remove_file, File, OpenOptions};
use std::io::{self, ErrorKind::NotFound};
use std::{path::PathBuf, sync::Arc};

#[derive(Debug, clap::Parser)]
#[command(author, version, about, long_about=None)]
struct Args {
    #[arg(short = 'v', action= clap::ArgAction::Count, global=true)]
    /// Increase verbosity (cumulative)
    verbose: u8,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, clap::Parser)]
/// Asm816, an assembler for 65816
enum Command {
    /// Assemble source code into binary objects
    Assemble {
        #[arg(short = 'I')]
        /// Add path to search list for includes
        include_paths: Vec<PathBuf>,
        #[arg(short = 'o', default_value = "a.out")]
        /// Set output file
        output_file: PathBuf,
        #[arg(short = 'g', default_value = "false")]
        /// Include debug records
        enable_debug: bool,
        /// Source files to be assembled
        input_files: Vec<PathBuf>,
    },
    /// Link binary objects into executables
    Link {
        #[arg(short = 'o', default_value = "a.out")]
        /// Set output file
        output_file: PathBuf,
        #[arg(short = 'g', default_value = "false")]
        /// Include debug records
        enable_debug: bool,
        /// Object files to be linked
        input_files: Vec<PathBuf>,
    },
}

pub type Span = std::ops::Range<usize>;

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// An atomic value within the assembler's expression evaluation language
pub enum Atom {
    Bool(bool),
    Char(char),
    Int(i64),
    String(String),
}

impl Debug for Atom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool(b) => write!(f, "Atom::Bool({b})"),
            Self::Char(c) => write!(f, "Atom::Char({c:?} {:3>} ${:2x})", *c as u32, *c as u32),
            Self::Int(i) => write!(f, "Atom::Int({i} ${i:0x})"),
            Self::String(s) => write!(f, "Atom::String({s:?})"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
/// A compound value within expressions
pub enum Value {
    Float(f64),
    Atom(Atom),
    BitMap(BitVec),
    ByteArray(Vec<u8>),
    List(Vec<Value>),
    Symbol(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Token {
    Operator(String),
    Structural(char),
    Atom(Atom),
    Identifier(String),
    Instruction(String),
    EndOfLine,
}

/// Fast decode ASCII digits
fn digit_val(c: char) -> Result<u8, &'static str> {
    if ('0'..='~').contains(&c) {
        if c <= '9' {
            return Ok((c as u8) - b'0');
        } else if c > '@' {
            return Ok(10 + ((c as u8) | 0x20) - b'a');
        }
    }
    Err("invalid digits")
}

/// List of ASCII digits for `radix`
fn radix_choices(radix: u32) -> &'static str {
    const CHARS: &str = "0123456789AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz";
    assert!((2..37).contains(&radix));
    if radix <= 10 {
        &CHARS[0..radix as usize]
    } else {
        &CHARS[0..(radix as usize) * 2 - 10]
    }
}

/// Parse a digit sequence for `radix` and return it as an i64
///
/// Unlike text::int(), this silently allows leading zeros and
/// performs the conversion.
fn based_int(radix: u32) -> impl chumsky::Parser<char, i64, Error = Simple<char>> {
    one_of::<char, &str, Simple<char>>(radix_choices(radix))
        .repeated()
        .at_least(1)
        .collect::<String>()
        .map(move |s: String| {
            s.chars()
                .flat_map(digit_val)
                .try_fold(0i64, |acc: i64, v: u8| -> Option<i64> {
                    if let Some(acc) = acc.checked_mul(radix as i64) {
                        acc.checked_add(v as i64)
                    } else {
                        None
                    }
                })
        })
        .try_map(|n: Option<i64>, span| {
            n.ok_or(Simple::custom(span, "numeric literal overflows i64"))
        })
        .labelled("numeric digits")
}

fn escaped_char(quote: char) -> impl chumsky::Parser<char, char, Error = Simple<char>> {
    just('\\')
        .ignore_then(
            just(quote)
                .or(just('\\'))
                .or(one_of("nrt").map(|c| match c {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    _ => unreachable!(),
                }))
                .or(just("u{")
                    .ignore_then(based_int(16))
                    .then_ignore(just('}'))
                    .try_map(|n: i64, span| {
                        char::from_u32(n as u32)
                            .ok_or(Simple::custom(span, "invalid unicode escape"))
                    })),
        )
        .labelled("escaped character")
}

fn justci(chars: &str) -> impl chumsky::Parser<char, char, Error = Simple<char>> {
    let mut buf = String::with_capacity(chars.len() * 2);
    buf.push_str(chars.to_ascii_lowercase().as_str());
    buf.push_str(chars.to_ascii_uppercase().as_str());
    one_of::<char, String, Simple<char>>(buf)
}

fn lexer() -> impl chumsky::Parser<char, Vec<(Token, Span)>, Error = Simple<char>> {
    let num = just('0')
        .to(0)
        .then_with(|z| {
            choice::<_, Simple<char>>((
                justci("d").or_not().ignore_then(based_int(10)),
                justci("x").ignore_then(based_int(16)),
                justci("o").ignore_then(based_int(8)),
                justci("b").ignore_then(based_int(2)),
            ))
            .or_not()
            .map(move |t| t.unwrap_or(z))
        })
        .or(choice::<_, Simple<char>>((
            just('$').ignore_then(based_int(16)),
            just('%').ignore_then(based_int(2)),
            based_int(10),
        )))
        .map(|i| Token::Atom(Atom::Int(i)))
        .labelled("number");

    let qchar = none_of("\\\'\r\n")
        .or(escaped_char('\''))
        .delimited_by(just('\''), just('\''))
        .map(|c| Token::Atom(Atom::Char(c)));

    let qstr = none_of("\\\"\r\n")
        .or(escaped_char('\"'))
        .repeated()
        .collect::<String>()
        .delimited_by(just('\"'), just('\"'))
        .map(|s| Token::Atom(Atom::String(s)));

    let comment_body = none_of::<char, &str, Simple<char>>("\r\n")
        .repeated()
        .ignore_then(text::newline())
        .to(Token::EndOfLine)
        .labelled("comment");
    let semi_comment = just(';').ignore_then(comment_body);

    let op = one_of::<char, &str, Simple<char>>("+-*/^#,.")
        .map(|c: char| Token::Operator(c.to_string()))
        .or(one_of("&|<>=").then_with(|c: char| {
            just(c).map(move |_| {
                let mut s = String::with_capacity(2);
                s.push(c);
                s.push(c);
                Token::Operator(s)
            })
        }))
        .or(just("!=")
            .or(just(":="))
            .map(|s| Token::Operator(s.to_string())))
        .or(one_of("&|!<>:").map(|c: char| Token::Operator(c.to_string())))
        .labelled("operator");

    let structural = one_of::<char, &str, Simple<char>>("[]{}()").map(Token::Structural);

    let id = text::ident::<char, Simple<char>>().map(|s| {
        if cpu816::is_insn(s.to_ascii_lowercase().as_str()) {
            Token::Instruction(s)
        } else {
            Token::Identifier(s)
        }
    });

    let horizontal_whitespace = one_of(" \t").repeated().to(());

    let nl = text::newline().to(Token::EndOfLine);

    let lexer = num
        .or(qchar)
        .or(qstr)
        .or(op)
        .or(semi_comment)
        .or(structural)
        .or(id)
        .or(nl);
    lexer
        .map_with_span(|t, span| (t, span))
        .padded_by(horizontal_whitespace)
        .repeated()
        .then_ignore(end())
}

#[derive(Clone, Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    BitAnd,
    BitOr,
    BitXor,
    LogicAnd,
    LogicOr,
    ShiftLeft,
    ShiftRight,
    Equal,
    NotEqual,
    Greater,
    Less,
}

#[derive(Clone, Debug)]
pub enum UnaryOp {
    NumericNegate,
    BitInvert,
}

type U24 = u32;

pub enum AddressReference {
    Direct(u8),
    Absolute(u16),
    AbsoluteLong(U24),
    Segment(Arc<String>, u32),
}

pub struct Symbol {
    name: Arc<String>,
    address: Option<AddressReference>,
    width: u8,
}

pub type Spanned<T> = (T, Span);

#[derive(Clone, Debug)]
pub enum Expr {
    Value(Value),
    Expr(Box<Expr>),
    BinOp(BinaryOp, Vec<Spanned<Self>>),
    UnOp(UnaryOp, Box<Spanned<Self>>),
}

pub enum AsmAST {
    LabelDef { name: String, value: Box<Expr> },
    Assignment { name: String, value: Box<Expr> },
    Operation { name: String, args: Vec<Expr> },
}

// like LevelFilter::from_usize, but not private to that module
fn level_filter_from_usize(u: usize) -> Option<LevelFilter> {
    match u {
        0 => Some(LevelFilter::Off),
        1 => Some(LevelFilter::Error),
        2 => Some(LevelFilter::Warn),
        3 => Some(LevelFilter::Info),
        4 => Some(LevelFilter::Debug),
        5 => Some(LevelFilter::Trace),
        _ => None,
    }
}

/// Safely open an output file
fn open_output(filename: &PathBuf) -> io::Result<File> {
    // remove existing file as a convenience
    if let Err(err) = remove_file(filename) {
        if err.kind() != NotFound {
            Err(err)?
        }
    } else {
        debug!("Removed existing output file ({filename:?})");
    }
    // use atomic create_new() to ensure it truly is a new file
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(filename)
}

/// Warn the user about non-existent include paths
fn check_include_path(includes: &Vec<PathBuf>) {
    for path in includes {
        if !path.exists() {
            warn!("Include path ({path:?}) does not exist");
        } else if !path.is_dir() {
            warn!("Include path ({path:?}) is not a directory");
        }
    }
}

fn main() {
    let args = Args::parse();
    // Log at WARN + number of verboses.
    // Overriden by RUST_LOG if present.
    env_logger::init_from_env(
        env_logger::Env::new().default_filter_or(
            level_filter_from_usize(args.verbose as usize + 2)
                .unwrap_or(LevelFilter::Trace)
                .as_str(),
        ),
    );
    debug!("Received arguments:\n{:#?}", args);
    match args.command {
        Command::Assemble {
            include_paths,
            output_file,
            enable_debug: _,
            input_files,
        } => {
            if input_files.is_empty() {
                error!("No input files specified");
            } else {
                check_include_path(&include_paths);
                let outfile = open_output(&output_file).expect("output file");
                for file in input_files {
                    let src = std::fs::read_to_string(file).expect("input");
                    let (tokens, errs) = lexer().parse_recovery(src.as_str());
                    errs.into_iter()
                        .map(|e| e.map(|c| c.to_string()))
                        .for_each(|e| {
                            let e = e.clone();
                            let mut report = Report::build(ReportKind::Error, (), e.span().start);
                            let mut expected: HashSet<&Option<String>> = HashSet::new();
                            e.expected().for_each(|oc| {
                                expected.insert(oc);
                            });
                            let open_string = expected.contains(&Some("\"".to_string()));
                            let open_char = expected.contains(&Some("\'".to_string()));
                            let subj = if open_string {
                                "string literal"
                            } else {
                                "character literal"
                            };
                            let delim = if open_string { "\"" } else { "\'" };
                            let litspan =
                                if let Some(idx) = src.as_str()[0..e.span().start].rfind(&delim) {
                                    Span {
                                        start: idx,
                                        end: e.span().start,
                                    }
                                } else {
                                    Span {
                                        start: e.span().start,
                                        end: e.span().start,
                                    }
                                };
                            if *e.reason() == SimpleReason::Unexpected && (open_string || open_char)
                            {
                                report = report
                                    .with_message(format!("Unclosed {}", subj))
                                    .with_label(
                                        Label::new(litspan)
                                            .with_message(format!("Unclosed {}", subj,))
                                            .with_color(Color::Fixed(9)),
                                    )
                                    .with_label(
                                        Label::new(e.span())
                                            .with_message(format!(
                                                "Must be closed before this {}",
                                                e.found()
                                                    .unwrap_or(&"end of file".to_string())
                                                    .escape_debug()
                                                    .fg(Color::Fixed(11))
                                            ))
                                            .with_color(Color::Fixed(11)),
                                    );
                            } else {
                                report = match e.reason() {
                                    chumsky::error::SimpleReason::Unclosed { span, delimiter } => {
                                        report
                                            .with_message(format!(
                                                "Unclosed delimiter {}",
                                                delimiter.fg(Color::Yellow)
                                            ))
                                            .with_label(
                                                Label::new(span.clone())
                                                    .with_message(format!(
                                                        "Unclosed delimiter {}",
                                                        delimiter.fg(Color::Yellow)
                                                    ))
                                                    .with_color(Color::Yellow),
                                            )
                                            .with_label(
                                                Label::new(e.span())
                                                    .with_message(format!(
                                                        "Must be closed before this {}",
                                                        e.found()
                                                            .unwrap_or(&"end of file".to_string())
                                                            .fg(Color::Red)
                                                    ))
                                                    .with_color(Color::Red),
                                            )
                                    }
                                    chumsky::error::SimpleReason::Unexpected => report
                                        .with_message(format!(
                                            "Unexpected {}, expected {}",
                                            if let Some(tok) = e.found() {
                                                format!("token in input ({tok:?})")
                                            } else {
                                                "end of input".to_string()
                                            },
                                            if e.expected().len() == 0 {
                                                "something else".to_string()
                                            } else {
                                                e.expected()
                                                    .map(|expected| match expected {
                                                        Some(expected) => expected.to_string(),
                                                        None => "end of input".to_string(),
                                                    })
                                                    .collect::<Vec<_>>()
                                                    .join(", ")
                                            }
                                        ))
                                        .with_label(
                                            Label::new(e.span())
                                                .with_message(format!(
                                                    "Unexpected token {}",
                                                    e.found()
                                                        .unwrap_or(&"end of file".to_string())
                                                        .fg(Color::Red)
                                                ))
                                                .with_color(Color::Red),
                                        ),
                                    chumsky::error::SimpleReason::Custom(msg) => {
                                        report.with_message(msg).with_label(
                                            Label::new(e.span())
                                                .with_message(format!("{}", msg.fg(Color::Red)))
                                                .with_color(Color::Red),
                                        )
                                    }
                                }
                            };

                            report.finish().print(Source::from(&src)).unwrap();
                        });
                    if let Some(tokens) = tokens {
                        let mut rb =
                            Report::build(ReportKind::Custom("Debug", Color::Magenta), (), 0)
                                .with_message("Holy shit tokens");
                        let mut colors = ColorGenerator::new();
                        for (tok, span) in tokens.into_iter() {
                            rb = rb.with_label(
                                Label::new(span)
                                    .with_message(format!("{tok:?}"))
                                    .with_color(colors.next()),
                            );
                        }
                        rb.finish().print(Source::from(&src)).unwrap();
                    }
                }
                outfile.sync_all().expect("successfully written");
            }
        }
        Command::Link {
            output_file,
            enable_debug: _,
            input_files,
        } => {
            if input_files.is_empty() {
                error!("No input files specified");
            } else {
                let outfile = open_output(&output_file).expect("output file");
                outfile.sync_all().expect("successfully written");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer_numbers() {
        assert_eq!(
            lexer().parse("1234"),
            Ok(vec!((Token::Atom(Atom::Int(1234)), 0..4)))
        );
        assert_eq!(
            lexer().parse("00"),
            Ok(vec!((Token::Atom(Atom::Int(0)), 0..2)))
        );
        assert_eq!(
            lexer().parse("0x001"),
            Ok(vec!((Token::Atom(Atom::Int(1)), 0..5)))
        );
        assert_eq!(radix_choices(10), "0123456789");
        assert_eq!(
            radix_choices(36),
            "0123456789AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz"
        );
        assert_eq!(
            lexer().parse("0xFfFf"),
            Ok(vec!((Token::Atom(Atom::Int(65535)), 0..6)))
        );
        assert_eq!(
            lexer().parse("$7fffffffffffffff"),
            Ok(vec!((Token::Atom(Atom::Int(9223372036854775807)), 0..17)))
        );
    }
}
