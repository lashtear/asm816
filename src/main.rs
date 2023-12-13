pub mod cpu816;

use bit_vec::BitVec;
use chumsky::prelude::*;
use clap::Parser as _;
use log::*;
use serde::{Deserialize, Serialize};
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
}

impl Debug for Atom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool(b) => write!(f, "Atom::Bool({b})"),
            Self::Char(c) => write!(
                f,
                "Atom::Char('{}' {:2>} ${:x})",
                c.escape_debug(),
                *c as u32,
                *c as u32
            ),
            Self::Int(i) => write!(f, "Atom::Int({i} ${i:0x})"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
/// A compound value within expressions
pub enum Value {
    Float(f64),
    Atom(Atom),
    BitMap(BitVec),
    String(String),
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

fn based_int(radix: u32) -> impl chumsky::Parser<char, Token, Error = Simple<char>> {
    text::int(radix)
        .map(move |s: String| {
            Token::Atom(Atom::Int(
                s.chars().flat_map(digit_val).fold(0, |acc, v| {
                    acc.checked_mul(radix as i64).expect("overflowed i64") + (v as i64)
                }),
            ))
        })
        .labelled("numeric digits")
}

fn lexer() -> impl chumsky::Parser<char, Vec<(Token, Span)>, Error = Simple<char>> {
    let num = just('0')
        .to(Token::Atom(Atom::Int(0)))
        .then_with(|z| {
            choice::<_, Simple<char>>((
                just('d').or_not().ignore_then(based_int(10)),
                just('x').ignore_then(based_int(16)),
                just('o').ignore_then(based_int(8)),
                just('b').ignore_then(based_int(2)),
            ))
            .or_not()
            .map(move |t| t.unwrap_or(z.clone()))
        })
        .or(choice::<_, Simple<char>>((
            just('$').ignore_then(based_int(16)),
            just('%').ignore_then(based_int(2)),
            based_int(10),
        )))
        .map_with_span(|t, span| (t, span));

    num.padded().repeated().then_ignore(end())
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
                    let (tokens, mut errs) = lexer().parse_recovery(src.as_str());
                    let summary = format!("{tokens:?} {errs:?}");
                    info!("{}", summary);
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
