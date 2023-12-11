pub mod cpu816;

use clap::Parser;
use log::*;
use std::{path::PathBuf, sync::Arc};

//use chumsky::prelude::*;

#[derive(Debug, Parser)]
#[command(author, version, about, long_about=None)]
struct Args {
    #[arg(short = 'v', action= clap::ArgAction::Count, global=true)]
    /// Increase verbosity (cumulative)
    verbose: u8,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Parser)]
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
        /// Enable debug mode
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
        /// Enable debug mode
        enable_debug: bool,
        /// Object files to be linked
        input_files: Vec<PathBuf>,
    },
}

pub enum AsmToken {
    Colon,
    Assign,
    Plus,
    Minus,
    Multiply,
    Divide,
    OpenP,
    CloseP,
    OpenB,
    CloseB,
    Comma,
    QuotedString(Arc<String>),
    Id(Arc<String>),
}

pub enum AddressReference {
    Direct(u8),
    Absolute(u16),
    AbsoluteLong(u32),
    Segment(Arc<String>, u32),
}

pub struct Symbol {
    name: Arc<String>,
    address: Option<AddressReference>,
    width: u8,
}

pub enum AsmExpr {
    Num(i64),
    FloatNum(f64),
    Str(Arc<String>),
    Expr(Box<AsmExpr>),
    BinOp(Binops, Vec<AsmExpr>),
    UnOp(Unops, Box<AsmExpr>),
}

pub enum Binops {
    Add,
    Sub,
    Mul,
    Div,
    ShiftLeft,
    ShiftRight,
}

pub enum Unops {
    NumNegate,
    LogicInvert,
}

pub enum AsmAST {
    LabelDef { name: String, value: Box<AsmExpr> },
    Assignment { name: String, value: Box<AsmExpr> },
    Operation { name: String, args: Vec<AsmExpr> },
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    info!("Received arguments:\n{:#?}", args);
}
