pub mod cpu816;

use clap::Parser;
use log::*;
use std::fs::{File, OpenOptions};
use std::io;
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
                .or_else(|| Some(LevelFilter::Trace))
                .unwrap()
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
