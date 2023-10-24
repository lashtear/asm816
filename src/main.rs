pub mod cpu816;

use chumsky::prelude::*;

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

    QuotedString(String),
    Id(String),
}

pub struct Symbol {
    name: String,
    address: Option<u64>,
    loc: (),
}

pub enum AsmExpr {
    Num(i64),
    FloatNum(f64),
    Str(String),
    Add(Box<AsmExpr>, Box<AsmExpr>),
    Sub(Box<AsmExpr>, Box<AsmExpr>),
    Mul(Box<AsmExpr>, Box<AsmExpr>),
    Div(Box<AsmExpr>, Box<AsmExpr>),
}

pub enum AsmAST {
    LabelDef { name: String, value: Box<AsmExpr> },
    Assignment { name: String, value: Box<AsmExpr> },
    Operation { name: String, args: Vec<AsmExpr> },
}

fn main() {
    println!("Hello, world!");
}
