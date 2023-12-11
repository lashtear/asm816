pub mod data;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum AddressMode {
    Absolute,
    AbsoluteIndexedX,
    AbsoluteIndexedY,
    LongAbsolute,
    LongAbsoluteIndexedX,
    AbsoluteIndirect,
    AbsoluteIndexedXIndirect,
    AbsoluteIndirectLong,
    Direct,
    DirectIndexedX,
    DirectIndexedY,
    DirectIndirect,
    DirectIndexedXIndirect,
    DirectIndirectIndexedY,
    DirectIndirectLong,
    DirectIndirectLongIndexedY,
    Immediate8,
    Immediate16,
    ImmediateMSize,
    ImmediateXSize,
    Implied,
    StackRelative,
    StackRelativeIndirectIndexedY,
    Relative8,
    Relative16,
    BlockMove,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct InstructionLengthInfo {
    length: u8,
    minus_m: bool,
    minus_x: bool,
}

impl InstructionLengthInfo {
    pub const fn from_u8(length: u8) -> Self {
        Self {
            length,
            minus_m: false,
            minus_x: false,
        }
    }
    pub const fn mm(self) -> Self {
        Self {
            minus_m: true,
            ..self
        }
    }
    pub const fn mx(self) -> Self {
        Self {
            minus_x: true,
            ..self
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct InstructionCycleInfo {
    cycles: u8,
    minus_e: bool,
    minus_m: bool,
    minus_x: bool,
    minus_2m: bool,
    minus_2x: bool,
    plus_t: bool,
    plus_xp: bool,
    plus_ep: bool,
    plus_tep: bool,
    plus_w: bool,
}

impl InstructionCycleInfo {
    pub const fn from_u8(cycles: u8) -> Self {
        Self {
            cycles,
            minus_e: false,
            minus_m: false,
            minus_x: false,
            minus_2m: false,
            minus_2x: false,
            plus_t: false,
            plus_xp: false,
            plus_ep: false,
            plus_tep: false,
            plus_w: false,
        }
    }
    pub const fn me(self) -> Self {
        Self {
            minus_e: true,
            ..self
        }
    }
    pub const fn mm(self) -> Self {
        Self {
            minus_m: true,
            ..self
        }
    }
    pub const fn mx(self) -> Self {
        Self {
            minus_x: true,
            ..self
        }
    }
    pub const fn m2m(self) -> Self {
        Self {
            minus_2m: true,
            ..self
        }
    }
    pub const fn m2x(self) -> Self {
        Self {
            minus_2m: true,
            ..self
        }
    }
    pub const fn pxp(self) -> Self {
        Self {
            plus_xp: true,
            ..self
        }
    }
    pub const fn pt(self) -> Self {
        Self {
            plus_t: true,
            ..self
        }
    }
    pub const fn pep(self) -> Self {
        Self {
            plus_ep: true,
            ..self
        }
    }
    pub const fn ptep(self) -> Self {
        Self {
            plus_tep: true,
            ..self
        }
    }
    pub const fn pw(self) -> Self {
        Self {
            plus_w: true,
            ..self
        }
    }
}

// see http://www.6502.org/tutorials/65c816opcodes.html#6
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct InstructionData {
    code: u8,
    mnemonic: &'static str,
    name: &'static str,
    example: Option<&'static str>,
    mode: AddressMode,
    len: InstructionLengthInfo,
    cyc: InstructionCycleInfo,
    flag_result: &'static str,
}

impl Default for InstructionData {
    fn default() -> Self {
        InstructionData {
            code: 0,
            mnemonic: "???",
            name: "Unknown instruction",
            example: None,
            mode: AddressMode::Immediate8,
            len: InstructionLengthInfo::from_u8(2),
            cyc: InstructionCycleInfo::from_u8(8).me(),
            flag_result: "....01.. .",
        }
    }
}

#[derive(Clone, Copy)]
#[repr(u16)]
pub enum Flag {
    Carry = 1 << 0,
    Zero = 1 << 1,
    InterruptDisable = 1 << 2,
    Decimal = 1 << 3,
    XSizeBreak = 1 << 4,
    MSize = 1 << 5,
    OVerflow = 1 << 6,
    Negative = 1 << 7,
    Emulation = 1 << 8,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct FlagState {
    flags: u16,
}

impl FlagState {
    pub fn from_flags(flags: Vec<Flag>) -> Option<Self> {
        if flags.iter().any(|f: &Flag| ((*f as u16) & !511 != 0)) {
            None
        } else {
            Some(Self {
                flags: flags.iter().fold(0, |acc: u16, x: &Flag| (acc | *x as u16)),
            })
        }
    }
}

impl Default for FlagState {
    fn default() -> Self {
        Self { flags: 0b100100100 } // Emulation + 1bit + IntDisable
    }
}

pub enum BitStatus {
    Zero,
    One,
    Unknown,
}

pub struct EMX {
    e: BitStatus,
    m: BitStatus,
    x: BitStatus,
}

impl EMX {
    pub fn new(e: BitStatus, m: BitStatus, x: BitStatus) -> Self {
        Self { e, m, x }
    }
    pub fn state_lost(&mut self) {
        self.e = BitStatus::Unknown;
        self.m = BitStatus::Unknown;
        self.x = BitStatus::Unknown;
    }
    pub fn set_emulation(&mut self) {
        self.e = BitStatus::One;
        self.m = BitStatus::One;
        self.x = BitStatus::One;
    }
}

#[cfg(test)]
mod tests {
    use super::data::*;

    #[test]
    fn instruction_data_is_valid() {
        assert_eq!(INST[0].mnemonic, "brk");
        assert_eq!(INST[0xfb].mnemonic, "xce");
        assert_eq!(INST[0xff].mnemonic, "sbc");
    }
}
