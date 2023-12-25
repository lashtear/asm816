#! /usr/bin/perl

use strict;
use warnings;

print <<'EOF';
use crate::cpu816::AddressMode;
use crate::cpu816::InstructionData;

use super::InstructionCycleInfo;
use super::InstructionLengthInfo;

pub const INST: [InstructionData; 256] = [
EOF

my %mode = ('imp' => 'Implied',
            '(dir,x)' => 'DirectIndexedXIndirect',
            'imm' => 'Immediate8',
            'stk,s' => 'StackRelative',
            'dir' => 'Direct',
            '[dir]' => 'DirectIndirectLong',
            'acc' => 'Accumulator',
            'abs' => 'Absolute',
            'long' => 'LongAbsolute',
            'rel8' => 'Relative8',
            '(dir),y' => 'DirectIndirectIndexedY',
            '(dir)' => 'DirectIndirect',
            '(stk,s),y' => 'StackRelativeIndirectIndexedY',
            'dir,x' => 'DirectIndexedX',
            '[dir],y' => 'DirectIndirectLong',
            'abs,y' => 'AbsoluteIndexedY',
            'abs,x' => 'AbsoluteIndexedX',
            'long,x' => 'LongAbsoluteIndexedX',
            'src,dest' => 'BlockMove',
            '(abs)' => 'AbsoluteIndirect',
            '(abs,x)' => 'AbsoluteIndexedXIndirect',
            'rel16' => 'Relative16',
            'dir,y' => 'DirectIndexedY',
            '[abs]' => 'AbsoluteIndirectLong',
           );

my %longname = ('brk' => 'Break/trap',
                'ora' => 'OR with Accumulator',
                'cop' => 'COProcessor operation',
                'tsb' => 'Test and Set Bits',
                'trb' => 'Test and Reset Bits',
                'asl' => 'Arithmetic Shift Left',
                'php' => 'PusH Processor status register',
                'phd' => 'PusH Direct register',
                'phb' => 'PusH data Bank register',
                'phk' => 'PusH K register',
                'plb' => 'PulL data Bank register',
                'pld' => 'PulL Direct register',
                'plp' => 'PulL Processor status register',
                'bcc' => 'Branch if Carry Clear',
                'bcs' => 'Branch if Carry Set',
                'beq' => 'Branch if EQual',
                'bmi' => 'Branch if MInus',
                'bne' => 'Branch if Not Equal',
                'bpl' => 'Branch if PLus',
                'bra' => 'BRanch Always',
                'bvc' => 'Branch if oVerflow Clear',
                'bvs' => 'Branch if oVerflow Set',
                'brl' => 'Branch long',
                'clc' => 'CLear Carry',
                'cld' => 'CLear Decimal mode',
                'cli' => 'CLear Interrupt disable',
                'clv' => 'CLear oVerflow',
                'sec' => 'SEt Carry',
                'sed' => 'SEt Decimal mode',
                'sei' => 'SEt Interrupt disable',
                'inc' => 'Increment',
                'dec' => 'Decrement',
                'tcd' => 'Transfer C accumulator to Direct register',
                'tcs' => 'Transfer C accumulator to Stack pointer',
                'tdc' => 'Transfer Direct register to C accumulator',
                'tsc' => 'Transfer Stack pointer to C accumulator',
                'jsr' => 'Jump to SubRoutine',
                'and' => 'AND with accumulator',
                'jsl' => 'Jump to SubRoutine Long',
                'bit' => 'test BITs',
                'rol' => 'ROtate Left',
                'ror' => 'ROtate Right',
                'lsr' => 'Logical Shift Right',
                'rti' => 'ReTurn from Interrupt',
                'eor' => 'Exclusive OR with accumulator',
                'wdm' => 'reserved by William D Mensch',
                'mvn' => 'MoVe memory Negative',
                'mvp' => 'MoVe memory Positive',
                'pla' => 'PulL Accumulator',
                'pha' => 'PusH Accumulator',
                'plx' => 'PulL X index',
                'phx' => 'PusH X index',
                'ply' => 'PulL Y index',
                'phy' => 'PusH Y index',
                'jmp' => 'JumP',
                'rts' => 'ReTurn from Subroutine',
                'sbc' => 'SuBtract with Carry',
                'adc' => 'ADd with Carry',
                'pea' => 'Push Effective Address',
                'pei' => 'Push Effective Indirect address',
                'per' => 'Push Effective Relative address',
                'sta' => 'STore Accumulator',
                'stx' => 'STore X index',
                'sty' => 'STore Y index',
                'stz' => 'STore Zero',
                'lda' => 'LoaD Accumulator',
                'ldx' => 'LoaD X index',
                'ldy' => 'LoaD Y index',
                'rtl' => 'ReTurn from Long subtroutine',
                'inx' => 'INcrement X index',
                'iny' => 'INcrement Y index',
                'dex' => 'DEcrement X index',
                'dey' => 'DEcrement Y index',
                'tax' => 'Transfer Accumulator to X index',
                'tay' => 'Transfer Accumulator to Y index',
                'tsx' => 'Transfer Stack pointer to X index',
                'txa' => 'Transfer X register to Accumulator',
                'txs' => 'Transfer X register to Stack pointer',
                'txy' => 'Transfer X register to Y index',
                'tya' => 'Transfer Y register to Accumulator',
                'tyx' => 'Transfer Y register to X index',
                'cmp' => 'CoMPare with Accumulator',
                'cpx' => 'ComPare with X index',
                'cpy' => 'ComPare with Y index',
                'sep' => 'SEt Processor status bits',
                'rep' => 'REset Processor status bits',
                'wai' => 'WAit for Interrupt',
                'stp' => 'SToP the clock',
                'nop' => 'No OPeration',
                'xba' => 'eXchange B and A accumulators',
                'xce' => 'eXchange Carry and Emulation bits',
               );

sub format_line {
    local $_ = shift;
    $_ = lc $_;
    if (my ($code, $len, $cyc, $mode, $flags, $name, $example) 
        = (/^([0-9a-f]{2}) ([-0-9mx]+)\s+([-*+0-9emptwx]+)\s+(\S+)\s+([.mx01*]{8} [.*]) (\w{3})(\s*\S*)\s*$/)) {

        $len =~ s/^(\d+)/InstructionLengthInfo::from_u8($1)/;
        $len =~ s/-([mx])/.m$1()/g;
        die "weird len $_ " if ($len =~ /[-+*]/);

        $cyc =~ s/^(\d+)/InstructionCycleInfo::from_u8($1)/;
        $cyc =~ s/-2\*([mx])/.m2$1()/g;
        $cyc =~ s/-2\*m/.m2m()/g;
        $cyc =~ s/-([mxe])/.m$1()/g;
        $cyc =~ s/\+(x\*p|e\*p|t\*e\*p|w)/.p$1()/g;
        $cyc =~ s/\+t/.pt()/g;
        $cyc =~ s/\*//g;
        die "weird cyc $_ " if ($cyc =~ /[-+*]/);

        if (not exists $mode{$mode}) {
            die "need mode for $mode";
        }
        $mode = $mode{$mode};
        if (not exists $longname{$name}) {
            die "need longname for $name";
        }
        my $longname = $longname{$name};

        print <<OUT
    InstructionData {
        code: 0x$code,
        mnemonic: "$name",
        name: "$longname",
        example: Some("$name$example"),
        mode: AddressMode::$mode,
        len: $len,
        cyc: $cyc,
        flag_result: "$flags",
    },
OUT

    } else {
        warn "bad line at $.: $_\n";
        next;
    }
}

# Generated from
# 65C816 Opcodes by Bruce Clark
# at http://www.6502.org/tutorials/65c816opcodes.html

my $data = <<'EOF';
00 1   8-e         imp       ....01.. . BRK
01 2   7-m+w       (dir,X)   m.....m. . ORA ($10,X)
02 2   8-e         imm       ....01.. . COP #$12
03 2   5-m         stk,S     m.....m. . ORA $32,S
04 2   7-2*m+w     dir       ......m. . TSB $10
05 2   4-m+w       dir       m.....m. . ORA $10
06 2   7-2*m+w     dir       m.....mm . ASL $10
07 2   7-m+w       [dir]     m.....m. . ORA [$10]
08 1   3           imp       ........ . PHP
09 3-m 3-m         imm       m.....m. . ORA #$54
0A 1   2           acc       m.....mm . ASL
0B 1   4           imp       ........ . PHD
0C 3   8-2*m       abs       ......m. . TSB $9876
0D 3   5-m         abs       m.....m. . ORA $9876
0E 3   8-2*m       abs       m.....mm . ASL $9876
0F 4   6-m         long      m.....m. . ORA $FEDCBA
10 2   2+t+t*e*p   rel8      ........ . BPL LABEL
11 2   7-m+w-x+x*p (dir),Y   m.....m. . ORA ($10),Y
12 2   6-m+w       (dir)     m.....m. . ORA ($10)
13 2   8-m         (stk,S),Y m.....m. . ORA ($32,S),Y
14 2   7-2*m+w     dir       ......m. . TRB $10
15 2   5-m+w       dir,X     m.....m. . ORA $10,X
16 2   8-2*m+w     dir,X     m.....mm . ASL $10,X
17 2   7-m+w       [dir],Y   m.....m. . ORA [$10],Y
18 1   2           imp       .......0 . CLC
19 3   6-m-x+x*p   abs,Y     m.....m. . ORA $9876,Y
1A 1   2           acc       m.....m. . INC
1B 1   2           imp       ........ . TCS
1C 3   8-2*m       abs       ......m. . TRB $9876
1D 3   6-m-x+x*p   abs,X     m.....m. . ORA $9876,X
1E 3   9-2*m       abs,X     m.....mm . ASL $9876,X
1F 4   6-m         long,X    m.....m. . ORA $FEDCBA,X
20 3   6           abs       ........ . JSR $1234
21 2   7-m+w       (dir,X)   m.....m. . AND ($10,X)
22 4   8           long      ........ . JSL $123456
23 2   5-m         stk,S     m.....m. . AND $32,S
24 2   4-m+w       dir       mm....m. . BIT $10
25 2   4-m+w       dir       m.....m. . AND $10
26 2   7-2*m+w     dir       m.....mm . ROL $10
27 2   7-m+w       [dir]     m.....m. . AND [$10]
28 1   4           imp       ******** . PLP
29 3-m 3-m         imm       m.....m. . AND #$54
2A 1   2           acc       m.....mm . ROL
2B 1   5           imp       *.....*. . PLD
2C 3   5-m         abs       mm....m. . BIT $9876
2D 3   5-m         abs       m.....m. . AND $9876
2E 3   8-2*m       abs       m.....mm . ROL $9876
2F 4   6-m         long      m.....m. . AND $FEDCBA
30 2   2+t+t*e*p   rel8      ........ . BMI LABEL
31 2   7-m+w-x+x*p (dir),Y   m.....m. . AND ($10),Y
32 2   6-m+w       (dir)     m.....m. . AND ($10)
33 2   8-m         (stk,S),Y m.....m. . AND ($32,S),Y
34 2   5-m+w       dir,X     mm....m. . BIT $10,X
35 2   5-m+w       dir,X     m.....m. . AND $10,X
36 2   8-2*m+w     dir,X     m.....mm . ROL $10,X
37 2   7-m+w       [dir],Y   m.....m. . AND [$10],Y
38 1   2           imp       .......1 . SEC
39 3   6-m-x+x*p   abs,Y     m.....m. . AND $9876,Y
3A 1   2           acc       m.....m. . DEC
3B 1   2           imp       *.....*. . TSC
3C 3   6-m-x+x*p   abs,X     mm....m. . BIT $9876,X
3D 3   6-m-x+x*p   abs,X     m.....m. . AND $9876,X
3E 3   9-2*m       abs,X     m.....mm . ROL $9876,X
3F 4   6-m         long,X    m.....m. . AND $FEDCBA,X
40 1   7-e         imp       ******** . RTI
41 2   7-m+w       (dir,X)   m.....m. . EOR ($10,X)
42 2   2           imm       ........ . WDM
43 2   5-m         stk,S     m.....m. . EOR $32,S
44 3   7           src,dest  ........ . MVP #$12,#$34
45 2   4-m+w       dir       m.....m. . EOR $10
46 2   7-2*m+w     dir       0.....m* . LSR $10
47 2   7-m+w       [dir]     m.....m. . EOR [$10]
48 1   4-m         imp       ........ . PHA
49 3-m 3-m         imm       m.....m. . EOR #$54
4A 1   2           acc       0.....m* . LSR
4B 1   3           imp       ........ . PHK
4C 3   3           abs       ........ . JMP $1234
4D 3   5-m         abs       m.....m. . EOR $9876
4E 3   8-2*m       abs       0.....m* . LSR $9876
4F 4   6-m         long      m.....m. . EOR $FEDCBA
50 2   2+t+t*e*p   rel8      ........ . BVC LABEL
51 2   7-m+w-x+x*p (dir),Y   m.....m. . EOR ($10),Y
52 2   6-m+w       (dir)     m.....m. . EOR ($10)
53 2   8-m         (stk,S),Y m.....m. . EOR ($32,S),Y
54 3   7           src,dest  ........ . MVN #$12,#$34
55 2   5-m+w       dir,X     m.....m. . EOR $10,X
56 2   8-2*m+w     dir,X     0.....m* . LSR $10,X
57 2   7-m+w       [dir],Y   m.....m. . EOR [$10],Y
58 1   2           imp       .....0.. . CLI
59 3   6-m-x+x*p   abs,Y     m.....m. . EOR $9876,Y
5A 1   4-x         imp       ........ . PHY
5B 1   2           imp       *.....*. . TCD
5C 4   4           long      ........ . JMP $FEDCBA
5D 3   6-m-x+x*p   abs,X     m.....m. . EOR $9876,X
5E 3   9-2*m       abs,X     0.....m* . LSR $9876,X
5F 4   6-m         long,X    m.....m. . EOR $FEDCBA,X
60 1   6           imp       ........ . RTS
61 2   7-m+w       (dir,X)   mm....mm . ADC ($10,X)
62 3   6           imm       ........ . PER LABEL
63 2   5-m         stk,S     mm....mm . ADC $32,S
64 2   4-m+w       dir       ........ . STZ $10
65 2   4-m+w       dir       mm....mm . ADC $10
66 2   7-2*m+w     dir       m.....m* . ROR $10
67 2   7-m+w       [dir]     mm....mm . ADC [$10]
68 1   5-m         imp       m.....m. . PLA
69 3-m 3-m         imm       mm....mm . ADC #$54
6A 1   2           acc       m.....m* . ROR
6B 1   6           imp       ........ . RTL
6C 3   5           (abs)     ........ . JMP ($1234)
6D 3   5-m         abs       mm....mm . ADC $9876
6E 3   8-2*m       abs       m.....m* . ROR $9876
6F 4   6-m         long      mm....mm . ADC $FEDCBA
70 2   2+t+t*e*p   rel8      ........ . BVS LABEL
71 2   7-m+w-x+x*p (dir),Y   mm....mm . ADC ($10),Y
72 2   6-m+w       (dir)     mm....mm . ADC ($10)
73 2   8-m         (stk,S),Y mm....mm . ADC ($32,S),Y
74 2   5-m+w       dir,X     ........ . STZ $10,X
75 2   5-m+w       dir,X     mm....mm . ADC $10,X
76 2   8-2*m+w     dir,X     m.....m* . ROR $10,X
77 2   7-m+w       [dir],Y   mm....mm . ADC [$10],Y
78 1   2           imp       .....1.. . SEI
79 3   6-m-x+x*p   abs,Y     mm....mm . ADC $9876,Y
7A 1   5-x         imp       x.....x. . PLY
7B 1   2           imp       *.....*. . TDC
7C 3   6           (abs,X)   ........ . JMP ($1234,X)
7D 3   6-m-x+x*p   abs,X     mm....mm . ADC $9876,X
7E 3   9-2*m       abs,X     m.....m* . ROR $9876,X
7F 4   6-m         long,X    mm....mm . ADC $FEDCBA,X
80 2   3+e*p       rel8      ........ . BRA LABEL
81 2   7-m+w       (dir,X)   ........ . STA ($10,X)
82 3   4           rel16     ........ . BRL LABEL
83 2   5-m         stk,S     ........ . STA $32,S
84 2   4-x+w       dir       ........ . STY $10
85 2   4-m+w       dir       ........ . STA $10
86 2   4-x+w       dir       ........ . STX $10
87 2   7-m+w       [dir]     ........ . STA [$10]
88 1   2           imp       x.....x. . DEY
89 3-m 3-m         imm       ......m. . BIT #$54
8A 1   2           imp       m.....m. . TXA
8B 1   3           imp       ........ . PHB
8C 3   5-x         abs       ........ . STY $9876
8D 3   5-m         abs       ........ . STA $9876
8E 3   5-x         abs       ........ . STX $9876
8F 4   6-m         long      ........ . STA $FEDCBA
90 2   2+t+t*e*p   rel8      ........ . BCC LABEL
91 2   7-m+w       (dir),Y   ........ . STA ($10),Y
92 2   6-m+w       (dir)     ........ . STA ($10)
93 2   8-m         (stk,S),Y ........ . STA ($32,S),Y
94 2   5-x+w       dir,X     ........ . STY $10,X
95 2   5-m+w       dir,X     ........ . STA $10,X
96 2   5-x+w       dir,Y     ........ . STX $10,Y
97 2   7-m+w       [dir],Y   ........ . STA [$10],Y
98 1   2           imp       m.....m. . TYA
99 3   6-m         abs,Y     ........ . STA $9876,Y
9A 1   2           imp       ........ . TXS
9B 1   2           imp       x.....x. . TXY
9C 3   5-m         abs       ........ . STZ $9876
9D 3   6-m         abs,X     ........ . STA $9876,X
9E 3   6-m         abs,X     ........ . STZ $9876,X
9F 4   6-m         long,X    ........ . STA $FEDCBA,X
A0 3-x 3-x         imm       x.....x. . LDY #$54
A1 2   7-m+w       (dir,X)   m.....m. . LDA ($10,X)
A2 3-x 3-x         imm       x.....x. . LDX #$54
A3 2   5-m         stk,S     m.....m. . LDA $32,S
A4 2   4-x+w       dir       x.....x. . LDY $10
A5 2   4-m+w       dir       m.....m. . LDA $10
A6 2   4-x+w       dir       x.....x. . LDX $10
A7 2   7-m+w       [dir]     m.....m. . LDA [$10]
A8 1   2           imp       x.....x. . TAY
A9 3-m 3-m         imm       m.....m. . LDA #$54
AA 1   2           imp       x.....x. . TAX
AB 1   4           imp       *.....*. . PLB
AC 3   5-x         abs       x.....x. . LDY $9876
AD 3   5-m         abs       m.....m. . LDA $9876
AE 3   5-x         abs       x.....x. . LDX $9876
AF 4   6-m         long      m.....m. . LDA $FEDCBA
B0 2   2+t+t*e*p   rel8      ........ . BCS LABEL
B1 2   7-m+w-x+x*p (dir),Y   m.....m. . LDA ($10),Y
B2 2   6-m+w       (dir)     m.....m. . LDA ($10)
B3 2   8-m         (stk,S),Y m.....m. . LDA ($32,S),Y
B4 2   5-x+w       dir,X     x.....x. . LDY $10,X
B5 2   5-m+w       dir,X     m.....m. . LDA $10,X
B6 2   5-x+w       dir,Y     x.....x. . LDX $10,Y
B7 2   7-m+w       [dir],Y   m.....m. . LDA [$10],Y
B8 1   2           imp       .0...... . CLV
B9 3   6-m-x+x*p   abs,Y     m.....m. . LDA $9876,Y
BA 1   2           imp       x.....x. . TSX
BB 1   2           imp       x.....x. . TYX
BC 3   6-2*x+x*p   abs,X     x.....x. . LDY $9876,X
BD 3   6-m-x+x*p   abs,X     m.....m. . LDA $9876,X
BE 3   6-2*x+x*p   abs,Y     x.....x. . LDX $9876,Y
BF 4   6-m         long,X    m.....m. . LDA $FEDCBA,X
C0 3-x 3-x         imm       x.....xx . CPY #$54
C1 2   7-m+w       (dir,X)   m.....mm . CMP ($10,X)
C2 2   3           imm       ******** . REP #$12
C3 2   5-m         stk,S     m.....mm . CMP $32,S
C4 2   4-x+w       dir       x.....xx . CPY $10
C5 2   4-m+w       dir       m.....mm . CMP $10
C6 2   7-2*m+w     dir       m.....m. . DEC $10
C7 2   7-m+w       [dir]     m.....mm . CMP [$10]
C8 1   2           imp       x.....x. . INY
C9 3-m 3-m         imm       m.....mm . CMP #$54
CA 1   2           imp       x.....x. . DEX
CB 1   3           imp       ........ . WAI
CC 3   5-x         abs       x.....xx . CPY $9876
CD 3   5-m         abs       m.....mm . CMP $9876
CE 3   8-2*m       abs       m.....m. . DEC $9876
CF 4   6-m         long      m.....mm . CMP $FEDCBA
D0 2   2+t+t*e*p   rel8      ........ . BNE LABEL
D1 2   7-m+w-x+x*p (dir),Y   m.....mm . CMP ($10),Y
D2 2   6-m+w       (dir)     m.....mm . CMP ($10)
D3 2   8-m         (stk,S),Y m.....mm . CMP ($32,S),Y
D4 2   6+w         dir       ........ . PEI $12
D5 2   5-m+w       dir,X     m.....mm . CMP $10,X
D6 2   8-2*m+w     dir,X     m.....m. . DEC $10,X
D7 2   7-m+w       [dir],Y   m.....mm . CMP [$10],Y
D8 1   2           imp       ....0... . CLD
D9 3   6-m-x+x*p   abs,Y     m.....mm . CMP $9876,Y
DA 1   4-x         imp       ........ . PHX
DB 1   3           imp       ........ . STP
DC 3   6           [abs]     ........ . JMP [$1234]
DD 3   6-m-x+x*p   abs,X     m.....mm . CMP $9876,X
DE 3   9-2*m       abs,X     m.....m. . DEC $9876,X
DF 4   6-m         long,X    m.....mm . CMP $FEDCBA,X
E0 3-x 3-x         imm       x.....xx . CPX #$54
E1 2   7-m+w       (dir,X)   mm....mm . SBC ($10,X)
E2 2   3           imm       ******** . SEP #$12
E3 2   5-m         stk,S     mm....mm . SBC $32,S
E4 2   4-x+w       dir       x.....xx . CPX $10
E5 2   4-m+w       dir       mm....mm . SBC $10
E6 2   7-2*m+w     dir       m.....m. . INC $10
E7 2   7-m+w       [dir]     mm....mm . SBC [$10]
E8 1   2           imp       x.....x. . INX
E9 3-m 3-m         imm       mm....mm . SBC #$54
EA 1   2           imp       ........ . NOP
EB 1   3           imp       *.....*. . XBA
EC 3   5-x         abs       x.....xx . CPX $9876
ED 3   5-m         abs       mm....mm . SBC $9876
EE 3   8-2*m       abs       m.....m. . INC $9876
EF 4   6-m         long      mm....mm . SBC $FEDCBA
F0 2   2+t+t*e*p   rel8      ........ . BEQ LABEL
F1 2   7-m+w-x+x*p (dir),Y   mm....mm . SBC ($10),Y
F2 2   6-m+w       (dir)     mm....mm . SBC ($10)
F3 2   8-m         (stk,S),Y mm....mm . SBC ($32,S),Y
F4 3   5           imm       ........ . PEA #$1234
F5 2   5-m+w       dir,X     mm....mm . SBC $10,X
F6 2   8-2*m+w     dir,X     m.....m. . INC $10,X
F7 2   7-m+w       [dir],Y   mm....mm . SBC [$10],Y
F8 1   2           imp       ....1... . SED
F9 3   6-m-x+x*p   abs,Y     mm....mm . SBC $9876,Y
FA 1   5-x         imp       x.....x. . PLX
FB 1   2           imp       .......* * XCE
FC 3   8           (abs,X)   ........ . JSR ($1234,X)
FD 3   6-m-x+x*p   abs,X     mm....mm . SBC $9876,X
FE 3   9-2*m       abs,X     m.....m. . INC $9876,X
FF 4   6-m         long,X    mm....mm . SBC $FEDCBA,X
EOF

foreach (split /\n/, $data) {
    format_line($_);
}

print <<'EOF';
];
EOF

