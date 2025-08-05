use std::io::{Read, Result, Write};

#[derive(Clone, Copy, Debug)]
pub(crate) enum Distribution {
    TernaryFixed(usize), // Ternary with fixed Hamming weight
    TernaryProb(f64),    // Ternary with probabilistic Hamming weight
    BinaryFixed(usize),  // Binary with fixed Hamming weight
    BinaryProb(f64),     // Binary with probabilistic Hamming weight
    BinaryBlock(usize),  // Binary split in block of size 2^k
    ZERO,                // Debug mod
    NONE,                // Unitialized
}

const TAG_TERNARY_FIXED: u8 = 0;
const TAG_TERNARY_PROB: u8 = 1;
const TAG_BINARY_FIXED: u8 = 2;
const TAG_BINARY_PROB: u8 = 3;
const TAG_BINARY_BLOCK: u8 = 4;
const TAG_ZERO: u8 = 5;
const TAG_NONE: u8 = 6;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl Distribution {
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        let word: u64 = match self {
            Distribution::TernaryFixed(v) => (TAG_TERNARY_FIXED as u64) << 56 | (*v as u64),
            Distribution::TernaryProb(p) => {
                let bits = p.to_bits(); // f64 -> u64 bit representation
                (TAG_TERNARY_PROB as u64) << 56 | (bits & 0x00FF_FFFF_FFFF_FFFF)
            }
            Distribution::BinaryFixed(v) => (TAG_BINARY_FIXED as u64) << 56 | (*v as u64),
            Distribution::BinaryProb(p) => {
                let bits = p.to_bits();
                (TAG_BINARY_PROB as u64) << 56 | (bits & 0x00FF_FFFF_FFFF_FFFF)
            }
            Distribution::BinaryBlock(v) => (TAG_BINARY_BLOCK as u64) << 56 | (*v as u64),
            Distribution::ZERO => (TAG_ZERO as u64) << 56,
            Distribution::NONE => (TAG_NONE as u64) << 56,
        };
        writer.write_u64::<LittleEndian>(word)
    }

    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let word = reader.read_u64::<LittleEndian>()?;
        let tag = (word >> 56) as u8;
        let payload = word & 0x00FF_FFFF_FFFF_FFFF;

        let dist = match tag {
            TAG_TERNARY_FIXED => Distribution::TernaryFixed(payload as usize),
            TAG_TERNARY_PROB => Distribution::TernaryProb(f64::from_bits(payload)),
            TAG_BINARY_FIXED => Distribution::BinaryFixed(payload as usize),
            TAG_BINARY_PROB => Distribution::BinaryProb(f64::from_bits(payload)),
            TAG_BINARY_BLOCK => Distribution::BinaryBlock(payload as usize),
            TAG_ZERO => Distribution::ZERO,
            TAG_NONE => Distribution::NONE,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Invalid tag",
                ));
            }
        };
        Ok(dist)
    }
}
