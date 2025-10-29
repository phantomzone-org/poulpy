mod bdd_2w_to_1w;
mod blind_rotation;
mod ciphertexts;
mod circuits;
mod eval;
mod key;

pub use bdd_2w_to_1w::*;
pub use blind_rotation::*;
pub use ciphertexts::*;
pub(crate) use circuits::*;
pub(crate) use eval::*;
pub use key::*;

pub mod tests;

pub trait UnsignedInteger: Copy + 'static {
    const BITS: u32;
    const LOG_BITS: u32;
    const LOG_BYTES: u32;
    const LOG_BYTES_MASK: usize;

    #[inline(always)]
    fn bit_index(i: usize) -> usize {
        ((i & Self::LOG_BYTES_MASK) << 3) | (i >> Self::LOG_BYTES)
    }
}

impl UnsignedInteger for u8 {
    const BITS: u32 = u8::BITS;
    const LOG_BITS: u32 = (u32::BITS - (Self::BITS - 1).leading_zeros());
    const LOG_BYTES: u32 = Self::LOG_BITS - 3;
    const LOG_BYTES_MASK: usize = (1 << Self::LOG_BYTES) - 1;
}
impl UnsignedInteger for u16 {
    const BITS: u32 = u16::BITS;
    const LOG_BITS: u32 = (u32::BITS - (Self::BITS - 1).leading_zeros());
    const LOG_BYTES: u32 = Self::LOG_BITS - 3;
    const LOG_BYTES_MASK: usize = (1 << Self::LOG_BYTES) - 1;
}
impl UnsignedInteger for u32 {
    const BITS: u32 = u32::BITS;
    const LOG_BITS: u32 = (u32::BITS - (Self::BITS - 1).leading_zeros());
    const LOG_BYTES: u32 = Self::LOG_BITS - 3;
    const LOG_BYTES_MASK: usize = (1 << Self::LOG_BYTES) - 1;
}
impl UnsignedInteger for u64 {
    const BITS: u32 = u64::BITS;
    const LOG_BITS: u32 = (u32::BITS - (Self::BITS - 1).leading_zeros());
    const LOG_BYTES: u32 = Self::LOG_BITS >> 3;
    const LOG_BYTES_MASK: usize = (1 << Self::LOG_BYTES) - 1;
}
impl UnsignedInteger for u128 {
    const BITS: u32 = u128::BITS;
    const LOG_BITS: u32 = (u32::BITS - (Self::BITS - 1).leading_zeros());
    const LOG_BYTES: u32 = Self::LOG_BITS >> 3;
    const LOG_BYTES_MASK: usize = (1 << Self::LOG_BYTES) - 1;
}

pub trait ToBits {
    fn bit(&self, i: usize) -> u8;
}

macro_rules! impl_tobits {
    ($($t:ty),*) => {
        $(
            impl ToBits for $t {
                fn bit(&self, i: usize) -> u8 {
                    if i >= (std::mem::size_of::<$t>() * 8) {
                        panic!("bit index {} out of range for {}", i, stringify!($t));
                    }
                    ((self >> i) & 1) as u8
                }
            }
        )*
    };
}

impl_tobits!(u8, u16, u32, u64, u128);

pub trait FromBits: Sized {
    fn from_bits(bits: &[u8]) -> Self;
}

macro_rules! impl_from_bits {
    ($($t:ty),*) => {
        $(
            impl FromBits for $t {
                fn from_bits(bits: &[u8]) -> Self {
                    let mut value: $t = 0;
                    let max_bits = std::mem::size_of::<$t>() * 8;
                    let n = bits.len().min(max_bits);

                    for (i, &bit) in bits.iter().take(n).enumerate() {
                        if bit != 0 {
                            value |= 1 << i;
                        }
                    }
                    value
                }
            }
        )*
    };
}

impl_from_bits!(u8, u16, u32, u64, u128);
