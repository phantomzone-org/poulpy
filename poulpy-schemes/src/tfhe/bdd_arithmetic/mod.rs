mod bdd_2w_to_1w;
mod bdd_rotation;
mod ciphertexts;
mod circuits;
mod eval;
mod key;

pub use bdd_2w_to_1w::*;
pub use bdd_rotation::*;
pub use ciphertexts::*;
pub(crate) use circuits::*;
pub(crate) use eval::*;
pub use key::*;

pub mod tests;

pub trait UnsignedInteger: Copy + 'static {
    const WORD_SIZE: usize;
}

impl UnsignedInteger for u8 {
    const WORD_SIZE: usize = 8;
}
impl UnsignedInteger for u16 {
    const WORD_SIZE: usize = 16;
}
impl UnsignedInteger for u32 {
    const WORD_SIZE: usize = 32;
}
impl UnsignedInteger for u64 {
    const WORD_SIZE: usize = 64;
}
impl UnsignedInteger for u128 {
    const WORD_SIZE: usize = 128;
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
