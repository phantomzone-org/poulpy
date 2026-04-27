use bytemuck::{Pod, Zeroable};
use rand_distr::num_traits::Zero;
use std::{fmt, ops::Add};

/// Persistent 128-bit big-coefficient storage for the NTT120 GPU backend.
///
/// Stores a signed 128-bit integer as four `u32` words in little-endian
/// two's-complement order: `word[0]` holds bits 0–31, `word[3]` holds
/// bits 96–127.
///
/// Used as `Backend::ScalarBig` for `CudaNtt120Backend`. Device kernels
/// accumulate using transient `__int128` and decompose to `Big32` at the
/// store boundary, keeping all persistent device buffers in 32-bit lanes.
///
/// `size_of::<Big32>() == 16 == size_of::<i128>()`, so
/// `bytes_of_vec_znx_big` uses the default HAL formula with no override.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Big32(pub [u32; 4]);

// SAFETY: Big32 is #[repr(C)] with a single [u32; 4] field.
// All bit patterns are valid; no padding, no uninit bytes.
unsafe impl Zeroable for Big32 {}
unsafe impl Pod for Big32 {}

impl fmt::Display for Big32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:#010x}, {:#010x}, {:#010x}, {:#010x}]",
            self.0[0], self.0[1], self.0[2], self.0[3]
        )
    }
}

// Required by the `Zero` bound on `Backend::ScalarBig`; interprets storage as
// little-endian i128 with wrapping semantics.
impl Add for Big32 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let a: i128 = bytemuck::cast(self);
        let b: i128 = bytemuck::cast(rhs);
        bytemuck::cast(a.wrapping_add(b))
    }
}

impl Zero for Big32 {
    fn zero() -> Self {
        Self([0u32; 4])
    }

    fn is_zero(&self) -> bool {
        self.0 == [0u32; 4]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn size_matches_i128() {
        assert_eq!(size_of::<Big32>(), size_of::<i128>());
    }

    #[test]
    fn align_is_four() {
        assert_eq!(align_of::<Big32>(), align_of::<u32>());
    }

    #[test]
    fn zero_is_all_zero_words() {
        assert_eq!(Big32::zero().0, [0u32; 4]);
    }

    #[test]
    fn pod_roundtrip() {
        let original = Big32([0xDEAD_BEEF, 0xCAFE_BABE, 0x1234_5678, 0xFFFF_0000]);
        let bytes: &[u8] = bytemuck::bytes_of(&original);
        assert_eq!(bytes.len(), 16);
        let recovered: Big32 = *bytemuck::from_bytes(bytes);
        assert_eq!(recovered, original);
    }

    #[test]
    fn little_endian_word_order() {
        // word[0] = bits 0-31 (least significant)
        let val = Big32([1, 0, 0, 0]);
        let bytes: &[u8] = bytemuck::bytes_of(&val);
        // first 4 bytes encode the value 1 in little-endian u32
        assert_eq!(&bytes[0..4], &1u32.to_ne_bytes());
        assert_eq!(&bytes[4..16], &[0u8; 12]);
    }
}
