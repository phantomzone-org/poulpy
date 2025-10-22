mod gglwe;
mod ggsw;
mod glwe;
mod glwe_automorphism_key;
mod glwe_plaintext;
mod glwe_public_key;
mod glwe_secret;
mod glwe_switching_key;
mod glwe_tensor_key;
mod glwe_to_lwe_switching_key;
mod lwe;
mod lwe_plaintext;
mod lwe_secret;
mod lwe_switching_key;
mod lwe_to_glwe_switching_key;

pub mod compressed;
pub mod prepared;

pub use compressed::*;
pub use gglwe::*;
pub use ggsw::*;
pub use glwe::*;
pub use glwe_automorphism_key::*;
pub use glwe_plaintext::*;
pub use glwe_public_key::*;
pub use glwe_secret::*;
pub use glwe_switching_key::*;
pub use glwe_tensor_key::*;
pub use glwe_to_lwe_switching_key::*;
pub use lwe::*;
pub use lwe_plaintext::*;
pub use lwe_secret::*;
pub use lwe_switching_key::*;
pub use lwe_to_glwe_switching_key::*;
pub use prepared::*;

use poulpy_hal::layouts::{Backend, Module};

pub trait GetDegree {
    fn ring_degree(&self) -> Degree;
}

impl<B: Backend> GetDegree for Module<B> {
    fn ring_degree(&self) -> Degree {
        Self::n(self).into()
    }
}

/// Newtype over `u32` with arithmetic and comparisons against same type and `u32`.
/// Arithmetic is **saturating** (add/sub/mul) to avoid debug-overflow panics.
macro_rules! newtype_u32 {
    ($name:ident) => {
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $name(pub u32);

        // ----- Conversions -----
        impl From<$name> for u32 {
            #[inline]
            fn from(v: $name) -> u32 {
                v.0
            }
        }
        impl From<$name> for usize {
            #[inline]
            fn from(v: $name) -> usize {
                v.0 as usize
            }
        }

        impl From<u32> for $name {
            #[inline]
            fn from(v: u32) -> $name {
                $name(v)
            }
        }
        impl From<usize> for $name {
            #[inline]
            fn from(v: usize) -> $name {
                $name(v as u32)
            }
        }

        // ----- Display -----
        impl ::core::fmt::Display for $name {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        // ===== Arithmetic (same type) =====
        impl ::core::ops::Add for $name {
            type Output = $name;
            #[inline]
            fn add(self, rhs: $name) -> $name {
                $name(self.0.saturating_add(rhs.0))
            }
        }
        impl ::core::ops::Sub for $name {
            type Output = $name;
            #[inline]
            fn sub(self, rhs: $name) -> $name {
                $name(self.0.saturating_sub(rhs.0))
            }
        }
        impl ::core::ops::Mul for $name {
            type Output = $name;
            #[inline]
            fn mul(self, rhs: $name) -> $name {
                $name(self.0.saturating_mul(rhs.0))
            }
        }

        // ===== Arithmetic (with u32) =====
        impl ::core::ops::Add<u32> for $name {
            type Output = $name;
            #[inline]
            fn add(self, rhs: u32) -> $name {
                $name(self.0.saturating_add(rhs))
            }
        }
        impl ::core::ops::Sub<u32> for $name {
            type Output = $name;
            #[inline]
            fn sub(self, rhs: u32) -> $name {
                $name(self.0.saturating_sub(rhs))
            }
        }
        impl ::core::ops::Mul<u32> for $name {
            type Output = $name;
            #[inline]
            fn mul(self, rhs: u32) -> $name {
                $name(self.0.saturating_mul(rhs))
            }
        }

        impl $name {
            #[inline]
            pub const fn as_u32(self) -> u32 {
                self.0
            }
            #[inline]
            pub const fn as_usize(self) -> usize {
                self.0 as usize
            }

            #[inline]
            pub fn div_ceil<T: Into<u32>>(self, rhs: T) -> u32 {
                self.0.div_ceil(rhs.into())
            }
        }

        // Optional symmetric forms: u32 (+|-|*) $name -> $name
        impl ::core::ops::Add<$name> for u32 {
            type Output = $name;
            #[inline]
            fn add(self, rhs: $name) -> $name {
                $name(self.saturating_add(rhs.0))
            }
        }
        impl ::core::ops::Sub<$name> for u32 {
            type Output = $name;
            #[inline]
            fn sub(self, rhs: $name) -> $name {
                $name(self.saturating_sub(rhs.0))
            }
        }
        impl ::core::ops::Mul<$name> for u32 {
            type Output = $name;
            #[inline]
            fn mul(self, rhs: $name) -> $name {
                $name(self.saturating_mul(rhs.0))
            }
        }

        // ===== Cross-type comparisons with u32 (both directions) =====
        impl ::core::cmp::PartialEq<u32> for $name {
            #[inline]
            fn eq(&self, other: &u32) -> bool {
                self.0 == *other
            }
        }
        impl ::core::cmp::PartialEq<$name> for u32 {
            #[inline]
            fn eq(&self, other: &$name) -> bool {
                *self == other.0
            }
        }

        impl ::core::cmp::PartialOrd<u32> for $name {
            #[inline]
            fn partial_cmp(&self, other: &u32) -> Option<::core::cmp::Ordering> {
                self.0.partial_cmp(other)
            }
        }
        impl ::core::cmp::PartialOrd<$name> for u32 {
            #[inline]
            fn partial_cmp(&self, other: &$name) -> Option<::core::cmp::Ordering> {
                self.partial_cmp(&other.0)
            }
        }
    };
}

newtype_u32!(Degree);
newtype_u32!(TorusPrecision);
newtype_u32!(Base2K);
newtype_u32!(Dnum);
newtype_u32!(Rank);
newtype_u32!(Dsize);

impl Degree {
    pub fn log2(&self) -> usize {
        let n: usize = self.0 as usize;
        (usize::BITS - (n - 1).leading_zeros()) as _
    }
}
