mod gglwe_atk;
mod gglwe_ct;
mod gglwe_ksk;
mod gglwe_tsk;
mod ggsw_ct;
mod glwe_ct;
mod glwe_pk;
mod glwe_pt;
mod glwe_sk;
mod glwe_to_lwe_ksk;
mod lwe_ct;
mod lwe_ksk;
mod lwe_pt;
mod lwe_sk;
mod lwe_to_glwe_ksk;

pub mod compressed;
pub mod prepared;

pub use gglwe_atk::*;
pub use gglwe_ct::*;
pub use gglwe_ksk::*;
pub use gglwe_tsk::*;
pub use ggsw_ct::*;
pub use glwe_ct::*;
pub use glwe_pk::*;
pub use glwe_pt::*;
pub use glwe_sk::*;
pub use glwe_to_lwe_ksk::*;
pub use lwe_ct::*;
pub use lwe_ksk::*;
pub use lwe_pt::*;
pub use lwe_sk::*;
pub use lwe_to_glwe_ksk::*;

#[derive(Debug)]
pub enum BuildError {
    MissingData,
    MissingBase2K,
    MissingK,
    MissingDigits,
    ZeroDegree,
    NonPowerOfTwoDegree,
    ZeroBase2K,
    ZeroTorusPrecision,
    ZeroCols,
    ZeroLimbs,
    ZeroRank,
    ZeroDigits,
    VecZnxColsNotOne,
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
