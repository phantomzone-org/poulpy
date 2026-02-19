//! Data layout types and trait definitions for the hardware abstraction layer.
//!
//! This module aggregates all layout-related types and re-exports them from
//! their respective sub-modules, including convolution kernels, matrix and
//! vector representations over polynomial rings, serialization support,
//! statistical utilities, and scratch-space management.
//!
//! It also defines a three-level trait alias hierarchy (`Data`, `DataRef`,
//! `DataMut`) that governs ownership and borrowing semantics for the
//! underlying byte-level data containers used throughout the crate.

mod convolution;
mod encoding;
mod mat_znx;
mod module;
mod scalar_znx;
mod scratch;
mod serialization;
mod stats;
mod svp_ppol;
mod vec_znx;
mod vec_znx_big;
mod vec_znx_dft;
mod vmp_pmat;
mod znx_base;

pub use convolution::*;
pub use mat_znx::*;
pub use module::*;
pub use scalar_znx::*;
pub use scratch::*;
pub use serialization::*;
pub use stats::*;
pub use svp_ppol::*;
pub use vec_znx::*;
pub use vec_znx_big::*;
pub use vec_znx_dft::*;
pub use vmp_pmat::*;
pub use znx_base::*;

/// Base trait alias for all data containers.
///
/// Requires equality comparison ([`PartialEq`], [`Eq`]), a known size at
/// compile time ([`Sized`]), and a default value ([`Default`]). Every
/// layout type that holds raw data must satisfy at least this bound.
pub trait Data = PartialEq + Eq + Sized + Default;

/// Trait alias for read-only (shared) data containers.
///
/// Extends [`Data`] with byte-level shared access via [`AsRef<[u8]>`] and
/// thread-safe sharing via [`Sync`]. Types satisfying this bound can be
/// borrowed immutably and read across threads.
pub trait DataRef = Data + AsRef<[u8]> + Sync;

/// Trait alias for mutable data containers.
///
/// Extends [`DataRef`] with byte-level mutable access via [`AsMut<[u8]>`]
/// and cross-thread transfer via [`Send`]. Types satisfying this bound
/// support in-place modification and can be moved between threads.
pub trait DataMut = DataRef + AsMut<[u8]> + Send;

/// Deep-clone a borrowed layout into a fully owned variant.
///
/// Unlike the standard [`Clone`] trait, `ToOwnedDeep` is intended for
/// types that may borrow their underlying storage. Calling
/// [`to_owned_deep`](ToOwnedDeep::to_owned_deep) produces an independent
/// copy whose lifetime is not tied to the original.
pub trait ToOwnedDeep {
    type Owned;
    fn to_owned_deep(&self) -> Self::Owned;
}

/// Compute a `u64` hash digest of a layout's contents.
///
/// Provides a lightweight fingerprint suitable for fast equality checks
/// and debugging. This is **not** cryptographically secure; it is a
/// convenience mechanism for detecting whether two values hold identical
/// data without performing a full byte-by-byte comparison.
pub trait DigestU64 {
    fn digest_u64(&self) -> u64;
}
