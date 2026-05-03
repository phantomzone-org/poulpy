//! Open extension points for `poulpy-core`.
//!
//! Backends implement the per-family `*Impl` traits exported here to inherit or
//! override the high-level `poulpy-core` algorithms that are exposed through
//! safe traits on [`poulpy_hal::layouts::Module`].

mod automorphism;
mod conversion;
mod decryption;
mod encryption;
mod external_product;
mod keyswitching;
mod operations;

pub use automorphism::*;
pub use conversion::*;
pub use decryption::*;
pub use encryption::*;
pub use external_product::*;
pub use keyswitching::*;
pub use operations::*;

pub use crate::impl_glwe_rotate_impl_from;
