//! Open Extension Points (OEP) for backend crates.
//!
//! This module defines the `unsafe` trait layer that backend crates implement
//! to provide concrete polynomial arithmetic. Each trait mirrors a corresponding
//! safe trait in the [`crate::api`] module, distinguished by an `Impl` suffix
//! (e.g., [`crate::api::VecZnxAdd`] is backed by `VecZnxAddImpl`).
//!
//! All traits in this module are `unsafe` because implementations must uphold
//! the backend safety contract.

mod convolution;
mod module;
mod scratch;
mod svp_ppol;
mod vec_znx;
mod vec_znx_big;
mod vec_znx_dft;
mod vmp_pmat;

pub use convolution::*;
pub use module::*;
pub use scratch::*;
pub use svp_ppol::*;
pub use vec_znx::*;
pub use vec_znx_big::*;
pub use vec_znx_dft::*;
pub use vmp_pmat::*;
