//! Open Extension Points (OEP) for backend crates.
//!
//! This module defines the `unsafe` backend extension layer centered on
//! [`crate::oep::HalImpl`]. Backend crates implement `HalImpl` directly.
//!
//! All extension points in this module are `unsafe` because implementations
//! must uphold the backend safety contract.

mod hal_impl;

pub use hal_impl::*;
