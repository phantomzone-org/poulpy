//! Open extension points for `poulpy-core`.
//!
//! Backends implement [`CoreImpl`] on their backend marker type to inherit or
//! override the high-level `poulpy-core` algorithms that are exposed through
//! safe traits on [`poulpy_hal::layouts::Module`].

mod automorphism;
mod conversion;
mod core_impl;
mod decryption;
mod encryption;
mod external_product;
mod keyswitching;
mod operations;

pub use automorphism::*;
pub use conversion::*;
pub use core_impl::*;
pub use decryption::*;
pub use encryption::*;
pub use external_product::*;
pub use keyswitching::*;
pub use operations::*;

pub use crate::{
    impl_core_automorphism_default_methods, impl_core_conversion_default_methods, impl_core_decryption_default_methods,
    impl_core_encryption_default_methods, impl_core_external_product_default_methods, impl_core_keyswitch_default_methods,
    impl_core_operations_default_methods,
};

#[macro_export]
macro_rules! impl_core_default_methods {
    ($be:ty) => {
        $crate::impl_core_keyswitch_default_methods!($be);
        $crate::impl_core_external_product_default_methods!($be);
        $crate::impl_core_decryption_default_methods!($be);
        $crate::impl_core_conversion_default_methods!($be);
        $crate::impl_core_automorphism_default_methods!($be);
        $crate::impl_core_operations_default_methods!($be);
        $crate::impl_core_encryption_default_methods!($be);
    };
}

pub use crate::impl_core_default_methods;
