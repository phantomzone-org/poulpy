//! Backend-parametric test functions.
//!
//! Provides fully generic test functions that can be instantiated for any
//! backend via the [`backend_test_suite!`](crate::backend_test_suite) and
//! [`cross_backend_test_suite!`](crate::cross_backend_test_suite) macros.
//! Tests validate correctness against the [`crate::reference`] implementation.

pub mod convolution;
pub mod serialization;
pub mod svp;
pub mod vec_znx;
pub mod vec_znx_big;
pub mod vec_znx_dft;
pub mod vmp;

/// Parameters passed to every test function in a [`backend_test_suite!`] or
/// [`cross_backend_test_suite!`].
///
/// Centralising these values at the macro call-site makes it possible to
/// instantiate the same test suite with backend-appropriate parameters
/// (e.g. different `base2k` for FFT64 vs NTT120).
#[derive(Clone, Copy, Debug)]
pub struct TestParams {
    /// Ring degree N (polynomial degree).
    pub size: usize,
    /// Primary decomposition base (limbs are base-2^`base2k`).
    ///
    /// Secondary base values used inside individual tests are derived from
    /// this value via fixed offsets that preserve the original relative
    /// relationships between bases.
    pub base2k: usize,
}

#[macro_export]
macro_rules! backend_test_suite {
    (
        mod $modname:ident,
        backend = $backend:ty,
        params = $params:expr,
        tests = {
            $( $(#[$attr:meta])* $test_name:ident => $impl:path ),+ $(,)?
        }
    ) => {
        mod $modname {
            use poulpy_hal::{api::ModuleNew, layouts::Module, test_suite::TestParams};

            use once_cell::sync::Lazy;

            static PARAMS: Lazy<TestParams> = Lazy::new(|| $params);
            static MODULE: Lazy<Module<$backend>> =
                Lazy::new(|| Module::<$backend>::new(PARAMS.size as u64));

            $(
                $(#[$attr])*
                #[test]
                fn $test_name() {
                    ($impl)(&*PARAMS, &*MODULE);
                }
            )+
        }
    };
}

#[macro_export]
macro_rules! cross_backend_test_suite {
    (
        mod $modname:ident,
        backend_ref = $backend_ref:ty,
        backend_test = $backend_test:ty,
        params = $params:expr,
        tests = {
            $( $(#[$attr:meta])* $test_name:ident => $impl:path ),+ $(,)?
        }
    ) => {
        mod $modname {
            use poulpy_hal::{api::ModuleNew, layouts::Module, test_suite::TestParams};

            use once_cell::sync::Lazy;

            static PARAMS: Lazy<TestParams> = Lazy::new(|| $params);
            static MODULE_REF: Lazy<Module<$backend_ref>> =
                Lazy::new(|| Module::<$backend_ref>::new(PARAMS.size as u64));
            static MODULE_TEST: Lazy<Module<$backend_test>> =
                Lazy::new(|| Module::<$backend_test>::new(PARAMS.size as u64));

            $(
                $(#[$attr])*
                #[test]
                fn $test_name() {
                    ($impl)(&*PARAMS, &*MODULE_REF, &*MODULE_TEST);
                }
            )+
        }
    };
}
