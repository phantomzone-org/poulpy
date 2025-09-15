pub mod serialization;
pub mod svp;
pub mod vec_znx;
pub mod vec_znx_big;
pub mod vec_znx_dft;
pub mod vmp;

#[macro_export]
macro_rules! backend_test_suite {
    (
        mod $modname:ident,
        backend = $backend:ty,
        size = $size:expr,
        tests = {
            $( $(#[$attr:meta])* $test_name:ident => $impl:path ),+ $(,)?
        }
    ) => {
        mod $modname {
            use poulpy_hal::{api::ModuleNew, layouts::Module};

            use once_cell::sync::Lazy;

            static MODULE: Lazy<Module<$backend>> =
                Lazy::new(|| Module::<$backend>::new($size));

            $(
                $(#[$attr])*
                #[test]
                fn $test_name() {
                    ($impl)(&*MODULE);
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
        size = $size:expr,
        basek = $basek:expr,
        tests = {
            $( $(#[$attr:meta])* $test_name:ident => $impl:path ),+ $(,)?
        }
    ) => {
        mod $modname {
            use poulpy_hal::{api::ModuleNew, layouts::Module};

            use once_cell::sync::Lazy;

            static MODULE_REF: Lazy<Module<$backend_ref>> =
                Lazy::new(|| Module::<$backend_ref>::new($size));
            static MODULE_TEST: Lazy<Module<$backend_test>> =
                Lazy::new(|| Module::<$backend_test>::new($size));

            $(
                $(#[$attr])*
                #[test]
                fn $test_name() {
                    ($impl)($basek, &*MODULE_REF, &*MODULE_TEST);
                }
            )+
        }
    };
}
