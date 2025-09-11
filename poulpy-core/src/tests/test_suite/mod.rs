pub mod automorphism;
pub mod encryption;
pub mod external_product;
pub mod keyswitch;

mod conversion;
mod packing;
mod trace;

pub use conversion::*;
pub use packing::*;
pub use trace::*;

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

            $(
                $(#[$attr])*
                #[test]
                fn $test_name() {
                    let module = Module::<$backend>::new($size);
                    ($impl)(&module);
                }
            )+
        }
    };
}
