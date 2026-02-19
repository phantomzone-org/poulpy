//! Backend extension point for [`Module`] construction.

use crate::layouts::{Backend, Module};

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See the [poulpy-backend/src/cpu_fft64_ref/module.rs](https://github.com/phantomzone-org/poulpy/blob/main/poulpy-backend/src/cpu_fft64_ref/module.rs) reference implementation.
/// * See [crate::api::ModuleNew] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ModuleNewImpl<B: Backend> {
    fn new_impl(n: u64) -> Module<B>;
}
