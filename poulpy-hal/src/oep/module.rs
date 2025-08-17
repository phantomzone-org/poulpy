use crate::layouts::{Backend, Module};

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ModuleNewImpl<B: Backend> {
    fn new_impl(n: u64) -> Module<B>;
}
