mod gglwe_atk;
mod gglwe_ct;
mod gglwe_ksk;
mod gglwe_tsk;
mod ggsw_ct;
mod glwe_pk;
mod glwe_sk;
mod glwe_to_lwe_ksk;
mod lwe_ksk;
mod lwe_to_glwe_ksk;

pub use gglwe_atk::*;
pub use gglwe_ct::*;
pub use gglwe_ksk::*;
pub use gglwe_tsk::*;
pub use ggsw_ct::*;
pub use glwe_pk::*;
pub use glwe_sk::*;
pub use glwe_to_lwe_ksk::*;
pub use lwe_ksk::*;
pub use lwe_to_glwe_ksk::*;
use poulpy_hal::layouts::{Backend, Module, Scratch};

pub trait PrepareScratchSpace<B: Backend, T> {
    fn prepare_scratch_space(module: &Module<B>, infos: &T) -> usize;
}

pub trait PrepareAlloc<B: Backend, T> {
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> T;
}

pub trait Prepare<B: Backend, T> {
    fn prepare(&mut self, module: &Module<B>, other: &T, scratch: &mut Scratch<B>);
}
