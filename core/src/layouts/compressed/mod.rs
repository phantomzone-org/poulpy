mod gglwe_atk;
mod gglwe_ct;
mod gglwe_ksk;
mod gglwe_tsk;
mod ggsw_ct;
mod glwe_ct;
mod glwe_to_lwe_ksk;
mod lwe_ct;
mod lwe_ksk;
mod lwe_to_glwe_ksk;

pub use gglwe_atk::*;
pub use gglwe_ct::*;
pub use gglwe_ksk::*;
pub use gglwe_tsk::*;
pub use ggsw_ct::*;
pub use glwe_ct::*;
pub use glwe_to_lwe_ksk::*;
pub use lwe_ct::*;
pub use lwe_ksk::*;
pub use lwe_to_glwe_ksk::*;

use backend::hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{Backend, Module},
};

pub trait Decompress<B: Backend, C> {
    fn decompress(&mut self, module: &Module<B>, other: &C)
    where
        Module<B>: VecZnxFillUniform + VecZnxCopy;
}
