mod circuit;
mod key;
pub mod tests;

pub use circuit::*;
pub use key::*;

use poulpy_core::layouts::{GGSW, LWECiphertext};

use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

pub trait CirtuitBootstrappingExecute<B: Backend> {
    fn execute_to_constant<DM: DataMut, DR: DataRef>(
        &self,
        module: &Module<B>,
        res: &mut GGSW<DM>,
        lwe: &LWECiphertext<DR>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut Scratch<B>,
    );

    #[allow(clippy::too_many_arguments)]
    fn execute_to_exponent<DM: DataMut, DR: DataRef>(
        &self,
        module: &Module<B>,
        log_gap_out: usize,
        res: &mut GGSW<DM>,
        lwe: &LWECiphertext<DR>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut Scratch<B>,
    );
}
