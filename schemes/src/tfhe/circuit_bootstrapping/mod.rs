mod circuit_bootstrapping;
mod key;
pub mod tests;

pub use circuit_bootstrapping::*;
pub use key::*;

use core::layouts::{GGSWCiphertext, LWECiphertext};

use backend::hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

pub trait CirtuitBootstrappingExecute<B: Backend> {
    fn execute_to_constant<DM: DataMut, DR: DataRef>(
        &self,
        module: &Module<B>,
        res: &mut GGSWCiphertext<DM>,
        lwe: &LWECiphertext<DR>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut Scratch<B>,
    );

    fn execute_to_exponent<DM: DataMut, DR: DataRef>(
        &self,
        module: &Module<B>,
        log_gap_out: usize,
        res: &mut GGSWCiphertext<DM>,
        lwe: &LWECiphertext<DR>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut Scratch<B>,
    );
}
