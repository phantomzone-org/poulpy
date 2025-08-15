mod cggi_algo;
mod cggi_key;
mod key;
mod key_compressed;
mod key_exec;
mod lut;
mod utils;

pub use cggi_algo::*;
pub use key::*;
pub use key_compressed::*;
pub use key_exec::*;
pub use lut::*;

#[cfg(test)]
mod tests;

use backend::hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};
use core::layouts::{GLWECiphertext, LWECiphertext};

pub trait BlindRotationAlgo {}

#[derive(Clone)]
pub struct CGGI {}
impl BlindRotationAlgo for CGGI {}

pub trait BlincRotationExecute<B: Backend> {
    fn execute<DR: DataMut, DI: DataRef>(
        &self,
        module: &Module<B>,
        res: &mut GLWECiphertext<DR>,
        lwe: &LWECiphertext<DI>,
        lut: &LookUpTable,
        scratch: &mut Scratch<B>,
    );
}
