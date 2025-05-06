use base2k::{
    Backend, FFT64, Module, Scalar, ScalarAlloc, ScalarZnxDft, ScalarZnxDftOps, ScalarZnxDftToMut, Scratch, VecZnx, VecZnxDft,
    VecZnxDftAlloc, VecZnxDftToMut,
};
use sampling::source::Source;

use crate::elem::derive_size;

pub struct SecretKey<T> {
    data: T,
}

impl<T> SecretKey<T> {
    pub fn data(&self) -> &T {
        &self.data
    }

    pub fn data_mut(&self) -> &mut T {
        &mut self.data
    }
}

impl SecretKey<Scalar<Vec<u8>>> {
    pub fn new<B: Backend>(module: &Module<B>) -> Self {
        Self {
            data: module.new_scalar(1),
        }
    }

    pub fn fill_ternary_prob(&mut self, prob: f64, source: &mut Source) {
        self.data.fill_ternary_prob(0, prob, source);
    }

    pub fn fill_ternary_hw(&mut self, hw: usize, source: &mut Source) {
        self.data.fill_ternary_hw(0, hw, source);
    }

    pub fn svp_prepare<D>(&self, module: &Module<FFT64>, sk_prep: &mut SecretKey<ScalarZnxDft<D, FFT64>>)
    where
        ScalarZnxDft<D, base2k::FFT64>: ScalarZnxDftToMut<base2k::FFT64>,
    {
        module.svp_prepare(&mut sk_prep.data, 0, &self.data, 0)
    }
}

pub struct PublicKey<D, B: Backend> {
    data: VecZnxDft<D, B>,
}

impl<B: Backend> PublicKey<Vec<u8>, B> {
    pub fn new(module: &Module<B>, log_base2k: usize, log_q: usize) -> Self {
        Self {
            data: module.new_vec_znx_dft(2, derive_size(log_base2k, log_q)),
        }
    }
}

impl<B: Backend, D: VecZnxDftToMut<B>> PublicKey<D, B> {
    pub fn generate<S>(&mut self, module: &Module<B>, sk: &SecretKey<ScalarZnxDft<S, B>>)
    where
        ScalarZnxDft<S, B>: ScalarZnxDftToMut<B>,
    {
    }
}
