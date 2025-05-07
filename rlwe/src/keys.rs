use base2k::{
    Backend, FFT64, Module, ScalarZnx, ScalarZnxAlloc, ScalarZnxDft, ScalarZnxDftAlloc, ScalarZnxDftOps, ScalarZnxDftToMut,
    ScalarZnxDftToRef, ScalarZnxToMut, ScalarZnxToRef, Scratch, VecZnxDft, VecZnxDftAlloc, VecZnxDftToMut,
};
use sampling::source::Source;

use crate::elem::derive_size;

pub struct SecretKey<T> {
    pub data: ScalarZnx<T>,
}

impl SecretKey<Vec<u8>> {
    pub fn new<B: Backend>(module: &Module<B>) -> Self {
        Self {
            data: module.new_scalar(1),
        }
    }
}

impl<S> SecretKey<S>
where
    S: AsMut<[u8]> + AsRef<[u8]>,
{
    pub fn fill_ternary_prob(&mut self, prob: f64, source: &mut Source) {
        self.data.fill_ternary_prob(0, prob, source);
    }

    pub fn fill_ternary_hw(&mut self, hw: usize, source: &mut Source) {
        self.data.fill_ternary_hw(0, hw, source);
    }
}

impl<C> ScalarZnxToMut for SecretKey<C>
where
    ScalarZnx<C>: ScalarZnxToMut,
{
    fn to_mut(&mut self) -> ScalarZnx<&mut [u8]> {
        self.data.to_mut()
    }
}

impl<C> ScalarZnxToRef for SecretKey<C>
where
    ScalarZnx<C>: ScalarZnxToRef,
{
    fn to_ref(&self) -> ScalarZnx<&[u8]> {
        self.data.to_ref()
    }
}

pub struct SecretKeyDft<T, B: Backend> {
    pub data: ScalarZnxDft<T, B>,
}

impl<B: Backend> SecretKeyDft<Vec<u8>, B> {
    pub fn new(module: &Module<B>) -> Self {
        Self {
            data: module.new_scalar_znx_dft(1),
        }
    }

    pub fn dft<S>(&mut self, module: &Module<FFT64>, sk: &SecretKey<S>)
    where
        SecretKeyDft<Vec<u8>, B>: ScalarZnxDftToMut<base2k::FFT64>,
        SecretKey<S>: ScalarZnxToRef,
    {
        module.svp_prepare(self, 0, sk, 0)
    }
}

impl<C, B: Backend> ScalarZnxDftToMut<B> for SecretKeyDft<C, B>
where
    ScalarZnxDft<C, B>: ScalarZnxDftToMut<B>,
{
    fn to_mut(&mut self) -> ScalarZnxDft<&mut [u8], B> {
        self.data.to_mut()
    }
}

impl<C, B: Backend> ScalarZnxDftToRef<B> for SecretKeyDft<C, B>
where
    ScalarZnxDft<C, B>: ScalarZnxDftToRef<B>,
{
    fn to_ref(&self) -> ScalarZnxDft<&[u8], B> {
        self.data.to_ref()
    }
}

pub struct PublicKey<D, B: Backend> {
    pub data: VecZnxDft<D, B>,
}

impl<B: Backend> PublicKey<Vec<u8>, B> {
    pub fn new(module: &Module<B>, log_base2k: usize, log_q: usize) -> Self {
        Self {
            data: module.new_vec_znx_dft(2, derive_size(log_base2k, log_q)),
        }
    }
}

impl<B: Backend, D: VecZnxDftToMut<B>> PublicKey<D, B> {
    pub fn generate<S>(&mut self, module: &Module<B>, sk: &SecretKey<ScalarZnxDft<S, B>>, scratch: &mut Scratch)
    where
        ScalarZnxDft<S, B>: ScalarZnxDftToMut<B>,
    {
    }
}
