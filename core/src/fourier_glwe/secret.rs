use backend::{Backend, FFT64, Module, ScalarZnxDft, ScalarZnxDftAlloc, ScalarZnxDftOps, ZnxInfos};

use crate::{GLWESecret, dist::Distribution};

pub struct FourierGLWESecret<T, B: Backend> {
    pub(crate) data: ScalarZnxDft<T, B>,
    pub(crate) dist: Distribution,
}

impl<B: Backend> FourierGLWESecret<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, rank: usize) -> Self {
        Self {
            data: module.new_scalar_znx_dft(rank),
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of(module: &Module<B>, rank: usize) -> usize {
        module.bytes_of_scalar_znx_dft(rank)
    }
}

impl FourierGLWESecret<Vec<u8>, FFT64> {
    pub fn from<D>(module: &Module<FFT64>, sk: &GLWESecret<D>) -> Self
    where
        D: AsRef<[u8]>,
    {
        let mut sk_dft: FourierGLWESecret<Vec<u8>, FFT64> = Self::alloc(module, sk.rank());
        sk_dft.set(module, sk);
        sk_dft
    }
}

impl<DataSelf, B: Backend> FourierGLWESecret<DataSelf, B> {
    pub fn n(&self) -> usize {
        self.data.n()
    }

    pub fn log_n(&self) -> usize {
        self.data.log_n()
    }

    pub fn rank(&self) -> usize {
        self.data.cols()
    }
}

impl<S: AsMut<[u8]> + AsRef<[u8]>> FourierGLWESecret<S, FFT64> {
    pub(crate) fn set<D>(&mut self, module: &Module<FFT64>, sk: &GLWESecret<D>)
    where
        D: AsRef<[u8]>,
    {
        (0..self.rank()).for_each(|i| {
            module.svp_prepare(&mut self.data, i, &sk.data, i);
        });
        self.dist = sk.dist
    }
}
