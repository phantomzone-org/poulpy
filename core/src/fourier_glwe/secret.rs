use backend::{Backend, Module, SvpPPol, SvpPPolAlloc, SvpPPolAllocBytes, SvpPPolPrepare, ZnxInfos};

use crate::{GLWESecret, dist::Distribution};

pub struct FourierGLWESecret<T, B: Backend> {
    pub(crate) data: SvpPPol<T, B>,
    pub(crate) dist: Distribution,
}

impl<B: Backend> FourierGLWESecret<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, rank: usize) -> Self
    where
        Module<B>: SvpPPolAlloc<B>,
    {
        Self {
            data: module.svp_ppol_alloc(rank),
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of(module: &Module<B>, rank: usize) -> usize
    where
        Module<B>: SvpPPolAllocBytes,
    {
        module.svp_ppol_alloc_bytes(rank)
    }
}

impl<B: Backend> FourierGLWESecret<Vec<u8>, B> {
    pub fn from<D>(module: &Module<B>, sk: &GLWESecret<D>) -> Self
    where
        D: AsRef<[u8]>,
        Module<B>: SvpPPolAllocBytes + SvpPPolAlloc<B> + SvpPPolPrepare<B>,
    {
        let mut sk_dft: FourierGLWESecret<Vec<u8>, B> = Self::alloc(module, sk.rank());
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

impl<S: AsMut<[u8]> + AsRef<[u8]>, B: Backend> FourierGLWESecret<S, B> {
    pub(crate) fn set<D>(&mut self, module: &Module<B>, sk: &GLWESecret<D>)
    where
        D: AsRef<[u8]>,
        Module<B>: SvpPPolPrepare<B>,
    {
        (0..self.rank()).for_each(|i| {
            module.svp_prepare(&mut self.data, i, &sk.data, i);
        });
        self.dist = sk.dist
    }
}
