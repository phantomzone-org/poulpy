use backend::hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyInplace, VecZnxBigNormalize, VecZnxDftAlloc, VecZnxDftAllocBytes,
        VecZnxDftFromVecZnx, VecZnxDftToVecZnxBigConsume,
    },
    layouts::{Backend, Module, ScratchOwned, VecZnxDft},
    oep::ScratchTakeVecZnxDftImpl,
};
use sampling::source::Source;

use crate::{GLWECiphertext, GLWESecretExec, Infos, dist::Distribution};

pub trait GLWEPublicKeyFamily<B: Backend> = VecZnxDftAlloc<B>
    + VecZnxDftAllocBytes
    + VecZnxBigNormalize<B>
    + VecZnxDftFromVecZnx<B>
    + SvpApplyInplace<B>
    + VecZnxDftToVecZnxBigConsume<B>;

pub struct GLWEPublicKey<D, B: Backend> {
    pub(crate) data: VecZnxDft<D, B>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) dist: Distribution,
}

impl<B: Backend> GLWEPublicKey<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rank: usize) -> Self
    where
        Module<B>: GLWEPublicKeyFamily<B>,
    {
        Self {
            data: module.vec_znx_dft_alloc(rank + 1, k.div_ceil(basek)),
            basek: basek,
            k: k,
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of(module: &Module<B>, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>: GLWEPublicKeyFamily<B>,
    {
        module.vec_znx_dft_alloc_bytes(rank + 1, k.div_ceil(basek))
    }
}

impl<T, B: Backend> Infos for GLWEPublicKey<T, B> {
    type Inner = VecZnxDft<T, B>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.basek
    }

    fn k(&self) -> usize {
        self.k
    }
}

impl<T, B: Backend> GLWEPublicKey<T, B> {
    pub fn rank(&self) -> usize {
        self.cols() - 1
    }
}

impl<C: AsRef<[u8]> + AsMut<[u8]>, B: Backend> GLWEPublicKey<C, B> {
    pub fn generate_from_sk<S: AsRef<[u8]>>(
        &mut self,
        module: &Module<B>,
        sk: &GLWESecretExec<S, B>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
    ) where
        Module<B>: GLWEPublicKeyFamily<B>,
        B: ScratchTakeVecZnxDftImpl<B>,
    {
        #[cfg(debug_assertions)]
        {
            match sk.dist {
                Distribution::NONE => panic!("invalid sk: SecretDistribution::NONE"),
                _ => {}
            }
        }

        // Its ok to allocate scratch space here since pk is usually generated only once.
        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GLWECiphertext::encrypt_sk_scratch_space(
            module,
            self.basek(),
            self.k(),
        ));

        let mut tmp: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, self.basek(), self.k(), self.rank());
        tmp.encrypt_zero_sk(module, sk, source_xa, source_xe, sigma, scratch.borrow());
        (0..self.cols()).for_each(|i| {
            module.vec_znx_dft_from_vec_znx(1, 0, &mut self.data, i, &tmp.data, i);
        });
        self.dist = sk.dist;
    }
}
