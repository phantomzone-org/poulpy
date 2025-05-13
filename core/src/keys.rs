use base2k::{
    Backend, FFT64, Module, ScalarZnx, ScalarZnxAlloc, ScalarZnxDft, ScalarZnxDftAlloc, ScalarZnxDftOps, ScalarZnxDftToMut,
    ScalarZnxDftToRef, ScalarZnxToMut, ScalarZnxToRef, ScratchOwned, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, ZnxInfos,
    ZnxZero,
};
use sampling::source::Source;

use crate::{elem::Infos, glwe_ciphertext_fourier::GLWECiphertextFourier};

#[derive(Clone, Copy, Debug)]
pub enum SecretDistribution {
    TernaryFixed(usize), // Ternary with fixed Hamming weight
    TernaryProb(f64),    // Ternary with probabilistic Hamming weight
    ZERO,                // Debug mod
    NONE,
}

pub struct SecretKey<T> {
    pub data: ScalarZnx<T>,
    pub dist: SecretDistribution,
}

impl SecretKey<Vec<u8>> {
    pub fn new<B: Backend>(module: &Module<B>, rank: usize) -> Self {
        Self {
            data: module.new_scalar_znx(rank),
            dist: SecretDistribution::NONE,
        }
    }
}

impl<DataSelf> SecretKey<DataSelf> {
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

impl<S> SecretKey<S>
where
    S: AsMut<[u8]> + AsRef<[u8]>,
{
    pub fn fill_ternary_prob(&mut self, prob: f64, source: &mut Source) {
        (0..self.rank()).for_each(|i| {
            self.data.fill_ternary_prob(i, prob, source);
        });
        self.dist = SecretDistribution::TernaryProb(prob);
    }

    pub fn fill_ternary_hw(&mut self, hw: usize, source: &mut Source) {
        (0..self.rank()).for_each(|i| {
            self.data.fill_ternary_hw(i, hw, source);
        });
        self.dist = SecretDistribution::TernaryFixed(hw);
    }

    pub fn fill_zero(&mut self) {
        self.data.zero();
        self.dist = SecretDistribution::ZERO;
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

pub struct SecretKeyFourier<T, B: Backend> {
    pub data: ScalarZnxDft<T, B>,
    pub dist: SecretDistribution,
}

impl<DataSelf, B: Backend> SecretKeyFourier<DataSelf, B> {
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

impl<B: Backend> SecretKeyFourier<Vec<u8>, B> {
    pub fn new(module: &Module<B>, rank: usize) -> Self {
        Self {
            data: module.new_scalar_znx_dft(rank),
            dist: SecretDistribution::NONE,
        }
    }

    pub fn dft<S>(&mut self, module: &Module<FFT64>, sk: &SecretKey<S>)
    where
        SecretKeyFourier<Vec<u8>, B>: ScalarZnxDftToMut<base2k::FFT64>,
        SecretKey<S>: ScalarZnxToRef,
    {
        #[cfg(debug_assertions)]
        {
            match sk.dist {
                SecretDistribution::NONE => panic!("invalid sk: SecretDistribution::NONE"),
                _ => {}
            }

            assert_eq!(self.n(), module.n());
            assert_eq!(sk.n(), module.n());
            assert_eq!(self.rank(), sk.rank());
        }

        (0..self.rank()).for_each(|i| {
            module.svp_prepare(self, i, sk, i);
        });
        self.dist = sk.dist;
    }
}

impl<C, B: Backend> ScalarZnxDftToMut<B> for SecretKeyFourier<C, B>
where
    ScalarZnxDft<C, B>: ScalarZnxDftToMut<B>,
{
    fn to_mut(&mut self) -> ScalarZnxDft<&mut [u8], B> {
        self.data.to_mut()
    }
}

impl<C, B: Backend> ScalarZnxDftToRef<B> for SecretKeyFourier<C, B>
where
    ScalarZnxDft<C, B>: ScalarZnxDftToRef<B>,
{
    fn to_ref(&self) -> ScalarZnxDft<&[u8], B> {
        self.data.to_ref()
    }
}

pub struct GLWEPublicKey<D, B: Backend> {
    pub data: GLWECiphertextFourier<D, B>,
    pub dist: SecretDistribution,
}

impl<B: Backend> GLWEPublicKey<Vec<u8>, B> {
    pub fn new(module: &Module<B>, log_base2k: usize, log_k: usize, rank: usize) -> Self {
        Self {
            data: GLWECiphertextFourier::new(module, log_base2k, log_k, rank),
            dist: SecretDistribution::NONE,
        }
    }
}

impl<T, B: Backend> Infos for GLWEPublicKey<T, B> {
    type Inner = VecZnxDft<T, B>;

    fn inner(&self) -> &Self::Inner {
        &self.data.data
    }

    fn basek(&self) -> usize {
        self.data.basek
    }

    fn k(&self) -> usize {
        self.data.k
    }
}

impl<T, B: Backend> GLWEPublicKey<T, B> {
    pub fn rank(&self) -> usize {
        self.cols() - 1
    }
}

impl<C, B: Backend> VecZnxDftToMut<B> for GLWEPublicKey<C, B>
where
    VecZnxDft<C, B>: VecZnxDftToMut<B>,
{
    fn to_mut(&mut self) -> VecZnxDft<&mut [u8], B> {
        self.data.to_mut()
    }
}

impl<C, B: Backend> VecZnxDftToRef<B> for GLWEPublicKey<C, B>
where
    VecZnxDft<C, B>: VecZnxDftToRef<B>,
{
    fn to_ref(&self) -> VecZnxDft<&[u8], B> {
        self.data.to_ref()
    }
}

impl<C> GLWEPublicKey<C, FFT64> {
    pub fn generate<S>(
        &mut self,
        module: &Module<FFT64>,
        sk_dft: &SecretKeyFourier<S, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        VecZnxDft<C, FFT64>: VecZnxDftToMut<FFT64> + VecZnxDftToRef<FFT64>,
        ScalarZnxDft<S, FFT64>: ScalarZnxDftToRef<FFT64> + ZnxInfos,
    {
        #[cfg(debug_assertions)]
        {
            match sk_dft.dist {
                SecretDistribution::NONE => panic!("invalid sk_dft: SecretDistribution::NONE"),
                _ => {}
            }
        }

        // Its ok to allocate scratch space here since pk is usually generated only once.
        let mut scratch: ScratchOwned = ScratchOwned::new(GLWECiphertextFourier::encrypt_sk_scratch_space(
            module,
            self.rank(),
            self.size(),
        ));
        self.data.encrypt_zero_sk(
            module,
            sk_dft,
            source_xa,
            source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );
        self.dist = sk_dft.dist;
    }
}
