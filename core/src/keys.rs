use backend::{
    Backend, FFT64, Module, ScalarZnx, ScalarZnxAlloc, ScalarZnxDft, ScalarZnxDftAlloc, ScalarZnxDftOps, ScratchOwned, VecZnxDft,
    ZnxInfos, ZnxZero,
};
use sampling::source::Source;

use crate::{elem::Infos, glwe_ciphertext_fourier::GLWECiphertextFourier};

#[derive(Clone, Copy, Debug)]
pub(crate) enum SecretDistribution {
    TernaryFixed(usize), // Ternary with fixed Hamming weight
    TernaryProb(f64),    // Ternary with probabilistic Hamming weight
    ZERO,                // Debug mod
    NONE,
}

pub struct SecretKey<T> {
    pub(crate) data: ScalarZnx<T>,
    pub(crate) dist: SecretDistribution,
}

impl SecretKey<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, rank: usize) -> Self {
        Self {
            data: module.new_scalar_znx(rank),
            dist: SecretDistribution::NONE,
        }
    }

    pub fn bytes_of(module: &Module<FFT64>, rank: usize) -> usize {
        module.bytes_of_scalar_znx(rank + 1)
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

impl<S: AsMut<[u8]> + AsRef<[u8]>> SecretKey<S> {
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

pub struct SecretKeyFourier<T, B: Backend> {
    pub(crate) data: ScalarZnxDft<T, B>,
    pub(crate) dist: SecretDistribution,
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
    pub fn alloc(module: &Module<B>, rank: usize) -> Self {
        Self {
            data: module.new_scalar_znx_dft(rank),
            dist: SecretDistribution::NONE,
        }
    }

    pub fn bytes_of(module: &Module<B>, rank: usize) -> usize {
        module.bytes_of_scalar_znx_dft(rank + 1)
    }
}

impl<D: AsRef<[u8]> + AsMut<[u8]>> SecretKeyFourier<D, FFT64> {
    pub fn dft<S: AsRef<[u8]>>(&mut self, module: &Module<FFT64>, sk: &SecretKey<S>) {
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
            module.svp_prepare(&mut self.data, i, &sk.data, i);
        });
        self.dist = sk.dist;
    }
}

pub struct GLWEPublicKey<D, B: Backend> {
    pub(crate) data: GLWECiphertextFourier<D, B>,
    pub(crate) dist: SecretDistribution,
}

impl<B: Backend> GLWEPublicKey<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rank: usize) -> Self {
        Self {
            data: GLWECiphertextFourier::alloc(module, basek, k, rank),
            dist: SecretDistribution::NONE,
        }
    }

    pub fn bytes_of(module: &Module<B>, basek: usize, k: usize, rank: usize) -> usize {
        GLWECiphertextFourier::<Vec<u8>, B>::bytes_of(module, basek, k, rank)
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

impl<C: AsRef<[u8]> + AsMut<[u8]>> GLWEPublicKey<C, FFT64> {
    pub fn generate_from_sk<S: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        sk_dft: &SecretKeyFourier<S, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
    ) {
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
            scratch.borrow(),
        );
        self.dist = sk_dft.dist;
    }
}
