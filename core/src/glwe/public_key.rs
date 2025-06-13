use backend::{Backend, FFT64, Module, ScratchOwned, VecZnxDft};
use sampling::source::Source;

use crate::{FourierGLWECiphertext, FourierGLWESecret, Infos, dist::Distribution};

pub struct GLWEPublicKey<D, B: Backend> {
    pub(crate) data: FourierGLWECiphertext<D, B>,
    pub(crate) dist: Distribution,
}

impl<B: Backend> GLWEPublicKey<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rank: usize) -> Self {
        Self {
            data: FourierGLWECiphertext::alloc(module, basek, k, rank),
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of(module: &Module<B>, basek: usize, k: usize, rank: usize) -> usize {
        FourierGLWECiphertext::<Vec<u8>, B>::bytes_of(module, basek, k, rank)
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
        sk: &FourierGLWESecret<S, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
    ) {
        #[cfg(debug_assertions)]
        {
            match sk.dist {
                Distribution::NONE => panic!("invalid sk: SecretDistribution::NONE"),
                _ => {}
            }
        }

        // Its ok to allocate scratch space here since pk is usually generated only once.
        let mut scratch: ScratchOwned = ScratchOwned::new(FourierGLWECiphertext::encrypt_sk_scratch_space(
            module,
            self.basek(),
            self.k(),
            self.rank(),
        ));

        self.data
            .encrypt_zero_sk(module, sk, source_xa, source_xe, sigma, scratch.borrow());
        self.dist = sk.dist;
    }
}
