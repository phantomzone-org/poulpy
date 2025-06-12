use backend::{Backend, FFT64, Module, ScalarZnx, ScalarZnxAlloc, ScalarZnxToRef, Scratch, ZnxView, ZnxViewMut};
use sampling::source::Source;

use crate::{AutomorphismKey, GGSWCiphertext, GLWESecret, SecretDistribution};

pub struct BlindRotationKeyCGGI<B: Backend> {
    pub(crate) data: Vec<GGSWCiphertext<Vec<u8>, B>>,
    pub(crate) dist: SecretDistribution,
}

pub struct BlindRotationKeyFHEW<B: Backend> {
    pub(crate) data: Vec<GGSWCiphertext<Vec<u8>, B>>,
    pub(crate) auto: Vec<AutomorphismKey<Vec<u8>, B>>,
}

impl BlindRotationKeyCGGI<FFT64> {
    pub fn allocate(module: &Module<FFT64>, lwe_degree: usize, basek: usize, k: usize, rows: usize, rank: usize) -> Self {
        let mut data: Vec<GGSWCiphertext<Vec<u8>, FFT64>> = Vec::with_capacity(lwe_degree);
        (0..lwe_degree).for_each(|_| data.push(GGSWCiphertext::alloc(module, basek, k, rows, 1, rank)));
        Self {
            data,
            dist: SecretDistribution::NONE,
        }
    }

    pub fn generate_from_sk<DataSkGLWE, DataSkLWE>(
        &mut self,
        module: &Module<FFT64>,
        sk_glwe: &GLWESecret<DataSkGLWE, FFT64>,
        sk_lwe: &GLWESecret<DataSkLWE, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        DataSkGLWE: AsRef<[u8]>,
        DataSkLWE: AsRef<[u8]>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.data.len(), sk_lwe.n());
            assert_eq!(sk_glwe.n(), module.n());
            assert_eq!(sk_glwe.rank(), self.data[0].rank());
            match sk_lwe.dist {
                SecretDistribution::BinaryBlock(_) | SecretDistribution::BinaryFixed(_) | SecretDistribution::BinaryProb(_) => {}
                _ => panic!("invalid GLWESecret distribution: must be BinaryBlock, BinaryFixed or BinaryProb"),
            }
        }

        self.dist = sk_lwe.dist;

        let mut pt: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
        let sk_ref: ScalarZnx<&[u8]> = sk_lwe.data.to_ref();

        self.data.iter_mut().enumerate().for_each(|(i, ggsw)| {
            pt.at_mut(0, 0)[0] = sk_ref.at(0, 0)[i];
            ggsw.encrypt_sk(module, &pt, sk_glwe, source_xa, source_xe, sigma, scratch);
        })
    }
}
