use backend::{
    Backend, FFT64, Module, ScalarZnx, ScalarZnxAlloc, ScalarZnxDft, ScalarZnxDftAlloc, ScalarZnxDftOps, ScalarZnxToRef, Scratch,
    ZnxView, ZnxViewMut,
};
use sampling::source::Source;

use crate::{Distribution, FourierGLWESecret, GGSWCiphertext, Infos, LWESecret};

pub struct BlindRotationKeyCGGI<D, B: Backend> {
    pub(crate) data: Vec<GGSWCiphertext<D, B>>,
    pub(crate) dist: Distribution,
    pub(crate) x_pow_a: Option<Vec<ScalarZnxDft<Vec<u8>, B>>>,
}

// pub struct BlindRotationKeyFHEW<B: Backend> {
//    pub(crate) data: Vec<GGSWCiphertext<Vec<u8>, B>>,
//    pub(crate) auto: Vec<GLWEAutomorphismKey<Vec<u8>, B>>,
//}

impl BlindRotationKeyCGGI<Vec<u8>, FFT64> {
    pub fn allocate(module: &Module<FFT64>, n_lwe: usize, basek: usize, k: usize, rows: usize, rank: usize) -> Self {
        let mut data: Vec<GGSWCiphertext<Vec<u8>, FFT64>> = Vec::with_capacity(n_lwe);
        (0..n_lwe).for_each(|_| data.push(GGSWCiphertext::alloc(module, basek, k, rows, 1, rank)));
        Self {
            data,
            dist: Distribution::NONE,
            x_pow_a: None::<Vec<ScalarZnxDft<Vec<u8>, FFT64>>>,
        }
    }

    pub fn generate_from_sk_scratch_space(module: &Module<FFT64>, basek: usize, k: usize, rank: usize) -> usize {
        GGSWCiphertext::encrypt_sk_scratch_space(module, basek, k, rank)
    }
}

impl<D: AsRef<[u8]>> BlindRotationKeyCGGI<D, FFT64> {
    #[allow(dead_code)]
    pub(crate) fn n(&self) -> usize {
        self.data[0].n()
    }

    #[allow(dead_code)]
    pub(crate) fn rows(&self) -> usize {
        self.data[0].rows()
    }

    #[allow(dead_code)]
    pub(crate) fn k(&self) -> usize {
        self.data[0].k()
    }

    #[allow(dead_code)]
    pub(crate) fn size(&self) -> usize {
        self.data[0].size()
    }

    #[allow(dead_code)]
    pub(crate) fn rank(&self) -> usize {
        self.data[0].rank()
    }

    pub(crate) fn basek(&self) -> usize {
        self.data[0].basek()
    }

    pub(crate) fn block_size(&self) -> usize {
        match self.dist {
            Distribution::BinaryBlock(value) => value,
            _ => 1,
        }
    }
}

impl<D: AsRef<[u8]> + AsMut<[u8]>> BlindRotationKeyCGGI<D, FFT64> {
    pub fn generate_from_sk<DataSkGLWE, DataSkLWE>(
        &mut self,
        module: &Module<FFT64>,
        sk_glwe: &FourierGLWESecret<DataSkGLWE, FFT64>,
        sk_lwe: &LWESecret<DataSkLWE>,
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
                Distribution::BinaryBlock(_)
                | Distribution::BinaryFixed(_)
                | Distribution::BinaryProb(_)
                | Distribution::ZERO => {}
                _ => panic!(
                    "invalid GLWESecret distribution: must be BinaryBlock, BinaryFixed or BinaryProb (or ZERO for debugging)"
                ),
            }
        }

        self.dist = sk_lwe.dist;

        let mut pt: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
        let sk_ref: ScalarZnx<&[u8]> = sk_lwe.data.to_ref();

        self.data.iter_mut().enumerate().for_each(|(i, ggsw)| {
            pt.at_mut(0, 0)[0] = sk_ref.at(0, 0)[i];
            ggsw.encrypt_sk(module, &pt, sk_glwe, source_xa, source_xe, sigma, scratch);
        });

        match sk_lwe.dist {
            Distribution::BinaryBlock(_) => {
                let mut x_pow_a: Vec<ScalarZnxDft<Vec<u8>, FFT64>> = Vec::with_capacity(module.n() << 1);
                let mut buf: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
                (0..module.n() << 1).for_each(|i| {
                    let mut res: ScalarZnxDft<Vec<u8>, FFT64> = module.new_scalar_znx_dft(1);
                    set_xai_plus_y(module, i, 0, &mut res, &mut buf);
                    x_pow_a.push(res);
                });
                self.x_pow_a = Some(x_pow_a);
            }
            _ => {}
        }
    }
}

pub fn set_xai_plus_y<A, B>(module: &Module<FFT64>, ai: usize, y: i64, res: &mut ScalarZnxDft<A, FFT64>, buf: &mut ScalarZnx<B>)
where
    A: AsRef<[u8]> + AsMut<[u8]>,
    B: AsRef<[u8]> + AsMut<[u8]>,
{
    let n: usize = module.n();

    {
        let raw: &mut [i64] = buf.at_mut(0, 0);
        if ai < n {
            raw[ai] = 1;
        } else {
            raw[(ai - n) & (n - 1)] = -1;
        }
        raw[0] += y;
    }

    module.svp_prepare(res, 0, buf, 0);

    {
        let raw: &mut [i64] = buf.at_mut(0, 0);

        if ai < n {
            raw[ai] = 0;
        } else {
            raw[(ai - n) & (n - 1)] = 0;
        }
        raw[0] = 0;
    }
}
