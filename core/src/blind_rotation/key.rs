use backend::{
    Backend, MatZnxAlloc, Module, ScalarZnx, ScalarZnxAlloc, ScalarZnxToRef, Scratch, SvpPPol, SvpPPolAlloc, SvpPrepare, ZnxView,
    ZnxViewMut,
};
use sampling::source::Source;

use crate::{
    Distribution, GGSWCiphertext, GGSWCiphertextExec, GGSWEncryptSkFamily, GGSWLayoutFamily, GLWESecretExec, Infos, LWESecret,
};

pub struct BlindRotationKeyCGGI<D> {
    pub(crate) data: Vec<GGSWCiphertext<D>>,
    pub(crate) dist: Distribution,
}

// pub struct BlindRotationKeyFHEW<B: Backend> {
//    pub(crate) data: Vec<GGSWCiphertext<Vec<u8>, B>>,
//    pub(crate) auto: Vec<GLWEAutomorphismKey<Vec<u8>, B>>,
//}

impl BlindRotationKeyCGGI<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, n_lwe: usize, basek: usize, k: usize, rows: usize, rank: usize) -> Self
    where
        Module<B>: MatZnxAlloc,
    {
        let mut data: Vec<GGSWCiphertext<Vec<u8>>> = Vec::with_capacity(n_lwe);
        (0..n_lwe).for_each(|_| data.push(GGSWCiphertext::alloc(module, basek, k, rows, 1, rank)));
        Self {
            data,
            dist: Distribution::NONE,
        }
    }

    pub fn generate_from_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>: GGSWEncryptSkFamily<B>,
    {
        GGSWCiphertext::encrypt_sk_scratch_space(module, basek, k, rank)
    }
}

impl<D: AsRef<[u8]>> BlindRotationKeyCGGI<D> {
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

    #[allow(dead_code)]
    pub(crate) fn block_size(&self) -> usize {
        match self.dist {
            Distribution::BinaryBlock(value) => value,
            _ => 1,
        }
    }
}

impl<D: AsRef<[u8]> + AsMut<[u8]>> BlindRotationKeyCGGI<D> {
    pub fn generate_from_sk<DataSkGLWE, DataSkLWE, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_glwe: &GLWESecretExec<DataSkGLWE, B>,
        sk_lwe: &LWESecret<DataSkLWE>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        DataSkGLWE: AsRef<[u8]>,
        DataSkLWE: AsRef<[u8]>,
        Module<B>: GGSWEncryptSkFamily<B>,
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

        let mut pt: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);
        let sk_ref: ScalarZnx<&[u8]> = sk_lwe.data.to_ref();

        self.data.iter_mut().enumerate().for_each(|(i, ggsw)| {
            pt.at_mut(0, 0)[0] = sk_ref.at(0, 0)[i];
            ggsw.encrypt_sk(module, &pt, sk_glwe, source_xa, source_xe, sigma, scratch);
        });
    }
}

pub struct BlindRotationKeyCGGIExec<D, B: Backend> {
    pub(crate) data: Vec<GGSWCiphertextExec<D, B>>,
    pub(crate) dist: Distribution,
    pub(crate) x_pow_a: Option<Vec<SvpPPol<Vec<u8>, B>>>,
}

impl<D: AsRef<[u8]>, B: Backend> BlindRotationKeyCGGIExec<D, B> {
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

pub trait BlindRotationKeyCGGIExecLayoutFamily<B: Backend> = GGSWLayoutFamily<B> + SvpPPolAlloc<B> + SvpPrepare<B>;

impl<B: Backend> BlindRotationKeyCGGIExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, n_lwe: usize, basek: usize, k: usize, rows: usize, rank: usize) -> Self
    where
        Module<B>: BlindRotationKeyCGGIExecLayoutFamily<B>,
    {
        let mut data: Vec<GGSWCiphertextExec<Vec<u8>, B>> = Vec::with_capacity(n_lwe);
        (0..n_lwe).for_each(|_| data.push(GGSWCiphertextExec::alloc(module, basek, k, rows, 1, rank)));
        Self {
            data,
            dist: Distribution::NONE,
            x_pow_a: None,
        }
    }

    pub fn from<DataOther>(module: &Module<B>, other: &BlindRotationKeyCGGI<DataOther>, scratch: &mut Scratch) -> Self
    where
        DataOther: AsRef<[u8]>,
        Module<B>: BlindRotationKeyCGGIExecLayoutFamily<B>,
    {
        let mut brk: BlindRotationKeyCGGIExec<Vec<u8>, B> = Self::alloc(
            module,
            other.data.len(),
            other.basek(),
            other.k(),
            other.rows(),
            other.rank(),
        );
        brk.prepare(module, other, scratch);
        brk
    }
}

impl<D: AsRef<[u8]> + AsMut<[u8]>, B: Backend> BlindRotationKeyCGGIExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &BlindRotationKeyCGGI<DataOther>, scratch: &mut Scratch)
    where
        DataOther: AsRef<[u8]>,
        Module<B>: BlindRotationKeyCGGIExecLayoutFamily<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.data.len(), other.data.len());
        }

        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(ggsw_exec, other)| {
                ggsw_exec.prepare(module, other, scratch);
            });

        self.dist = other.dist;

        match other.dist {
            Distribution::BinaryBlock(_) => {
                let mut x_pow_a: Vec<SvpPPol<Vec<u8>, B>> = Vec::with_capacity(module.n() << 1);
                let mut buf: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);
                (0..module.n() << 1).for_each(|i| {
                    let mut res: SvpPPol<Vec<u8>, B> = module.svp_ppol_alloc(1);
                    set_xai_plus_y(module, i, 0, &mut res, &mut buf);
                    x_pow_a.push(res);
                });
                self.x_pow_a = Some(x_pow_a);
            }
            _ => {}
        }
    }
}

pub fn set_xai_plus_y<A, C, B: Backend>(module: &Module<B>, ai: usize, y: i64, res: &mut SvpPPol<A, B>, buf: &mut ScalarZnx<C>)
where
    A: AsRef<[u8]> + AsMut<[u8]>,
    C: AsRef<[u8]> + AsMut<[u8]>,
    Module<B>: SvpPrepare<B>,
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
