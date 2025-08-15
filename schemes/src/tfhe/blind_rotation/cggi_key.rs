use backend::hal::{
    api::{
        ScratchAvailable, SvpPPolAlloc, TakeVecZnx, TakeVecZnxDft, VecZnxAddScalarInplace, VmpPMatAlloc, VmpPMatPrepare, ZnxView,
        ZnxViewMut,
    },
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnx, ScalarZnxToRef, Scratch, SvpPPol},
};
use sampling::source::Source;

use std::marker::PhantomData;

use core::{
    Distribution,
    layouts::{
        GGSWCiphertext, LWESecret,
        compressed::GGSWCiphertextCompressed,
        prepared::{GGSWCiphertextExec, GLWESecretExec},
    },
    trait_families::GGSWEncryptSkFamily,
};

use crate::tfhe::blind_rotation::{
    BlindRotationKey, BlindRotationKeyCompressed, BlindRotationKeyExec, BlindRotationKeyExecLayoutFamily, CGGI,
    utils::set_xai_plus_y,
};

impl BlindRotationKey<Vec<u8>, CGGI> {
    pub fn alloc(n_gglwe: usize, n_lwe: usize, basek: usize, k: usize, rows: usize, rank: usize) -> Self {
        let mut data: Vec<GGSWCiphertext<Vec<u8>>> = Vec::with_capacity(n_lwe);
        (0..n_lwe).for_each(|_| data.push(GGSWCiphertext::alloc(n_gglwe, basek, k, rows, 1, rank)));
        Self {
            keys: data,
            dist: Distribution::NONE,
            _phantom: PhantomData,
        }
    }

    pub fn generate_from_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>: GGSWEncryptSkFamily<B>,
    {
        GGSWCiphertext::encrypt_sk_scratch_space(module, n, basek, k, rank)
    }
}

impl<D: DataMut> BlindRotationKey<D, CGGI> {
    pub fn generate_from_sk<DataSkGLWE, DataSkLWE, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_glwe: &GLWESecretExec<DataSkGLWE, B>,
        sk_lwe: &LWESecret<DataSkLWE>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        DataSkGLWE: DataRef,
        DataSkLWE: DataRef,
        Module<B>: GGSWEncryptSkFamily<B> + VecZnxAddScalarInplace,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.keys.len(), sk_lwe.n());
            assert!(sk_glwe.n() <= module.n());
            assert_eq!(sk_glwe.rank(), self.keys[0].rank());
            match sk_lwe.dist() {
                Distribution::BinaryBlock(_)
                | Distribution::BinaryFixed(_)
                | Distribution::BinaryProb(_)
                | Distribution::ZERO => {}
                _ => panic!(
                    "invalid GLWESecret distribution: must be BinaryBlock, BinaryFixed or BinaryProb (or ZERO for debugging)"
                ),
            }
        }

        self.dist = sk_lwe.dist();

        let mut pt: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(sk_glwe.n(), 1);
        let sk_ref: ScalarZnx<&[u8]> = sk_lwe.data().to_ref();

        self.keys.iter_mut().enumerate().for_each(|(i, ggsw)| {
            pt.at_mut(0, 0)[0] = sk_ref.at(0, 0)[i];
            ggsw.encrypt_sk(module, &pt, sk_glwe, source_xa, source_xe, sigma, scratch);
        });
    }
}

impl<B: Backend> BlindRotationKeyExec<Vec<u8>, CGGI, B> {
    pub fn alloc(module: &Module<B>, n_glwe: usize, n_lwe: usize, basek: usize, k: usize, rows: usize, rank: usize) -> Self
    where
        Module<B>: BlindRotationKeyExecLayoutFamily<B> + VmpPMatAlloc<B> + VmpPMatPrepare<B>,
    {
        let mut data: Vec<GGSWCiphertextExec<Vec<u8>, B>> = Vec::with_capacity(n_lwe);
        (0..n_lwe).for_each(|_| {
            data.push(GGSWCiphertextExec::alloc(
                module, n_glwe, basek, k, rows, 1, rank,
            ))
        });
        Self {
            data,
            dist: Distribution::NONE,
            x_pow_a: None,
            _phantom: PhantomData,
        }
    }

    pub fn from<DataOther>(module: &Module<B>, other: &BlindRotationKey<DataOther, CGGI>, scratch: &mut Scratch<B>) -> Self
    where
        DataOther: DataRef,
        Module<B>: BlindRotationKeyExecLayoutFamily<B> + VmpPMatAlloc<B> + VmpPMatPrepare<B>,
    {
        let mut brk: BlindRotationKeyExec<Vec<u8>, CGGI, B> = Self::alloc(
            module,
            other.n(),
            other.keys.len(),
            other.basek(),
            other.k(),
            other.rows(),
            other.rank(),
        );
        brk.prepare(module, other, scratch);
        brk
    }
}

impl<D: DataMut, B: Backend> BlindRotationKeyExec<D, CGGI, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &BlindRotationKey<DataOther, CGGI>, scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: BlindRotationKeyExecLayoutFamily<B> + VmpPMatAlloc<B> + VmpPMatPrepare<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.data.len(), other.keys.len());
        }

        let n: usize = other.n();

        self.data
            .iter_mut()
            .zip(other.keys.iter())
            .for_each(|(ggsw_exec, other)| {
                ggsw_exec.prepare(module, other, scratch);
            });

        self.dist = other.dist;

        match other.dist {
            Distribution::BinaryBlock(_) => {
                let mut x_pow_a: Vec<SvpPPol<Vec<u8>, B>> = Vec::with_capacity(n << 1);
                let mut buf: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
                (0..n << 1).for_each(|i| {
                    let mut res: SvpPPol<Vec<u8>, B> = module.svp_ppol_alloc(n, 1);
                    set_xai_plus_y(module, i, 0, &mut res, &mut buf);
                    x_pow_a.push(res);
                });
                self.x_pow_a = Some(x_pow_a);
            }
            _ => {}
        }
    }
}

impl BlindRotationKeyCompressed<Vec<u8>, CGGI> {
    pub fn alloc(n_gglwe: usize, n_lwe: usize, basek: usize, k: usize, rows: usize, rank: usize) -> Self {
        let mut data: Vec<GGSWCiphertextCompressed<Vec<u8>>> = Vec::with_capacity(n_lwe);
        (0..n_lwe).for_each(|_| {
            data.push(GGSWCiphertextCompressed::alloc(
                n_gglwe, basek, k, rows, 1, rank,
            ))
        });
        Self {
            keys: data,
            dist: Distribution::NONE,
            _phantom: PhantomData,
        }
    }

    pub fn generate_from_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>: GGSWEncryptSkFamily<B>,
    {
        GGSWCiphertextCompressed::encrypt_sk_scratch_space(module, n, basek, k, rank)
    }
}

impl<D: DataMut> BlindRotationKeyCompressed<D, CGGI> {
    pub fn generate_from_sk<DataSkGLWE, DataSkLWE, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_glwe: &GLWESecretExec<DataSkGLWE, B>,
        sk_lwe: &LWESecret<DataSkLWE>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        DataSkGLWE: DataRef,
        DataSkLWE: DataRef,
        Module<B>: GGSWEncryptSkFamily<B> + VecZnxAddScalarInplace,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.keys.len(), sk_lwe.n());
            assert!(sk_glwe.n() <= module.n());
            assert_eq!(sk_glwe.rank(), self.keys[0].rank());
            match sk_lwe.dist() {
                Distribution::BinaryBlock(_)
                | Distribution::BinaryFixed(_)
                | Distribution::BinaryProb(_)
                | Distribution::ZERO => {}
                _ => panic!(
                    "invalid GLWESecret distribution: must be BinaryBlock, BinaryFixed or BinaryProb (or ZERO for debugging)"
                ),
            }
        }

        self.dist = sk_lwe.dist();

        let mut pt: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(sk_glwe.n(), 1);
        let sk_ref: ScalarZnx<&[u8]> = sk_lwe.data().to_ref();

        let mut source_xa: Source = Source::new(seed_xa);

        self.keys.iter_mut().enumerate().for_each(|(i, ggsw)| {
            pt.at_mut(0, 0)[0] = sk_ref.at(0, 0)[i];
            ggsw.encrypt_sk(
                module,
                &pt,
                sk_glwe,
                source_xa.new_seed(),
                source_xe,
                sigma,
                scratch,
            );
        });
    }
}
