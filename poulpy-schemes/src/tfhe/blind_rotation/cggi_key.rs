use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyInplace, TakeVecZnx, TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace,
        VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize,
        VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubABInplace, VmpPMatAlloc, VmpPrepare,
    },
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnx, ScalarZnxToRef, Scratch, ZnxView, ZnxViewMut},
    source::Source,
};

use std::marker::PhantomData;

use poulpy_core::{
    Distribution,
    layouts::{
        GGSWCiphertext, LWESecret,
        compressed::GGSWCiphertextCompressed,
        prepared::{GGSWCiphertextPrepared, GLWESecretPrepared},
    },
};

use crate::tfhe::blind_rotation::{
    BlindRotationKey, BlindRotationKeyAlloc, BlindRotationKeyCompressed, BlindRotationKeyEncryptSk, BlindRotationKeyPrepared,
    BlindRotationKeyPreparedAlloc, CGGI,
};

impl BlindRotationKeyAlloc for BlindRotationKey<Vec<u8>, CGGI> {
    fn alloc(n_gglwe: usize, n_lwe: usize, basek: usize, k: usize, rows: usize, rank: usize) -> Self {
        let mut data: Vec<GGSWCiphertext<Vec<u8>>> = Vec::with_capacity(n_lwe);
        (0..n_lwe).for_each(|_| data.push(GGSWCiphertext::alloc(n_gglwe, basek, k, rows, 1, rank)));
        Self {
            keys: data,
            dist: Distribution::NONE,
            _phantom: PhantomData,
        }
    }
}

impl BlindRotationKey<Vec<u8>, CGGI> {
    pub fn generate_from_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes,
    {
        GGSWCiphertext::encrypt_sk_scratch_space(module, basek, k, rank)
    }
}

impl<D: DataMut, B: Backend> BlindRotationKeyEncryptSk<B> for BlindRotationKey<D, CGGI>
where
    Module<B>: VecZnxAddScalarInplace
        + VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxFillUniform
        + VecZnxSubABInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxAddNormal
        + VecZnxNormalize<B>
        + VecZnxSub,
    Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
{
    fn encrypt_sk<DataSkGLWE, DataSkLWE>(
        &mut self,
        module: &Module<B>,
        sk_glwe: &GLWESecretPrepared<DataSkGLWE, B>,
        sk_lwe: &LWESecret<DataSkLWE>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        DataSkGLWE: DataRef,
        DataSkLWE: DataRef,
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
            ggsw.encrypt_sk(module, &pt, sk_glwe, source_xa, source_xe, scratch);
        });
    }
}

impl<B: Backend> BlindRotationKeyPreparedAlloc<B> for BlindRotationKeyPrepared<Vec<u8>, CGGI, B>
where
    Module<B>: VmpPMatAlloc<B> + VmpPrepare<B>,
{
    fn alloc(module: &Module<B>, n_lwe: usize, basek: usize, k: usize, rows: usize, rank: usize) -> Self {
        let mut data: Vec<GGSWCiphertextPrepared<Vec<u8>, B>> = Vec::with_capacity(n_lwe);
        (0..n_lwe).for_each(|_| {
            data.push(GGSWCiphertextPrepared::alloc(
                module, basek, k, rows, 1, rank,
            ))
        });
        Self {
            data,
            dist: Distribution::NONE,
            x_pow_a: None,
            _phantom: PhantomData,
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

    pub fn generate_from_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes,
    {
        GGSWCiphertextCompressed::encrypt_sk_scratch_space(module, basek, k, rank)
    }
}

impl<D: DataMut> BlindRotationKeyCompressed<D, CGGI> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<DataSkGLWE, DataSkLWE, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_glwe: &GLWESecretPrepared<DataSkGLWE, B>,
        sk_lwe: &LWESecret<DataSkLWE>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        DataSkGLWE: DataRef,
        DataSkLWE: DataRef,
        Module<B>: VecZnxAddScalarInplace
            + VecZnxDftAllocBytes
            + VecZnxBigNormalize<B>
            + VecZnxDftApply<B>
            + SvpApplyInplace<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxFillUniform
            + VecZnxSubABInplace
            + VecZnxAddInplace
            + VecZnxNormalizeInplace<B>
            + VecZnxAddNormal
            + VecZnxNormalize<B>
            + VecZnxSub,
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
                scratch,
            );
        });
    }
}
