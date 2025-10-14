use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyDftToDftInplace, TakeVecZnx, TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal,
        VecZnxAddScalarInplace, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxFillUniform, VecZnxIdftApplyConsume,
        VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace, VmpPMatAlloc, VmpPrepare,
    },
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnx, ScalarZnxToRef, Scratch, ZnxView, ZnxViewMut},
    source::Source,
};

use std::marker::PhantomData;

use poulpy_core::{
    Distribution,
    layouts::{
        GGSW, GGSWInfos, LWESecret,
        compressed::GGSWCompressed,
        prepared::{GGSWPrepared, GLWESecretPrepared},
    },
};

use crate::tfhe::blind_rotation::{
    BlindRotationKey, BlindRotationKeyAlloc, BlindRotationKeyCompressed, BlindRotationKeyEncryptSk, BlindRotationKeyInfos,
    BlindRotationKeyPrepared, BlindRotationKeyPreparedAlloc, CGGI,
};

impl BlindRotationKeyAlloc for BlindRotationKey<Vec<u8>, CGGI> {
    fn alloc<A>(infos: &A) -> Self
    where
        A: BlindRotationKeyInfos,
    {
        let mut data: Vec<GGSW<Vec<u8>>> = Vec::with_capacity(infos.n_lwe().into());
        for _ in 0..infos.n_lwe().as_usize() {
            data.push(GGSW::alloc_from_infos(infos));
        }

        Self {
            keys: data,
            dist: Distribution::NONE,
            _phantom: PhantomData,
        }
    }
}

impl BlindRotationKey<Vec<u8>, CGGI> {
    pub fn generate_from_sk_scratch_space<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGSWInfos,
        Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftBytesOf,
    {
        GGSW::encrypt_sk_scratch_space(module, infos)
    }
}

impl<D: DataMut, B: Backend> BlindRotationKeyEncryptSk<B> for BlindRotationKey<D, CGGI>
where
    Module<B>: VecZnxAddScalarInplace
        + VecZnxDftBytesOf
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxFillUniform
        + VecZnxSubInplace
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
            use poulpy_core::layouts::{GLWEInfos, LWEInfos};

            assert_eq!(self.keys.len() as u32, sk_lwe.n());
            assert!(sk_glwe.n() <= module.n() as u32);
            assert_eq!(sk_glwe.rank(), self.rank());
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

        let mut pt: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(sk_glwe.n().into(), 1);
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
    fn alloc<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: BlindRotationKeyInfos,
    {
        let mut data: Vec<GGSWPrepared<Vec<u8>, B>> = Vec::with_capacity(infos.n_lwe().into());
        (0..infos.n_lwe().as_usize()).for_each(|_| data.push(GGSWPrepared::alloc_from_infos(module, infos)));
        Self {
            data,
            dist: Distribution::NONE,
            x_pow_a: None,
            _phantom: PhantomData,
        }
    }
}

impl BlindRotationKeyCompressed<Vec<u8>, CGGI> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: BlindRotationKeyInfos,
    {
        let mut data: Vec<GGSWCompressed<Vec<u8>>> = Vec::with_capacity(infos.n_lwe().into());
        (0..infos.n_lwe().as_usize()).for_each(|_| data.push(GGSWCompressed::alloc_from_infos(infos)));
        Self {
            keys: data,
            dist: Distribution::NONE,
            _phantom: PhantomData,
        }
    }

    pub fn generate_from_sk_scratch_space<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGSWInfos,
        Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftBytesOf,
    {
        GGSWCompressed::encrypt_sk_scratch_space(module, infos)
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
            + VecZnxDftBytesOf
            + VecZnxBigNormalize<B>
            + VecZnxDftApply<B>
            + SvpApplyDftToDftInplace<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxFillUniform
            + VecZnxSubInplace
            + VecZnxAddInplace
            + VecZnxNormalizeInplace<B>
            + VecZnxAddNormal
            + VecZnxNormalize<B>
            + VecZnxSub,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            use poulpy_core::layouts::{GLWEInfos, LWEInfos};

            assert_eq!(self.n_lwe(), sk_lwe.n());
            assert!(sk_glwe.n() <= module.n() as u32);
            assert_eq!(sk_glwe.rank(), self.rank());
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

        let mut pt: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(sk_glwe.n().into(), 1);
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
