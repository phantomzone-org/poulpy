use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyInplace, TakeVecZnx, TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal, VecZnxBigNormalize,
        VecZnxDftAllocBytes, VecZnxDftApply, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace,
        VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubABInplace,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
    source::Source,
};

use crate::{
    encryption::{SIGMA, glwe_ct::glwe_encrypt_sk_internal},
    layouts::{GLWECiphertext, GLWEPlaintext, Infos, compressed::GLWECiphertextCompressed, prepared::GLWESecretPrepared},
};

impl GLWECiphertextCompressed<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize) -> usize
    where
        Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes,
    {
        GLWECiphertext::encrypt_sk_scratch_space(module, basek, k)
    }
}

impl<D: DataMut> GLWECiphertextCompressed<D> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<DataPt: DataRef, DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &GLWEPlaintext<DataPt>,
        sk: &GLWESecretPrepared<DataSk, B>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
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
        self.encrypt_sk_internal(module, Some((pt, 0)), sk, seed_xa, source_xe, scratch);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encrypt_sk_internal<DataPt: DataRef, DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: Option<(&GLWEPlaintext<DataPt>, usize)>,
        sk: &GLWESecretPrepared<DataSk, B>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
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
        let mut source_xa = Source::new(seed_xa);
        let cols: usize = self.rank() + 1;
        glwe_encrypt_sk_internal(
            module,
            self.basek(),
            self.k(),
            &mut self.data,
            cols,
            true,
            pt,
            sk,
            &mut source_xa,
            source_xe,
            SIGMA,
            scratch,
        );
        self.seed = seed_xa;
    }
}
