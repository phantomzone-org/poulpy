use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyDftToDftInplace, TakeVecZnx, TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal,
        VecZnxAddScalarInplace, VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxFillUniform,
        VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace,
    },
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnx, Scratch, ZnxZero},
    source::Source,
};

use crate::{
    TakeGLWEPt,
    encryption::{SIGMA, glwe_encrypt_sk_internal},
    layouts::{
        GGSWCiphertext, GGSWInfos, GLWEInfos, LWEInfos, compressed::GGSWCiphertextCompressed, prepared::GLWESecretPrepared,
    },
};

impl GGSWCiphertextCompressed<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGSWInfos,
        Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes,
    {
        GGSWCiphertext::encrypt_sk_scratch_space(module, infos)
    }
}

impl<DataSelf: DataMut> GGSWCiphertextCompressed<DataSelf> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<DataPt: DataRef, DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &ScalarZnx<DataPt>,
        sk: &GLWESecretPrepared<DataSk, B>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxAddScalarInplace
            + VecZnxDftAllocBytes
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
            use poulpy_hal::layouts::ZnxInfos;

            assert_eq!(self.rank(), sk.rank());
            assert_eq!(self.n(), sk.n());
            assert_eq!(pt.n() as u32, sk.n());
        }

        let base2k: usize = self.base2k().into();
        let rank: usize = self.rank().into();
        let cols: usize = rank + 1;
        let dsize: usize = self.dsize().into();

        let (mut tmp_pt, scratch_1) = scratch.take_glwe_pt(&self.glwe_layout());

        let mut source = Source::new(seed_xa);

        self.seed = vec![[0u8; 32]; self.dnum().0 as usize * cols];

        (0..self.dnum().into()).for_each(|row_i| {
            tmp_pt.data.zero();

            // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
            module.vec_znx_add_scalar_inplace(&mut tmp_pt.data, 0, (dsize - 1) + row_i * dsize, pt, 0);
            module.vec_znx_normalize_inplace(base2k, &mut tmp_pt.data, 0, scratch_1);

            (0..rank + 1).for_each(|col_j| {
                // rlwe encrypt of vec_znx_pt into vec_znx_ct

                let (seed, mut source_xa_tmp) = source.branch();

                self.seed[row_i * cols + col_j] = seed;

                glwe_encrypt_sk_internal(
                    module,
                    self.base2k().into(),
                    self.k().into(),
                    &mut self.at_mut(row_i, col_j).data,
                    cols,
                    true,
                    Some((&tmp_pt, col_j)),
                    sk,
                    &mut source_xa_tmp,
                    source_xe,
                    SIGMA,
                    scratch_1,
                );
            });
        });
    }
}
