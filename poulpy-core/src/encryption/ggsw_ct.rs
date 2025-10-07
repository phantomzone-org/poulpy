use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyDftToDftInplace, TakeVecZnx, TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal,
        VecZnxAddScalarInplace, VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxFillUniform,
        VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace,
    },
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnx, Scratch, VecZnx, ZnxZero},
    source::Source,
};

use crate::{
    TakeGLWEPt,
    layouts::{GGSWCiphertext, GGSWInfos, GLWECiphertext, GLWEInfos, LWEInfos, prepared::GLWESecretPrepared},
};

impl GGSWCiphertext<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGSWInfos,
        Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes,
    {
        let size = infos.size();
        GLWECiphertext::encrypt_sk_scratch_space(module, &infos.glwe_layout())
            + VecZnx::alloc_bytes(module.n(), (infos.rank() + 1).into(), size)
            + VecZnx::alloc_bytes(module.n(), 1, size)
            + module.vec_znx_dft_alloc_bytes((infos.rank() + 1).into(), size)
    }
}

impl<DataSelf: DataMut> GGSWCiphertext<DataSelf> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<DataPt: DataRef, DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &ScalarZnx<DataPt>,
        sk: &GLWESecretPrepared<DataSk, B>,
        source_xa: &mut Source,
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
        let dsize: usize = self.dsize().into();

        let (mut tmp_pt, scratch_1) = scratch.take_glwe_pt(&self.glwe_layout());

        (0..self.dnum().into()).for_each(|row_i| {
            tmp_pt.data.zero();

            // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
            module.vec_znx_add_scalar_inplace(&mut tmp_pt.data, 0, (dsize - 1) + row_i * dsize, pt, 0);
            module.vec_znx_normalize_inplace(base2k, &mut tmp_pt.data, 0, scratch_1);

            (0..rank + 1).for_each(|col_j| {
                // rlwe encrypt of vec_znx_pt into vec_znx_ct

                self.at_mut(row_i, col_j).encrypt_sk_internal(
                    module,
                    Some((&tmp_pt, col_j)),
                    sk,
                    source_xa,
                    source_xe,
                    scratch_1,
                );
            });
        });
    }
}
