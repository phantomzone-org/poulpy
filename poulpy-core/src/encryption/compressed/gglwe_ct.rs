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
    layouts::{GGLWECiphertext, GGLWEInfos, LWEInfos, compressed::GGLWECiphertextCompressed, prepared::GLWESecretPrepared},
};

impl GGLWECiphertextCompressed<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes + VecZnxNormalizeTmpBytes,
    {
        GGLWECiphertext::encrypt_sk_scratch_space(module, infos)
    }
}

impl<D: DataMut> GGLWECiphertextCompressed<D> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<DataPt: DataRef, DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &ScalarZnx<DataPt>,
        sk: &GLWESecretPrepared<DataSk, B>,
        seed: [u8; 32],
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

            assert_eq!(
                self.rank_in(),
                pt.cols() as u32,
                "self.rank_in(): {} != pt.cols(): {}",
                self.rank_in(),
                pt.cols()
            );
            assert_eq!(
                self.rank_out(),
                sk.rank(),
                "self.rank_out(): {} != sk.rank(): {}",
                self.rank_out(),
                sk.rank()
            );
            assert_eq!(self.n(), sk.n());
            assert_eq!(pt.n() as u32, sk.n());
            assert!(
                scratch.available() >= GGLWECiphertextCompressed::encrypt_sk_scratch_space(module, self),
                "scratch.available: {} < GGLWECiphertext::encrypt_sk_scratch_space: {}",
                scratch.available(),
                GGLWECiphertextCompressed::encrypt_sk_scratch_space(module, self)
            );
            assert!(
                self.dnum().0 * self.dsize().0 * self.base2k().0 <= self.k().0,
                "self.dnum() : {} * self.dsize() : {} * self.base2k() : {} = {} >= self.k() = {}",
                self.dnum(),
                self.dsize(),
                self.base2k(),
                self.dnum().0 * self.dsize().0 * self.base2k().0,
                self.k()
            );
        }

        let dnum: usize = self.dnum().into();
        let dsize: usize = self.dsize().into();
        let base2k: usize = self.base2k().into();
        let rank_in: usize = self.rank_in().into();
        let cols: usize = (self.rank_out() + 1).into();

        let mut source_xa = Source::new(seed);

        let (mut tmp_pt, scrach_1) = scratch.take_glwe_pt(self);
        (0..rank_in).for_each(|col_i| {
            (0..dnum).for_each(|d_i| {
                // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
                tmp_pt.data.zero(); // zeroes for next iteration
                module.vec_znx_add_scalar_inplace(&mut tmp_pt.data, 0, (dsize - 1) + d_i * dsize, pt, col_i);
                module.vec_znx_normalize_inplace(base2k, &mut tmp_pt.data, 0, scrach_1);

                let (seed, mut source_xa_tmp) = source_xa.branch();
                self.seed[col_i * dnum + d_i] = seed;

                glwe_encrypt_sk_internal(
                    module,
                    self.base2k().into(),
                    self.k().into(),
                    &mut self.at_mut(d_i, col_i).data,
                    cols,
                    true,
                    Some((&tmp_pt, 0)),
                    sk,
                    &mut source_xa_tmp,
                    source_xe,
                    SIGMA,
                    scrach_1,
                );
            });
        });
    }
}
