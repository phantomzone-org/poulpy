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
    layouts::{GGLWECiphertext, GGLWEInfos, GLWECiphertext, GLWEPlaintext, LWEInfos, prepared::GLWESecretPrepared},
};

impl GGLWECiphertext<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes + VecZnxNormalizeTmpBytes,
    {
        GLWECiphertext::encrypt_sk_scratch_space(module, &infos.glwe_layout())
            + (GLWEPlaintext::alloc_bytes(&infos.glwe_layout()) | module.vec_znx_normalize_tmp_bytes())
    }

    pub fn encrypt_pk_scratch_space<B: Backend, A>(_module: &Module<B>, _infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        unimplemented!()
    }
}

impl<DataSelf: DataMut> GGLWECiphertext<DataSelf> {
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
                scratch.available() >= GGLWECiphertext::encrypt_sk_scratch_space(module, self),
                "scratch.available: {} < GGLWECiphertext::encrypt_sk_scratch_space(module, self.rank()={}, self.size()={}): {}",
                scratch.available(),
                self.rank_out(),
                self.size(),
                GGLWECiphertext::encrypt_sk_scratch_space(module, self)
            );
            assert!(
                self.rows().0 * self.digits().0 * self.base2k().0 <= self.k().0,
                "self.rows() : {} * self.digits() : {} * self.base2k() : {} = {} >= self.k() = {}",
                self.rows(),
                self.digits(),
                self.base2k(),
                self.rows().0 * self.digits().0 * self.base2k().0,
                self.k()
            );
        }

        let rows: usize = self.rows().into();
        let digits: usize = self.digits().into();
        let base2k: usize = self.base2k().into();
        let rank_in: usize = self.rank_in().into();

        let (mut tmp_pt, scrach_1) = scratch.take_glwe_pt(self);
        // For each input column (i.e. rank) produces a GGLWE ciphertext of rank_out+1 columns
        //
        // Example for ksk rank 2 to rank 3:
        //
        // (-(a0*s0 + a1*s1 + a2*s2) + s0', a0, a1, a2)
        // (-(b0*s0 + b1*s1 + b2*s2) + s0', b0, b1, b2)
        //
        // Example ksk rank 2 to rank 1
        //
        // (-(a*s) + s0, a)
        // (-(b*s) + s1, b)
        (0..rank_in).for_each(|col_i| {
            (0..rows).for_each(|row_i| {
                // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
                tmp_pt.data.zero(); // zeroes for next iteration
                module.vec_znx_add_scalar_inplace(
                    &mut tmp_pt.data,
                    0,
                    (digits - 1) + row_i * digits,
                    pt,
                    col_i,
                );
                module.vec_znx_normalize_inplace(base2k, &mut tmp_pt.data, 0, scrach_1);

                // rlwe encrypt of vec_znx_pt into vec_znx_ct
                self.at_mut(row_i, col_i)
                    .encrypt_sk(module, &tmp_pt, sk, source_xa, source_xe, scrach_1);
            });
        });
    }
}
