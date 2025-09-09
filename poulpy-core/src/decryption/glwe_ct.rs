use poulpy_hal::{
    api::{
        SvpApplyInplace, TakeVecZnxBig, TakeVecZnxDft, VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigNormalize,
        VecZnxDftAllocBytes, VecZnxDftApply, VecZnxIdftApplyConsume, VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, DataViewMut, Module, Scratch},
};

use crate::layouts::{GLWECiphertext, GLWEPlaintext, Infos, prepared::GLWESecretPrepared};

impl GLWECiphertext<Vec<u8>> {
    pub fn decrypt_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes,
    {
        let size: usize = k.div_ceil(basek);
        (module.vec_znx_normalize_tmp_bytes() | module.vec_znx_dft_alloc_bytes(1, size)) + module.vec_znx_dft_alloc_bytes(1, size)
    }
}

impl<DataSelf: DataRef> GLWECiphertext<DataSelf> {
    pub fn decrypt<DataPt: DataMut, DataSk: DataRef, B: Backend>(
        &self,
        module: &Module<B>,
        pt: &mut GLWEPlaintext<DataPt>,
        sk: &GLWESecretPrepared<DataSk, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftApply<B>
            + SvpApplyInplace<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddInplace<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + TakeVecZnxBig<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk.rank());
            assert_eq!(self.n(), sk.n());
            assert_eq!(pt.n(), sk.n());
        }

        let cols: usize = self.rank() + 1;

        let (mut c0_big, scratch_1) = scratch.take_vec_znx_big(self.n(), 1, self.size()); // TODO optimize size when pt << ct
        c0_big.data_mut().fill(0);

        {
            (1..cols).for_each(|i| {
                // ci_dft = DFT(a[i]) * DFT(s[i])
                let (mut ci_dft, _) = scratch_1.take_vec_znx_dft(self.n(), 1, self.size()); // TODO optimize size when pt << ct
                module.vec_znx_dft_apply(1, 0, &mut ci_dft, 0, &self.data, i);
                module.svp_apply_inplace(&mut ci_dft, 0, &sk.data, i - 1);
                let ci_big = module.vec_znx_idft_apply_consume(ci_dft);

                // c0_big += a[i] * s[i]
                module.vec_znx_big_add_inplace(&mut c0_big, 0, &ci_big, 0);
            });
        }

        // c0_big = (a * s) + (-a * s + m + e) = BIG(m + e)
        module.vec_znx_big_add_small_inplace(&mut c0_big, 0, &self.data, 0);

        // pt = norm(BIG(m + e))
        module.vec_znx_big_normalize(self.basek(), &mut pt.data, 0, &c0_big, 0, scratch_1);

        pt.basek = self.basek();
        pt.k = pt.k().min(self.k());
    }
}
