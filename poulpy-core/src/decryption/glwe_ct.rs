use poulpy_hal::{
    api::{
        SvpApplyDftToDftInplace, TakeVecZnxBig, TakeVecZnxDft, VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigNormalize,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, DataViewMut, Module, Scratch},
};

use crate::layouts::{GLWE, GLWEInfos, GLWEPlaintext, LWEInfos, prepared::GLWESecretPrepared};

impl GLWE<Vec<u8>> {
    pub fn decrypt_scratch_space<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GLWEInfos,
        Module<B>: VecZnxDftBytesOf + VecZnxNormalizeTmpBytes + VecZnxDftBytesOf,
    {
        let size: usize = infos.size();
        (module.vec_znx_normalize_tmp_bytes() | module.bytes_of_vec_znx_dft(1, size)) + module.bytes_of_vec_znx_dft(1, size)
    }
}

impl<DataSelf: DataRef> GLWE<DataSelf> {
    pub fn decrypt<DataPt: DataMut, DataSk: DataRef, B: Backend>(
        &self,
        module: &Module<B>,
        pt: &mut GLWEPlaintext<DataPt>,
        sk: &GLWESecretPrepared<DataSk, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftApply<B>
            + SvpApplyDftToDftInplace<B>
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

        let cols: usize = (self.rank() + 1).into();

        let (mut c0_big, scratch_1) = scratch.take_vec_znx_big(self.n().into(), 1, self.size()); // TODO optimize size when pt << ct
        c0_big.data_mut().fill(0);

        {
            (1..cols).for_each(|i| {
                // ci_dft = DFT(a[i]) * DFT(s[i])
                let (mut ci_dft, _) = scratch_1.take_vec_znx_dft(self.n().into(), 1, self.size()); // TODO optimize size when pt << ct
                module.vec_znx_dft_apply(1, 0, &mut ci_dft, 0, &self.data, i);
                module.svp_apply_dft_to_dft_inplace(&mut ci_dft, 0, &sk.data, i - 1);
                let ci_big = module.vec_znx_idft_apply_consume(ci_dft);

                // c0_big += a[i] * s[i]
                module.vec_znx_big_add_inplace(&mut c0_big, 0, &ci_big, 0);
            });
        }

        // c0_big = (a * s) + (-a * s + m + e) = BIG(m + e)
        module.vec_znx_big_add_small_inplace(&mut c0_big, 0, &self.data, 0);

        // pt = norm(BIG(m + e))
        module.vec_znx_big_normalize(
            self.base2k().into(),
            &mut pt.data,
            0,
            self.base2k().into(),
            &c0_big,
            0,
            scratch_1,
        );

        pt.base2k = self.base2k();
        pt.k = pt.k().min(self.k());
    }
}
