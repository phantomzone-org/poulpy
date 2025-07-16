use backend::{
    Backend, DataViewMut, Module, Scratch, SvpApplyInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAllocBytes,
    VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftFromVecZnx, VecZnxDftToVecZnxBigConsume, VecZnxScratch,
};

use crate::{GLWECiphertext, GLWEPlaintext, GLWESecretExec, Infos};

pub trait GLWEDecryptFamily<B: Backend> = VecZnxDftAllocBytes
    + VecZnxBigAllocBytes
    + VecZnxDftFromVecZnx<B>
    + SvpApplyInplace<B>
    + VecZnxDftToVecZnxBigConsume<B>
    + VecZnxBigAddInplace<B>
    + VecZnxBigAddSmallInplace<B>
    + VecZnxBigNormalize<B>;

impl GLWECiphertext<Vec<u8>> {
    pub fn decrypt_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize) -> usize
    where
        Module<B>: GLWEDecryptFamily<B>,
    {
        let size: usize = k.div_ceil(basek);
        (module.vec_znx_normalize_tmp_bytes() | module.vec_znx_dft_alloc_bytes(1, size)) + module.vec_znx_dft_alloc_bytes(1, size)
    }
}

impl<DataSelf: AsRef<[u8]>> GLWECiphertext<DataSelf> {
    pub fn decrypt<DataPt: AsMut<[u8]> + AsRef<[u8]>, DataSk: AsRef<[u8]>, B: Backend>(
        &self,
        module: &Module<B>,
        pt: &mut GLWEPlaintext<DataPt>,
        sk: &GLWESecretExec<DataSk, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>: GLWEDecryptFamily<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk.rank());
            assert_eq!(self.n(), module.n());
            assert_eq!(pt.n(), module.n());
            assert_eq!(sk.n(), module.n());
        }

        let cols: usize = self.rank() + 1;

        let (mut c0_big, scratch_1) = scratch.tmp_vec_znx_big(module, 1, self.size()); // TODO optimize size when pt << ct
        c0_big.data_mut().fill(0);

        {
            (1..cols).for_each(|i| {
                // ci_dft = DFT(a[i]) * DFT(s[i])
                let (mut ci_dft, _) = scratch_1.tmp_vec_znx_dft(module, 1, self.size()); // TODO optimize size when pt << ct
                module.vec_znx_dft_from_vec_znx(1, 0, &mut ci_dft, 0, &self.data, i);
                module.svp_apply_inplace(&mut ci_dft, 0, &sk.data, i - 1);
                let ci_big = module.vec_znx_dft_to_vec_znx_big_consume(ci_dft);

                // c0_big += a[i] * s[i]
                module.vec_znx_big_add_inplace(&mut c0_big, 0, &ci_big, 0);
            });
        }

        // c0_big = (a * s) + (-a * s + m + e) = BIG(m + e)
        module.vec_znx_big_add_small_inplace(&mut c0_big, 0, &self.data, 0);

        // pt = norm(BIG(m + e))
        module.vec_znx_big_normalize(self.basek(), &mut pt.data, 0, &mut c0_big, 0, scratch_1);

        pt.basek = self.basek();
        pt.k = pt.k().min(self.k());
    }
}
