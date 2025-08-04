use backend::hal::{
    api::{
        DataViewMut, ScratchAvailable, TakeVecZnxDft, VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftFromVecZnx,
        VecZnxDftToVecZnxBigConsume, VecZnxNormalizeTmpBytes, VmpApply, VmpApplyAdd, VmpApplyTmpBytes,
    },
    layouts::{Backend, Module, Scratch, VecZnxBig},
};

use crate::{GGSWCiphertextExec, GLWECiphertext, Infos};

pub trait GLWEExternalProductFamily<B: Backend> = VecZnxDftAllocBytes
    + VmpApplyTmpBytes
    + VmpApply<B>
    + VmpApplyAdd<B>
    + VecZnxDftFromVecZnx<B>
    + VecZnxDftToVecZnxBigConsume<B>
    + VecZnxBigNormalize<B>
    + VecZnxNormalizeTmpBytes;

impl GLWECiphertext<Vec<u8>> {
    pub fn external_product_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_in: usize,
        k_ggsw: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GLWEExternalProductFamily<B>,
    {
        let in_size: usize = k_in.div_ceil(basek).div_ceil(digits);
        let out_size: usize = k_out.div_ceil(basek);
        let ggsw_size: usize = k_ggsw.div_ceil(basek);
        let res_dft: usize = module.vec_znx_dft_alloc_bytes(rank + 1, ggsw_size);
        let a_dft: usize = module.vec_znx_dft_alloc_bytes(rank + 1, in_size);
        let vmp: usize = module.vmp_apply_tmp_bytes(
            out_size,
            in_size,
            in_size,  // rows
            rank + 1, // cols in
            rank + 1, // cols out
            ggsw_size,
        );
        let normalize: usize = module.vec_znx_normalize_tmp_bytes(module.n());
        res_dft + a_dft + (vmp | normalize)
    }

    pub fn external_product_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_ggsw: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GLWEExternalProductFamily<B>,
    {
        Self::external_product_scratch_space(module, basek, k_out, k_out, k_ggsw, digits, rank)
    }
}

impl<DataSelf: AsRef<[u8]> + AsMut<[u8]>> GLWECiphertext<DataSelf> {
    pub fn external_product<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &GGSWCiphertextExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEExternalProductFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        let basek: usize = self.basek();

        #[cfg(debug_assertions)]
        {
            use backend::hal::api::ScratchAvailable;

            assert_eq!(rhs.rank(), lhs.rank());
            assert_eq!(rhs.rank(), self.rank());
            assert_eq!(self.basek(), basek);
            assert_eq!(lhs.basek(), basek);
            assert_eq!(rhs.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(lhs.n(), module.n());
            assert!(
                scratch.available()
                    >= GLWECiphertext::external_product_scratch_space(
                        module,
                        self.basek(),
                        self.k(),
                        lhs.k(),
                        rhs.k(),
                        rhs.digits(),
                        rhs.rank(),
                    )
            );
        }

        let cols: usize = rhs.rank() + 1;
        let digits: usize = rhs.digits();

        let (mut res_dft, scratch1) = scratch.take_vec_znx_dft(module, cols, rhs.size()); // Todo optimise
        let (mut a_dft, scratch2) = scratch1.take_vec_znx_dft(module, cols, (lhs.size() + digits - 1) / digits);

        a_dft.data_mut().fill(0);

        {
            (0..digits).for_each(|di| {
                // (lhs.size() + di) / digits = (a - (digit - di - 1) + digit - 1) / digits
                a_dft.set_size((lhs.size() + di) / digits);

                // Small optimization for digits > 2
                // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
                // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(digits-1) * B}.
                // As such we can ignore the last digits-2 limbs safely of the sum of vmp products.
                // It is possible to further ignore the last digits-1 limbs, but this introduce
                // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                // noise is kept with respect to the ideal functionality.
                res_dft.set_size(rhs.size() - ((digits - di) as isize - 2).max(0) as usize);

                (0..cols).for_each(|col_i| {
                    module.vec_znx_dft_from_vec_znx(digits, digits - 1 - di, &mut a_dft, col_i, &lhs.data, col_i);
                });

                if di == 0 {
                    module.vmp_apply(&mut res_dft, &a_dft, &rhs.data, scratch2);
                } else {
                    module.vmp_apply_add(&mut res_dft, &a_dft, &rhs.data, di, scratch2);
                }
            });
        }

        let res_big: VecZnxBig<&mut [u8], B> = module.vec_znx_dft_to_vec_znx_big_consume(res_dft);

        (0..cols).for_each(|i| {
            module.vec_znx_big_normalize(basek, &mut self.data, i, &res_big, i, scratch1);
        });
    }

    pub fn external_product_inplace<DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGSWCiphertextExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEExternalProductFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        unsafe {
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            self.external_product(&module, &*self_ptr, rhs, scratch);
        }
    }
}
