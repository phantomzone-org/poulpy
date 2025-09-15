use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnxDft, VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxIdftApplyConsume,
        VecZnxNormalizeTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, DataViewMut, Module, Scratch, VecZnxBig},
};

use crate::layouts::{GLWECiphertext, Infos, prepared::GGSWCiphertextPrepared};

impl GLWECiphertext<Vec<u8>> {
    #[allow(clippy::too_many_arguments)]
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
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxNormalizeTmpBytes,
    {
        let in_size: usize = k_in.div_ceil(basek).div_ceil(digits);
        let out_size: usize = k_out.div_ceil(basek);
        let ggsw_size: usize = k_ggsw.div_ceil(basek);
        let res_dft: usize = module.vec_znx_dft_alloc_bytes(rank + 1, ggsw_size);
        let a_dft: usize = module.vec_znx_dft_alloc_bytes(rank + 1, in_size);
        let vmp: usize = module.vmp_apply_dft_to_dft_tmp_bytes(
            out_size,
            in_size,
            in_size,  // rows
            rank + 1, // cols in
            rank + 1, // cols out
            ggsw_size,
        );
        let normalize: usize = module.vec_znx_normalize_tmp_bytes();
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
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxNormalizeTmpBytes,
    {
        Self::external_product_scratch_space(module, basek, k_out, k_out, k_ggsw, digits, rank)
    }
}

impl<DataSelf: DataMut> GLWECiphertext<DataSelf> {
    pub fn external_product<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &GGSWCiphertextPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftApply<B>
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        let basek: usize = self.basek();

        #[cfg(debug_assertions)]
        {
            use poulpy_hal::api::ScratchAvailable;

            assert_eq!(rhs.rank(), lhs.rank());
            assert_eq!(rhs.rank(), self.rank());
            assert_eq!(self.basek(), basek);
            assert_eq!(lhs.basek(), basek);
            assert_eq!(rhs.n(), self.n());
            assert_eq!(lhs.n(), self.n());
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

        let (mut res_dft, scratch_1) = scratch.take_vec_znx_dft(self.n(), cols, rhs.size()); // Todo optimise
        let (mut a_dft, scratch_2) = scratch_1.take_vec_znx_dft(self.n(), cols, lhs.size().div_ceil(digits));

        a_dft.data_mut().fill(0);

        {
            (0..digits).for_each(|di| {
                // (lhs.size() + di) / digits = (a - (digit - di - 1)).div_ceil(digits)
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
                    module.vec_znx_dft_apply(digits, digits - 1 - di, &mut a_dft, col_i, &lhs.data, col_i);
                });

                if di == 0 {
                    module.vmp_apply_dft_to_dft(&mut res_dft, &a_dft, &rhs.data, scratch_2);
                } else {
                    module.vmp_apply_dft_to_dft_add(&mut res_dft, &a_dft, &rhs.data, di, scratch_2);
                }
            });
        }

        let res_big: VecZnxBig<&mut [u8], B> = module.vec_znx_idft_apply_consume(res_dft);

        (0..cols).for_each(|i| {
            module.vec_znx_big_normalize(basek, &mut self.data, i, &res_big, i, scratch_1);
        });
    }

    pub fn external_product_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGSWCiphertextPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftApply<B>
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        let basek: usize = self.basek();

        #[cfg(debug_assertions)]
        {
            use poulpy_hal::api::ScratchAvailable;

            assert_eq!(rhs.rank(), self.rank());
            assert_eq!(self.basek(), basek);
            assert_eq!(rhs.n(), self.n());
            assert!(
                scratch.available()
                    >= GLWECiphertext::external_product_scratch_space(
                        module,
                        self.basek(),
                        self.k(),
                        self.k(),
                        rhs.k(),
                        rhs.digits(),
                        rhs.rank(),
                    )
            );
        }

        let cols: usize = rhs.rank() + 1;
        let digits: usize = rhs.digits();

        let (mut res_dft, scratch_1) = scratch.take_vec_znx_dft(self.n(), cols, rhs.size()); // Todo optimise
        let (mut a_dft, scratch_2) = scratch_1.take_vec_znx_dft(self.n(), cols, self.size().div_ceil(digits));

        a_dft.data_mut().fill(0);

        {
            (0..digits).for_each(|di| {
                // (lhs.size() + di) / digits = (a - (digit - di - 1)).div_ceil(digits)
                a_dft.set_size((self.size() + di) / digits);

                // Small optimization for digits > 2
                // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
                // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(digits-1) * B}.
                // As such we can ignore the last digits-2 limbs safely of the sum of vmp products.
                // It is possible to further ignore the last digits-1 limbs, but this introduce
                // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                // noise is kept with respect to the ideal functionality.
                res_dft.set_size(rhs.size() - ((digits - di) as isize - 2).max(0) as usize);

                (0..cols).for_each(|col_i| {
                    module.vec_znx_dft_apply(
                        digits,
                        digits - 1 - di,
                        &mut a_dft,
                        col_i,
                        &self.data,
                        col_i,
                    );
                });

                if di == 0 {
                    module.vmp_apply_dft_to_dft(&mut res_dft, &a_dft, &rhs.data, scratch_2);
                } else {
                    module.vmp_apply_dft_to_dft_add(&mut res_dft, &a_dft, &rhs.data, di, scratch_2);
                }
            });
        }

        let res_big: VecZnxBig<&mut [u8], B> = module.vec_znx_idft_apply_consume(res_dft);

        (0..cols).for_each(|i| {
            module.vec_znx_big_normalize(basek, &mut self.data, i, &res_big, i, scratch_1);
        });
    }
}
