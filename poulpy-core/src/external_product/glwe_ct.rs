use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftApply,
        VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAdd,
        VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, DataViewMut, Module, Scratch, VecZnx, VecZnxBig},
};

use crate::layouts::{GLWECiphertext, Infos, prepared::GGSWCiphertextPrepared};

impl GLWECiphertext<Vec<u8>> {
    #[allow(clippy::too_many_arguments)]
    pub fn external_product_scratch_space<B: Backend>(
        module: &Module<B>,
        basek_out: usize,
        k_out: usize,
        basek_in: usize,
        k_in: usize,
        basek_ggsw: usize,
        k_ggsw: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxNormalizeTmpBytes,
    {
        let in_size: usize = k_in.div_ceil(basek_ggsw).div_ceil(digits);
        let out_size: usize = k_out.div_ceil(basek_out);
        let ggsw_size: usize = k_ggsw.div_ceil(basek_ggsw);
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
        let normalize_big: usize = module.vec_znx_normalize_tmp_bytes();

        if basek_in == basek_ggsw {
            res_dft + a_dft + (vmp | normalize_big)
        } else {
            let normalize_conv: usize = VecZnx::alloc_bytes(module.n(), rank + 1, in_size);
            res_dft + ((a_dft + normalize_conv + (module.vec_znx_normalize_tmp_bytes() | vmp)) | normalize_big)
        }
    }

    pub fn external_product_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        basek_out: usize,
        k_out: usize,
        basek_ggsw: usize,
        k_ggsw: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxNormalizeTmpBytes,
    {
        Self::external_product_scratch_space(
            module, basek_out, k_out, basek_out, k_out, k_ggsw, basek_ggsw, digits, rank,
        )
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
            + VecZnxBigNormalize<B>
            + VecZnxNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        let basek_in: usize = lhs.basek();
        let basek_ggsw: usize = rhs.basek();
        let basek_out: usize = self.basek();

        #[cfg(debug_assertions)]
        {
            use poulpy_hal::api::ScratchAvailable;

            assert_eq!(rhs.rank(), lhs.rank());
            assert_eq!(rhs.rank(), self.rank());
            assert_eq!(rhs.n(), self.n());
            assert_eq!(lhs.n(), self.n());
            assert!(
                scratch.available()
                    >= GLWECiphertext::external_product_scratch_space(
                        module,
                        basek_out,
                        self.k(),
                        basek_in,
                        lhs.k(),
                        basek_ggsw,
                        rhs.k(),
                        rhs.digits(),
                        rhs.rank(),
                    )
            );
        }

        let cols: usize = rhs.rank() + 1;
        let digits: usize = rhs.digits();

        let a_size: usize = (lhs.size() * basek_in).div_ceil(basek_ggsw);

        let (mut res_dft, scratch_1) = scratch.take_vec_znx_dft(self.n(), cols, rhs.size()); // Todo optimise
        let (mut a_dft, scratch_2) = scratch_1.take_vec_znx_dft(self.n(), cols, a_size.div_ceil(digits));
        a_dft.data_mut().fill(0);

        if basek_in == basek_ggsw {
            for di in 0..digits {
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

                for j in 0..cols {
                    module.vec_znx_dft_apply(digits, digits - 1 - di, &mut a_dft, j, &lhs.data, j);
                }

                if di == 0 {
                    module.vmp_apply_dft_to_dft(&mut res_dft, &a_dft, &rhs.data, scratch_2);
                } else {
                    module.vmp_apply_dft_to_dft_add(&mut res_dft, &a_dft, &rhs.data, di, scratch_2);
                }
            }
        } else {
            let (mut a_conv, scratch_3) = scratch_2.take_vec_znx(module.n(), cols, a_size);

            for j in 0..cols {
                module.vec_znx_normalize(
                    basek_ggsw,
                    &mut a_conv,
                    j,
                    basek_in,
                    &lhs.data,
                    j,
                    scratch_3,
                );
            }

            for di in 0..digits {
                // (lhs.size() + di) / digits = (a - (digit - di - 1)).div_ceil(digits)
                a_dft.set_size((a_size + di) / digits);

                // Small optimization for digits > 2
                // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
                // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(digits-1) * B}.
                // As such we can ignore the last digits-2 limbs safely of the sum of vmp products.
                // It is possible to further ignore the last digits-1 limbs, but this introduce
                // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                // noise is kept with respect to the ideal functionality.
                res_dft.set_size(rhs.size() - ((digits - di) as isize - 2).max(0) as usize);

                for j in 0..cols {
                    module.vec_znx_dft_apply(digits, digits - 1 - di, &mut a_dft, j, &a_conv, j);
                }

                if di == 0 {
                    module.vmp_apply_dft_to_dft(&mut res_dft, &a_dft, &rhs.data, scratch_3);
                } else {
                    module.vmp_apply_dft_to_dft_add(&mut res_dft, &a_dft, &rhs.data, di, scratch_3);
                }
            }
        }

        let res_big: VecZnxBig<&mut [u8], B> = module.vec_znx_idft_apply_consume(res_dft);

        (0..cols).for_each(|i| {
            module.vec_znx_big_normalize(
                basek_out,
                &mut self.data,
                i,
                basek_ggsw,
                &res_big,
                i,
                scratch_1,
            );
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
            + VecZnxBigNormalize<B>
            + VecZnxNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        let basek_in: usize = self.basek();
        let basek_ggsw: usize = rhs.basek();

        #[cfg(debug_assertions)]
        {
            use poulpy_hal::api::ScratchAvailable;

            assert_eq!(rhs.rank(), self.rank());
            assert_eq!(rhs.n(), self.n());
            assert!(
                scratch.available()
                    >= GLWECiphertext::external_product_inplace_scratch_space(
                        module,
                        basek_in,
                        self.k(),
                        basek_ggsw,
                        rhs.k(),
                        rhs.digits(),
                        rhs.rank(),
                    )
            );
        }

        let cols: usize = rhs.rank() + 1;
        let digits: usize = rhs.digits();
        let a_size: usize = (self.size() * basek_in).div_ceil(basek_ggsw);

        let (mut res_dft, scratch_1) = scratch.take_vec_znx_dft(self.n(), cols, rhs.size()); // Todo optimise
        let (mut a_dft, scratch_2) = scratch_1.take_vec_znx_dft(self.n(), cols, a_size.div_ceil(digits));
        a_dft.data_mut().fill(0);

        if basek_in == basek_ggsw {
            for di in 0..digits {
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

                for j in 0..cols {
                    module.vec_znx_dft_apply(digits, digits - 1 - di, &mut a_dft, j, &self.data, j);
                }

                if di == 0 {
                    module.vmp_apply_dft_to_dft(&mut res_dft, &a_dft, &rhs.data, scratch_2);
                } else {
                    module.vmp_apply_dft_to_dft_add(&mut res_dft, &a_dft, &rhs.data, di, scratch_2);
                }
            }
        } else {
            let (mut a_conv, scratch_3) = scratch_2.take_vec_znx(module.n(), cols, a_size);

            for j in 0..cols {
                module.vec_znx_normalize(
                    basek_ggsw,
                    &mut a_conv,
                    j,
                    basek_in,
                    &self.data,
                    j,
                    scratch_3,
                );
            }

            for di in 0..digits {
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

                for j in 0..cols {
                    module.vec_znx_dft_apply(digits, digits - 1 - di, &mut a_dft, j, &self.data, j);
                }

                if di == 0 {
                    module.vmp_apply_dft_to_dft(&mut res_dft, &a_dft, &rhs.data, scratch_2);
                } else {
                    module.vmp_apply_dft_to_dft_add(&mut res_dft, &a_dft, &rhs.data, di, scratch_2);
                }
            }
        }

        let res_big: VecZnxBig<&mut [u8], B> = module.vec_znx_idft_apply_consume(res_dft);

        for j in 0..cols {
            module.vec_znx_big_normalize(
                basek_in,
                &mut self.data,
                j,
                basek_ggsw,
                &res_big,
                j,
                scratch_1,
            );
        }
    }
}
