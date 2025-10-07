use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftApply,
        VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAdd,
        VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, DataViewMut, Module, Scratch, VecZnx, VecZnxBig},
};

use crate::{
    external_product::ExternalProduct,
    layouts::{
        GGSWInfos, GLWECiphertext, GLWEInfos, LWEInfos,
        prepared::{GGSWCiphertextPrepared, GGSWCiphertextPreparedToRef},
    },
};

impl GLWECiphertext<Vec<u8>> {
    #[allow(clippy::too_many_arguments)]
    pub fn external_product_scratch_space<B: Backend, OUT, IN, GGSW>(
        module: &Module<B>,
        out_infos: &OUT,
        in_infos: &IN,
        apply_infos: &GGSW,
    ) -> usize
    where
        OUT: GLWEInfos,
        IN: GLWEInfos,
        GGSW: GGSWInfos,
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxNormalizeTmpBytes,
    {
        let in_size: usize = in_infos
            .k()
            .div_ceil(apply_infos.base2k())
            .div_ceil(apply_infos.dsize().into()) as usize;
        let out_size: usize = out_infos.size();
        let ggsw_size: usize = apply_infos.size();
        let res_dft: usize = module.vec_znx_dft_alloc_bytes((apply_infos.rank() + 1).into(), ggsw_size);
        let a_dft: usize = module.vec_znx_dft_alloc_bytes((apply_infos.rank() + 1).into(), in_size);
        let vmp: usize = module.vmp_apply_dft_to_dft_tmp_bytes(
            out_size,
            in_size,
            in_size,                         // rows
            (apply_infos.rank() + 1).into(), // cols in
            (apply_infos.rank() + 1).into(), // cols out
            ggsw_size,
        );
        let normalize_big: usize = module.vec_znx_normalize_tmp_bytes();

        if in_infos.base2k() == apply_infos.base2k() {
            res_dft + a_dft + (vmp | normalize_big)
        } else {
            let normalize_conv: usize = VecZnx::alloc_bytes(module.n(), (apply_infos.rank() + 1).into(), in_size);
            res_dft + ((a_dft + normalize_conv + (module.vec_znx_normalize_tmp_bytes() | vmp)) | normalize_big)
        }
    }

    pub fn external_product_inplace_scratch_space<B: Backend, OUT, GGSW>(
        module: &Module<B>,
        out_infos: &OUT,
        apply_infos: &GGSW,
    ) -> usize
    where
        OUT: GLWEInfos,
        GGSW: GGSWInfos,
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxNormalizeTmpBytes,
    {
        Self::external_product_scratch_space(module, out_infos, out_infos, apply_infos)
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
        Module<B>: ExternalProduct<Self, GLWECiphertext<DataLhs>, B>,
    {
        module.external_product(self, lhs, rhs, scratch);
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
        let basek_in: usize = self.base2k().into();
        let basek_ggsw: usize = rhs.base2k().into();

        #[cfg(debug_assertions)]
        {
            use poulpy_hal::api::ScratchAvailable;

            assert_eq!(rhs.rank(), self.rank());
            assert_eq!(rhs.n(), self.n());
            assert!(scratch.available() >= GLWECiphertext::external_product_inplace_scratch_space(module, self, rhs,));
        }

        let cols: usize = (rhs.rank() + 1).into();
        let dsize: usize = rhs.dsize().into();
        let a_size: usize = (self.size() * basek_in).div_ceil(basek_ggsw);

        let (mut res_dft, scratch_1) = scratch.take_vec_znx_dft(self.n().into(), cols, rhs.size()); // Todo optimise
        let (mut a_dft, scratch_2) = scratch_1.take_vec_znx_dft(self.n().into(), cols, a_size.div_ceil(dsize));
        a_dft.data_mut().fill(0);

        if basek_in == basek_ggsw {
            for di in 0..dsize {
                // (lhs.size() + di) / dsize = (a - (digit - di - 1)).div_ceil(dsize)
                a_dft.set_size((self.size() + di) / dsize);

                // Small optimization for dsize > 2
                // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
                // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(dsize-1) * B}.
                // As such we can ignore the last dsize-2 limbs safely of the sum of vmp products.
                // It is possible to further ignore the last dsize-1 limbs, but this introduce
                // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                // noise is kept with respect to the ideal functionality.
                res_dft.set_size(rhs.size() - ((dsize - di) as isize - 2).max(0) as usize);

                for j in 0..cols {
                    module.vec_znx_dft_apply(dsize, dsize - 1 - di, &mut a_dft, j, &self.data, j);
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

            for di in 0..dsize {
                // (lhs.size() + di) / dsize = (a - (digit - di - 1)).div_ceil(dsize)
                a_dft.set_size((self.size() + di) / dsize);

                // Small optimization for dsize > 2
                // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
                // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(dsize-1) * B}.
                // As such we can ignore the last dsize-2 limbs safely of the sum of vmp products.
                // It is possible to further ignore the last dsize-1 limbs, but this introduce
                // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                // noise is kept with respect to the ideal functionality.
                res_dft.set_size(rhs.size() - ((dsize - di) as isize - 2).max(0) as usize);

                for j in 0..cols {
                    module.vec_znx_dft_apply(dsize, dsize - 1 - di, &mut a_dft, j, &self.data, j);
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

impl<DM: DataMut, DR: DataRef, BE: Backend> ExternalProduct<GLWECiphertext<DM>, GLWECiphertext<DR>, BE> for Module<BE>
where
    Module<BE>: VecZnxDftAllocBytes
        + VmpApplyDftToDftTmpBytes
        + VecZnxNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAdd<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxNormalize<BE>,
    Scratch<BE>: TakeVecZnxDft<BE> + ScratchAvailable + TakeVecZnx,
{
    fn external_product<D>(&self, res: &mut GLWECiphertext<DM>, lhs: &GLWECiphertext<DR>, rhs: &D, scratch: &mut Scratch<BE>)
    where
        D: GGSWCiphertextPreparedToRef<BE>,
    {
        let rhs: &GGSWCiphertextPrepared<&[u8], BE> = &rhs.to_ref();

        let basek_in: usize = lhs.base2k().into();
        let basek_ggsw: usize = rhs.base2k().into();
        let basek_out: usize = res.base2k().into();

        #[cfg(debug_assertions)]
        {
            use poulpy_hal::api::ScratchAvailable;

            assert_eq!(rhs.rank(), lhs.rank());
            assert_eq!(rhs.rank(), res.rank());
            assert_eq!(rhs.n(), res.n());
            assert_eq!(lhs.n(), res.n());
            assert!(scratch.available() >= GLWECiphertext::external_product_scratch_space(self, res, lhs, rhs));
        }

        let cols: usize = (rhs.rank() + 1).into();
        let dsize: usize = rhs.dsize().into();

        let a_size: usize = (lhs.size() * basek_in).div_ceil(basek_ggsw);

        let (mut res_dft, scratch_1) = scratch.take_vec_znx_dft(self.n().into(), cols, rhs.size()); // Todo optimise
        let (mut a_dft, scratch_2) = scratch_1.take_vec_znx_dft(self.n().into(), cols, a_size.div_ceil(dsize));
        a_dft.data_mut().fill(0);

        if basek_in == basek_ggsw {
            for di in 0..dsize {
                // (lhs.size() + di) / dsize = (a - (digit - di - 1)).div_ceil(dsize)
                a_dft.set_size((lhs.size() + di) / dsize);

                // Small optimization for dsize > 2
                // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
                // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(dsize-1) * B}.
                // As such we can ignore the last dsize-2 limbs safely of the sum of vmp products.
                // It is possible to further ignore the last dsize-1 limbs, but this introduce
                // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                // noise is kept with respect to the ideal functionality.
                res_dft.set_size(rhs.size() - ((dsize - di) as isize - 2).max(0) as usize);

                for j in 0..cols {
                    self.vec_znx_dft_apply(dsize, dsize - 1 - di, &mut a_dft, j, &lhs.data, j);
                }

                if di == 0 {
                    self.vmp_apply_dft_to_dft(&mut res_dft, &a_dft, &rhs.data, scratch_2);
                } else {
                    self.vmp_apply_dft_to_dft_add(&mut res_dft, &a_dft, &rhs.data, di, scratch_2);
                }
            }
        } else {
            let (mut a_conv, scratch_3) = scratch_2.take_vec_znx(self.n(), cols, a_size);

            for j in 0..cols {
                self.vec_znx_normalize(
                    basek_ggsw,
                    &mut a_conv,
                    j,
                    basek_in,
                    &lhs.data,
                    j,
                    scratch_3,
                );
            }

            for di in 0..dsize {
                // (lhs.size() + di) / dsize = (a - (digit - di - 1)).div_ceil(dsize)
                a_dft.set_size((a_size + di) / dsize);

                // Small optimization for dsize > 2
                // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
                // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(dsize-1) * B}.
                // As such we can ignore the last dsize-2 limbs safely of the sum of vmp products.
                // It is possible to further ignore the last dsize-1 limbs, but this introduce
                // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                // noise is kept with respect to the ideal functionality.
                res_dft.set_size(rhs.size() - ((dsize - di) as isize - 2).max(0) as usize);

                for j in 0..cols {
                    self.vec_znx_dft_apply(dsize, dsize - 1 - di, &mut a_dft, j, &a_conv, j);
                }

                if di == 0 {
                    self.vmp_apply_dft_to_dft(&mut res_dft, &a_dft, &rhs.data, scratch_3);
                } else {
                    self.vmp_apply_dft_to_dft_add(&mut res_dft, &a_dft, &rhs.data, di, scratch_3);
                }
            }
        }

        let res_big: VecZnxBig<&mut [u8], BE> = self.vec_znx_idft_apply_consume(res_dft);

        (0..cols).for_each(|i| {
            self.vec_znx_big_normalize(
                basek_out,
                res.data_mut(),
                i,
                basek_ggsw,
                &res_big,
                i,
                scratch_1,
            );
        });
    }
}
