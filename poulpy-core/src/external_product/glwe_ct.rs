use poulpy_hal::{
    api::{
        ModuleN, ScratchTakeBasic, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxNormalize,
        VecZnxNormalizeTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataViewMut, Module, Scratch, VecZnx, VecZnxBig},
};

use crate::{
    ScratchTakeCore,
    layouts::{
        GGSWInfos, GLWE, GLWEInfos, GLWEToMut, GLWEToRef, LWEInfos,
        prepared::{GGSWPrepared, GGSWPreparedToRef},
    },
};

impl GLWE<Vec<u8>> {
    pub fn external_product_tmp_bytes<R, A, B, M, BE: Backend>(module: &M, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos,
        M: GLWEExternalProduct<BE>,
    {
        module.glwe_external_product_tmp_bytes(res_infos, a_infos, b_infos)
    }
}

impl<DataSelf: DataMut> GLWE<DataSelf> {
    pub fn external_product<A, B, M, BE: Backend>(&mut self, module: &M, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        A: GLWEToRef,
        B: GGSWPreparedToRef<BE>,
        M: GLWEExternalProduct<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_external_product(self, a, b, scratch);
    }

    pub fn external_product_inplace<A, M, BE: Backend>(&mut self, module: &M, a: &A, scratch: &mut Scratch<BE>)
    where
        A: GGSWPreparedToRef<BE>,
        M: GLWEExternalProduct<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_external_product_inplace(self, a, scratch);
    }
}

pub trait GLWEExternalProduct<BE: Backend>
where
    Self: Sized
        + ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAdd<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxNormalize<BE>,
{
    fn glwe_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos,
    {
        let in_size: usize = a_infos
            .k()
            .div_ceil(b_infos.base2k())
            .div_ceil(b_infos.dsize().into()) as usize;
        let out_size: usize = res_infos.size();
        let ggsw_size: usize = b_infos.size();
        let res_dft: usize = self.bytes_of_vec_znx_dft((b_infos.rank() + 1).into(), ggsw_size);
        let a_dft: usize = self.bytes_of_vec_znx_dft((b_infos.rank() + 1).into(), in_size);
        let vmp: usize = self.vmp_apply_dft_to_dft_tmp_bytes(
            out_size,
            in_size,
            in_size,                     // rows
            (b_infos.rank() + 1).into(), // cols in
            (b_infos.rank() + 1).into(), // cols out
            ggsw_size,
        );
        let normalize_big: usize = self.vec_znx_normalize_tmp_bytes();

        if a_infos.base2k() == b_infos.base2k() {
            res_dft + a_dft + (vmp | normalize_big)
        } else {
            let normalize_conv: usize = VecZnx::bytes_of(self.n(), (b_infos.rank() + 1).into(), in_size);
            res_dft + ((a_dft + normalize_conv + (self.vec_znx_normalize_tmp_bytes() | vmp)) | normalize_big)
        }
    }

    fn glwe_external_product_inplace<R, D>(&self, res: &mut R, a: &D, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        D: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let rhs: &GGSWPrepared<&[u8], BE> = &a.to_ref();

        let basek_in: usize = res.base2k().into();
        let basek_ggsw: usize = rhs.base2k().into();

        #[cfg(debug_assertions)]
        {
            use poulpy_hal::api::ScratchAvailable;

            assert_eq!(rhs.rank(), res.rank());
            assert_eq!(rhs.n(), res.n());
            assert!(scratch.available() >= self.glwe_external_product_tmp_bytes(res, res, rhs));
        }

        let cols: usize = (rhs.rank() + 1).into();
        let dsize: usize = rhs.dsize().into();
        let a_size: usize = (res.size() * basek_in).div_ceil(basek_ggsw);

        let (mut res_dft, scratch_1) = scratch.take_vec_znx_dft(self, cols, rhs.size()); // Todo optimise
        let (mut a_dft, scratch_2) = scratch_1.take_vec_znx_dft(self, cols, a_size.div_ceil(dsize));
        a_dft.data_mut().fill(0);

        if basek_in == basek_ggsw {
            for di in 0..dsize {
                // (lhs.size() + di) / dsize = (a - (digit - di - 1)).div_ceil(dsize)
                a_dft.set_size((res.size() + di) / dsize);

                // Small optimization for dsize > 2
                // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
                // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(dsize-1) * B}.
                // As such we can ignore the last dsize-2 limbs safely of the sum of vmp products.
                // It is possible to further ignore the last dsize-1 limbs, but this introduce
                // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                // noise is kept with respect to the ideal functionality.
                res_dft.set_size(rhs.size() - ((dsize - di) as isize - 2).max(0) as usize);

                for j in 0..cols {
                    self.vec_znx_dft_apply(dsize, dsize - 1 - di, &mut a_dft, j, &res.data, j);
                }

                if di == 0 {
                    self.vmp_apply_dft_to_dft(&mut res_dft, &a_dft, &rhs.data, scratch_2);
                } else {
                    self.vmp_apply_dft_to_dft_add(&mut res_dft, &a_dft, &rhs.data, di, scratch_2);
                }
            }
        } else {
            let (mut a_conv, scratch_3) = scratch_2.take_vec_znx(self, cols, a_size);

            for j in 0..cols {
                self.vec_znx_normalize(
                    basek_ggsw,
                    &mut a_conv,
                    j,
                    basek_in,
                    &res.data,
                    j,
                    scratch_3,
                );
            }

            for di in 0..dsize {
                // (lhs.size() + di) / dsize = (a - (digit - di - 1)).div_ceil(dsize)
                a_dft.set_size((res.size() + di) / dsize);

                // Small optimization for dsize > 2
                // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
                // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(dsize-1) * B}.
                // As such we can ignore the last dsize-2 limbs safely of the sum of vmp products.
                // It is possible to further ignore the last dsize-1 limbs, but this introduce
                // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                // noise is kept with respect to the ideal functionality.
                res_dft.set_size(rhs.size() - ((dsize - di) as isize - 2).max(0) as usize);

                for j in 0..cols {
                    self.vec_znx_dft_apply(dsize, dsize - 1 - di, &mut a_dft, j, &res.data, j);
                }

                if di == 0 {
                    self.vmp_apply_dft_to_dft(&mut res_dft, &a_dft, &rhs.data, scratch_2);
                } else {
                    self.vmp_apply_dft_to_dft_add(&mut res_dft, &a_dft, &rhs.data, di, scratch_2);
                }
            }
        }

        let res_big: VecZnxBig<&mut [u8], BE> = self.vec_znx_idft_apply_consume(res_dft);

        for j in 0..cols {
            self.vec_znx_big_normalize(
                basek_in,
                &mut res.data,
                j,
                basek_ggsw,
                &res_big,
                j,
                scratch_1,
            );
        }
    }

    fn glwe_external_product<R, A, D>(&self, res: &mut R, lhs: &A, rhs: &D, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        D: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let lhs: &GLWE<&[u8]> = &lhs.to_ref();

        let rhs: &GGSWPrepared<&[u8], BE> = &rhs.to_ref();

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
            assert!(scratch.available() >= self.glwe_external_product_tmp_bytes(res, lhs, rhs));
        }

        let cols: usize = (rhs.rank() + 1).into();
        let dsize: usize = rhs.dsize().into();

        let a_size: usize = (lhs.size() * basek_in).div_ceil(basek_ggsw);

        let (mut res_dft, scratch_1) = scratch.take_vec_znx_dft(self, cols, rhs.size()); // Todo optimise
        let (mut a_dft, scratch_2) = scratch_1.take_vec_znx_dft(self, cols, a_size.div_ceil(dsize));
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
            let (mut a_conv, scratch_3) = scratch_2.take_vec_znx(self, cols, a_size);

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

impl<BE: Backend> GLWEExternalProduct<BE> for Module<BE> where
    Self: ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAdd<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxNormalize<BE>
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxNormalizeTmpBytes
{
}
