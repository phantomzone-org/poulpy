use poulpy_hal::{
    api::{
        ModuleN, ScratchTakeBasic, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftApply,
        VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataViewMut, Module, Scratch, VecZnx, VecZnxBig, VecZnxDft},
};

use crate::{
    GLWENormalize, ScratchTakeCore,
    layouts::{
        GGSWInfos, GLWE, GLWEInfos, GLWELayout, GLWEToMut, GLWEToRef, LWEInfos,
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
        B: GGSWPreparedToRef<BE> + GGSWInfos,
        M: GLWEExternalProduct<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_external_product(self, a, b, scratch);
    }

    pub fn external_product_inplace<A, M, BE: Backend>(&mut self, module: &M, a: &A, scratch: &mut Scratch<BE>)
    where
        A: GGSWPreparedToRef<BE> + GGSWInfos,
        M: GLWEExternalProduct<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_external_product_inplace(self, a, scratch);
    }
}

pub trait GLWEExternalProduct<BE: Backend> {
    fn glwe_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos;

    fn glwe_external_product_inplace<R, D>(&self, res: &mut R, a: &D, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        D: GGSWPreparedToRef<BE> + GGSWInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_external_product<R, A, D>(&self, res: &mut R, lhs: &A, rhs: &D, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        D: GGSWPreparedToRef<BE> + GGSWInfos,
        Scratch<BE>: ScratchTakeCore<BE>;
    fn glwe_external_product_add<R, A, D>(&self, res: &mut R, lhs: &A, rhs: &D, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        D: GGSWPreparedToRef<BE> + GGSWInfos,
        Scratch<BE>: ScratchTakeCore<BE>;
}

impl<BE: Backend> GLWEExternalProduct<BE> for Module<BE>
where
    Self: GLWEExternalProductInternal<BE>
        + VecZnxDftBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigAddSmallInplace<BE>
        + GLWENormalize<BE>,
{
    fn glwe_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos,
    {
        let res_dft: usize = self.bytes_of_vec_znx_dft((b_infos.rank() + 1).into(), b_infos.size());
        res_dft
            + self
                .glwe_external_product_internal_tmp_bytes(res_infos, a_infos, b_infos)
                .max(self.vec_znx_big_normalize_tmp_bytes())
    }

    fn glwe_external_product_inplace<R, D>(&self, res: &mut R, a: &D, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        D: GGSWPreparedToRef<BE> + GGSWInfos,
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

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), a.size()); // Todo optimise
        let res_big = self.glwe_external_product_internal(res_dft, res, a, scratch_1);
        for j in 0..(res.rank() + 1).into() {
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

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), rhs.size()); // Todo optimise
        let res_big = self.glwe_external_product_internal(res_dft, lhs, rhs, scratch_1);

        for j in 0..(res.rank() + 1).into() {
            self.vec_znx_big_normalize(
                basek_out,
                &mut res.data,
                j,
                basek_ggsw,
                &res_big,
                j,
                scratch_1,
            );
        }
    }

    fn glwe_external_product_add<R, A, D>(&self, res: &mut R, a: &A, key: &D, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        D: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();
        let key: &GGSWPrepared<&[u8], BE> = &key.to_ref();

        assert_eq!(a.base2k(), res.base2k());

        let res_base2k: usize = res.base2k().into();
        let key_base2k: usize = key.base2k().into();

        #[cfg(debug_assertions)]
        {
            use poulpy_hal::api::ScratchAvailable;

            assert_eq!(key.rank(), a.rank());
            assert_eq!(key.rank(), res.rank());
            assert_eq!(key.n(), res.n());
            assert_eq!(a.n(), res.n());
            assert!(scratch.available() >= self.glwe_external_product_tmp_bytes(res, a, key));
        }

        if res_base2k == key_base2k {
            let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // Todo optimise
            let mut res_big = self.glwe_external_product_internal(res_dft, a, key, scratch_1);
            for j in 0..(res.rank() + 1).into() {
                self.vec_znx_big_add_small_inplace(&mut res_big, j, res.data(), j);
                self.vec_znx_big_normalize(
                    res_base2k,
                    &mut res.data,
                    j,
                    key_base2k,
                    &res_big,
                    j,
                    scratch_1,
                );
            }
        } else {
            let (mut a_conv, scratch_1) = scratch.take_glwe(&GLWELayout {
                n: a.n(),
                base2k: key.base2k(),
                k: a.k(),
                rank: a.rank(),
            });
            let (mut res_conv, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: res.n(),
                base2k: key.base2k(),
                k: res.k(),
                rank: res.rank(),
            });
            self.glwe_normalize(&mut a_conv, a, scratch_2);
            self.glwe_normalize(&mut res_conv, res, scratch_2);
            let (res_dft, scratch_2) = scratch_2.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // Todo optimise
            let mut res_big = self.glwe_external_product_internal(res_dft, &a_conv, key, scratch_2);
            for j in 0..(res.rank() + 1).into() {
                self.vec_znx_big_add_small_inplace(&mut res_big, j, res_conv.data(), j);
                self.vec_znx_big_normalize(
                    res_base2k,
                    &mut res.data,
                    j,
                    key_base2k,
                    &res_big,
                    j,
                    scratch_2,
                );
            }
        }
    }
}

pub trait GLWEExternalProductInternal<BE: Backend> {
    fn glwe_external_product_internal_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos;
    fn glwe_external_product_internal<DR, A, G>(
        &self,
        res_dft: VecZnxDft<DR, BE>,
        a: &A,
        ggsw: &G,
        scratch: &mut Scratch<BE>,
    ) -> VecZnxBig<DR, BE>
    where
        DR: DataMut,
        A: GLWEToRef,
        G: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;
}

impl<BE: Backend> GLWEExternalProductInternal<BE> for Module<BE>
where
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
        + VecZnxNormalizeTmpBytes,
{
    fn glwe_external_product_internal_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
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
            a_dft + (vmp | normalize_big)
        } else {
            let normalize_conv: usize = VecZnx::bytes_of(self.n(), (b_infos.rank() + 1).into(), in_size);
            (a_dft + normalize_conv + (self.vec_znx_normalize_tmp_bytes() | vmp)) | normalize_big
        }
    }

    fn glwe_external_product_internal<DR, A, G>(
        &self,
        mut res_dft: VecZnxDft<DR, BE>,
        a: &A,
        ggsw: &G,
        scratch: &mut Scratch<BE>,
    ) -> VecZnxBig<DR, BE>
    where
        DR: DataMut,
        A: GLWEToRef,
        G: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let a: &GLWE<&[u8]> = &a.to_ref();
        let ggsw: &GGSWPrepared<&[u8], BE> = &ggsw.to_ref();

        let basek_in: usize = a.base2k().into();
        let basek_ggsw: usize = ggsw.base2k().into();

        let cols: usize = (ggsw.rank() + 1).into();
        let dsize: usize = ggsw.dsize().into();
        let a_size: usize = (a.size() * basek_in).div_ceil(basek_ggsw);

        let (mut a_dft, scratch_1) = scratch.take_vec_znx_dft(self, cols, a_size.div_ceil(dsize));
        a_dft.data_mut().fill(0);

        if basek_in == basek_ggsw {
            for di in 0..dsize {
                // (lhs.size() + di) / dsize = (a - (digit - di - 1)).div_ceil(dsize)
                a_dft.set_size((a.size() + di) / dsize);

                // Small optimization for dsize > 2
                // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
                // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(dsize-1) * B}.
                // As such we can ignore the last dsize-2 limbs safely of the sum of vmp products.
                // It is possible to further ignore the last dsize-1 limbs, but this introduce
                // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                // noise is kept with respect to the ideal functionality.
                res_dft.set_size(ggsw.size() - ((dsize - di) as isize - 2).max(0) as usize);

                for j in 0..cols {
                    self.vec_znx_dft_apply(dsize, dsize - 1 - di, &mut a_dft, j, &a.data, j);
                }

                if di == 0 {
                    self.vmp_apply_dft_to_dft(&mut res_dft, &a_dft, &ggsw.data, scratch_1);
                } else {
                    self.vmp_apply_dft_to_dft_add(&mut res_dft, &a_dft, &ggsw.data, di, scratch_1);
                }
            }
        } else {
            let (mut a_conv, scratch_3) = scratch_1.take_vec_znx(self.n(), cols, a_size);

            for j in 0..cols {
                self.vec_znx_normalize(basek_ggsw, &mut a_conv, j, basek_in, &a.data, j, scratch_3);
            }

            for di in 0..dsize {
                // (lhs.size() + di) / dsize = (a - (digit - di - 1)).div_ceil(dsize)
                a_dft.set_size((a.size() + di) / dsize);

                // Small optimization for dsize > 2
                // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
                // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(dsize-1) * B}.
                // As such we can ignore the last dsize-2 limbs safely of the sum of vmp products.
                // It is possible to further ignore the last dsize-1 limbs, but this introduce
                // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                // noise is kept with respect to the ideal functionality.
                res_dft.set_size(ggsw.size() - ((dsize - di) as isize - 2).max(0) as usize);

                for j in 0..cols {
                    self.vec_znx_dft_apply(dsize, dsize - 1 - di, &mut a_dft, j, &a.data, j);
                }

                if di == 0 {
                    self.vmp_apply_dft_to_dft(&mut res_dft, &a_dft, &ggsw.data, scratch_1);
                } else {
                    self.vmp_apply_dft_to_dft_add(&mut res_dft, &a_dft, &ggsw.data, di, scratch_1);
                }
            }
        }

        self.vec_znx_idft_apply_consume(res_dft)
    }
}
