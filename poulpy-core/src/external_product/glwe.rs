use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, ScratchTakeBasic, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataViewMut, Module, Scratch, VecZnxBig, VecZnxDft, ZnxZero},
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
        A: GLWEToRef + GLWEInfos,
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
        R: GLWEToMut + GLWEInfos,
        D: GGSWPreparedToRef<BE> + GGSWInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_external_product<R, A, D>(&self, res: &mut R, lhs: &A, rhs: &D, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
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
    fn glwe_external_product_tmp_bytes<R, A, B>(&self, res: &R, a: &A, ggsw: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos,
    {
        let cols: usize = res.rank().as_usize() + 1;
        let lvl_0: usize = self.bytes_of_vec_znx_dft(cols, ggsw.size());
        let lvl_1: usize = self.vec_znx_big_normalize_tmp_bytes();
        let lvl_2: usize = if a.base2k() != ggsw.base2k() {
            let a_conv_infos: GLWELayout = GLWELayout {
                n: a.n(),
                base2k: ggsw.base2k(),
                k: a.k(),
                rank: a.rank(),
            };
            let lvl_2_0: usize = GLWE::bytes_of_from_infos(&a_conv_infos);
            let lvl_2_1: usize = self
                .glwe_normalize_tmp_bytes()
                .max(self.glwe_external_product_internal_tmp_bytes(res, &a_conv_infos, ggsw));
            lvl_2_0 + lvl_2_1
        } else {
            self.glwe_external_product_internal_tmp_bytes(res, a, ggsw)
        };

        lvl_0 + lvl_1.max(lvl_2)
    }

    fn glwe_external_product_inplace<R, D>(&self, res: &mut R, ggsw: &D, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        D: GGSWPreparedToRef<BE> + GGSWInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert_eq!(ggsw.rank(), res.rank());
        assert_eq!(ggsw.n(), res.n());
        assert!(
            scratch.available() >= self.glwe_external_product_tmp_bytes(res, res, ggsw),
            "scratch.available(): {} < GLWEExternalProduct::glwe_external_product_tmp_bytes: {}",
            scratch.available(),
            self.glwe_external_product_tmp_bytes(res, res, ggsw)
        );

        let res_base2k: usize = res.base2k().as_usize();
        let ggsw_base2k: usize = ggsw.base2k().as_usize();

        let (mut res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), ggsw.size()); // Todo optimise
        res_dft.zero(); // TODO: REMOVE ONCE ABOVE TAKES CORRECT SIZE

        let res_big: VecZnxBig<&mut [u8], BE> = if res_base2k != ggsw_base2k {
            let (mut res_conv, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: res.n(),
                base2k: ggsw.base2k(),
                k: res.k(),
                rank: res.rank(),
            });
            self.glwe_normalize(&mut res_conv, res, scratch_2);
            self.glwe_external_product_internal(res_dft, &res_conv, ggsw, scratch_2)
        } else {
            self.glwe_external_product_internal(res_dft, res, ggsw, scratch_1)
        };

        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        for j in 0..(res.rank() + 1).into() {
            self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, j, &res_big, ggsw_base2k, j, scratch_1);
        }
    }

    fn glwe_external_product<R, A, G>(&self, res: &mut R, a: &A, ggsw: &G, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        G: GGSWPreparedToRef<BE> + GGSWInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert_eq!(ggsw.rank(), a.rank());
        assert_eq!(ggsw.rank(), res.rank());
        assert_eq!(ggsw.n(), res.n());
        assert_eq!(a.n(), res.n());
        assert!(
            scratch.available() >= self.glwe_external_product_tmp_bytes(res, a, ggsw),
            "scratch.available(): {} < GLWEExternalProduct::glwe_external_product_tmp_bytes: {}",
            scratch.available(),
            self.glwe_external_product_tmp_bytes(res, a, ggsw)
        );

        let a_base2k: usize = a.base2k().into();
        let ggsw_base2k: usize = ggsw.base2k().into();
        let res_base2k: usize = res.base2k().into();

        let (mut res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), ggsw.size()); // Todo optimise
        res_dft.zero(); // TODO: REMOVE ONCE ABOVE TAKES CORRECT SIZE

        let res_big: VecZnxBig<&mut [u8], BE> = if a_base2k != ggsw_base2k {
            let (mut a_conv, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: a.n(),
                base2k: ggsw.base2k(),
                k: a.k(),
                rank: a.rank(),
            });
            self.glwe_normalize(&mut a_conv, a, scratch_2);
            self.glwe_external_product_internal(res_dft, &a_conv, ggsw, scratch_2)
        } else {
            self.glwe_external_product_internal(res_dft, a, ggsw, scratch_1)
        };

        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        for j in 0..(res.rank() + 1).into() {
            self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, j, &res_big, ggsw_base2k, j, scratch_1);
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
        Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable;
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
        let in_size: usize = a_infos.k().div_ceil(b_infos.base2k()).div_ceil(b_infos.dsize().into()) as usize;
        let out_size: usize = res_infos.size();
        let ggsw_size: usize = b_infos.size();
        let lvl_0: usize = self.bytes_of_vec_znx_dft((b_infos.rank() + 1).into(), in_size);
        let lvl_1: usize = self.vmp_apply_dft_to_dft_tmp_bytes(
            out_size,
            in_size,
            in_size,                     // rows
            (b_infos.rank() + 1).into(), // cols in
            (b_infos.rank() + 1).into(), // cols out
            ggsw_size,
        );
        lvl_0 + lvl_1
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
        Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
    {
        let a: &GLWE<&[u8]> = &a.to_ref();
        let ggsw: &GGSWPrepared<&[u8], BE> = &ggsw.to_ref();

        assert_eq!(a.base2k(), ggsw.base2k());
        assert!(
            scratch.available() >= self.glwe_external_product_internal_tmp_bytes(ggsw, a, ggsw),
            "scratch.available(): {} < GLWEExternalProductInternal::glwe_external_product_internal_tmp_bytes: {}",
            scratch.available(),
            self.glwe_external_product_internal_tmp_bytes(ggsw, a, ggsw)
        );

        let cols: usize = (ggsw.rank() + 1).into();
        let dsize: usize = ggsw.dsize().into();
        let a_size: usize = a.size();

        let (mut a_dft, scratch_1) = scratch.take_vec_znx_dft(self, cols, a_size.div_ceil(dsize));
        a_dft.data_mut().fill(0);

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

        self.vec_znx_idft_apply_consume(res_dft)
    }
}
