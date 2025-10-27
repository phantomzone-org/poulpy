use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, ScratchTakeBasic, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataViewMut, Module, Scratch, VecZnx, VecZnxBig, VecZnxDft, VecZnxDftToRef, VmpPMat, ZnxInfos},
};

use crate::{
    ScratchTakeCore,
    layouts::{GGLWEInfos, GGLWEPrepared, GGLWEPreparedToRef, GLWE, GLWEInfos, GLWEToMut, GLWEToRef, LWEInfos},
};

impl GLWE<Vec<u8>> {
    pub fn keyswitch_tmp_bytes<M, R, A, B, BE: Backend>(module: &M, res_infos: &R, a_infos: &A, key_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos,
        M: GLWEKeyswitch<BE>,
    {
        module.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
    }
}

impl<D: DataMut> GLWE<D> {
    pub fn keyswitch<A, B, M, BE: Backend>(&mut self, module: &M, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        A: GLWEToRef,
        B: GGLWEPreparedToRef<BE>,
        M: GLWEKeyswitch<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_keyswitch(self, a, b, scratch);
    }

    pub fn keyswitch_inplace<A, M, BE: Backend>(&mut self, module: &M, a: &A, scratch: &mut Scratch<BE>)
    where
        A: GGLWEPreparedToRef<BE>,
        M: GLWEKeyswitch<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_keyswitch_inplace(self, a, scratch);
    }
}

impl<BE: Backend> GLWEKeyswitch<BE> for Module<BE>
where
    Self: Sized + GLWEKeySwitchInternal<BE> + VecZnxBigNormalizeTmpBytes + VecZnxBigNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_keyswitch_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, key_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos,
    {
        let cols: usize = res_infos.rank().as_usize() + 1;
        self.glwe_keyswitch_internal_tmp_bytes(res_infos, a_infos, key_infos)
            .max(self.vec_znx_big_normalize_tmp_bytes())
            + self.bytes_of_vec_znx_dft(cols, key_infos.size())
    }

    fn glwe_keyswitch<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        K: GGLWEPreparedToRef<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();
        let b: &GGLWEPrepared<&[u8], BE> = &key.to_ref();

        assert_eq!(
            a.rank(),
            b.rank_in(),
            "a.rank(): {} != b.rank_in(): {}",
            a.rank(),
            b.rank_in()
        );
        assert_eq!(
            res.rank(),
            b.rank_out(),
            "res.rank(): {} != b.rank_out(): {}",
            res.rank(),
            b.rank_out()
        );

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(b.n(), self.n() as u32);

        let scrach_needed: usize = self.glwe_keyswitch_tmp_bytes(res, a, b);

        assert!(
            scratch.available() >= scrach_needed,
            "scratch.available()={} < glwe_keyswitch_tmp_bytes={scrach_needed}",
            scratch.available(),
        );

        let basek_out: usize = res.base2k().into();
        let base2k_out: usize = b.base2k().into();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), b.size()); // Todo optimise
        let res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, a, b, scratch_1);
        for i in 0..(res.rank() + 1).into() {
            self.vec_znx_big_normalize(
                basek_out,
                &mut res.data,
                i,
                base2k_out,
                &res_big,
                i,
                scratch_1,
            );
        }
    }

    fn glwe_keyswitch_inplace<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GGLWEPreparedToRef<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let key: &GGLWEPrepared<&[u8], BE> = &key.to_ref();

        assert_eq!(
            res.rank(),
            key.rank_in(),
            "res.rank(): {} != a.rank_in(): {}",
            res.rank(),
            key.rank_in()
        );
        assert_eq!(
            res.rank(),
            key.rank_out(),
            "res.rank(): {} != b.rank_out(): {}",
            res.rank(),
            key.rank_out()
        );

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(key.n(), self.n() as u32);

        let scrach_needed: usize = self.glwe_keyswitch_tmp_bytes(res, res, key);

        assert!(
            scratch.available() >= scrach_needed,
            "scratch.available()={} < glwe_keyswitch_tmp_bytes={scrach_needed}",
            scratch.available(),
        );

        let base2k_in: usize = res.base2k().into();
        let base2k_out: usize = key.base2k().into();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // Todo optimise
        let res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, res, key, scratch_1);
        for i in 0..(res.rank() + 1).into() {
            self.vec_znx_big_normalize(
                base2k_in,
                &mut res.data,
                i,
                base2k_out,
                &res_big,
                i,
                scratch_1,
            );
        }
    }
}

pub trait GLWEKeyswitch<BE: Backend> {
    fn glwe_keyswitch_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, key_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;

    fn glwe_keyswitch<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        K: GGLWEPreparedToRef<BE>;

    fn glwe_keyswitch_inplace<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GGLWEPreparedToRef<BE>;
}

impl<BE: Backend> GLWEKeySwitchInternal<BE> for Module<BE> where
    Self: GGLWEProduct<BE>
        + VecZnxDftApply<BE>
        + VecZnxNormalize<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddSmallInplace<BE>
        + VecZnxNormalizeTmpBytes
{
}

pub(crate) trait GLWEKeySwitchInternal<BE: Backend>
where
    Self: GGLWEProduct<BE>
        + VecZnxDftApply<BE>
        + VecZnxNormalize<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddSmallInplace<BE>
        + VecZnxNormalizeTmpBytes,
{
    fn glwe_keyswitch_internal_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        let cols: usize = (a_infos.rank() + 1).into();
        let a_size: usize = a_infos.size();

        let a_conv = if a_infos.base2k() == key_infos.base2k() {
            0
        } else {
            VecZnx::bytes_of(self.n(), 1, a_size) + self.vec_znx_normalize_tmp_bytes()
        };

        self.gglwe_product_dft_tmp_bytes(res_infos.size(), a_size, key_infos) + self.bytes_of_vec_znx_dft(cols, a_size) + a_conv
    }

    fn glwe_keyswitch_internal<DR, A, K>(
        &self,
        mut res: VecZnxDft<DR, BE>,
        a: &A,
        key: &K,
        scratch: &mut Scratch<BE>,
    ) -> VecZnxBig<DR, BE>
    where
        DR: DataMut,
        A: GLWEToRef,
        K: GGLWEPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let a: &GLWE<&[u8]> = &a.to_ref();
        let key: &GGLWEPrepared<&[u8], BE> = &key.to_ref();

        let base2k_in: usize = a.base2k().into();
        let base2k_out: usize = key.base2k().into();
        let cols: usize = (a.rank() + 1).into();
        let a_size: usize = (a.size() * base2k_in).div_ceil(base2k_out);

        let (mut a_dft, scratch_1) = scratch.take_vec_znx_dft(self, cols - 1, a_size);

        if base2k_in == base2k_out {
            for col_i in 0..cols - 1 {
                self.vec_znx_dft_apply(1, 0, &mut a_dft, col_i, a.data(), col_i + 1);
            }
        } else {
            let (mut a_conv, scratch_2) = scratch_1.take_vec_znx(self.n(), 1, a_size);
            for i in 0..cols - 1 {
                self.vec_znx_normalize(
                    base2k_out,
                    &mut a_conv,
                    0,
                    base2k_in,
                    a.data(),
                    i + 1,
                    scratch_2,
                );
                self.vec_znx_dft_apply(1, 0, &mut a_dft, i, &a_conv, 0);
            }
        }

        self.gglwe_product_dft(&mut res, &a_dft, key, scratch_1);

        let mut res_big: VecZnxBig<DR, BE> = self.vec_znx_idft_apply_consume(res);
        self.vec_znx_big_add_small_inplace(&mut res_big, 0, a.data(), 0);
        res_big
    }
}

impl<BE: Backend> GGLWEProduct<BE> for Module<BE> where
    Self: Sized
        + ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAdd<BE>
        + VecZnxDftCopy<BE>
{
}

pub(crate) trait GGLWEProduct<BE: Backend>
where
    Self: Sized
        + ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAdd<BE>
        + VecZnxDftCopy<BE>,
{
    fn gglwe_product_dft_tmp_bytes<K>(&self, res_size: usize, a_size: usize, key_infos: &K) -> usize
    where
        K: GGLWEInfos,
    {
        let dsize: usize = key_infos.dsize().as_usize();

        if dsize == 1 {
            self.vmp_apply_dft_to_dft_tmp_bytes(
                res_size,
                a_size,
                key_infos.dnum().into(),
                (key_infos.rank_in()).into(),
                (key_infos.rank_out() + 1).into(),
                key_infos.size(),
            )
        } else {
            let dnum: usize = key_infos.dnum().into();
            let a_size: usize = a_size.div_ceil(dsize).min(dnum);
            let ai_dft: usize = self.bytes_of_vec_znx_dft(key_infos.rank_in().into(), a_size);

            let vmp: usize = self.vmp_apply_dft_to_dft_tmp_bytes(
                res_size,
                a_size,
                dnum,
                (key_infos.rank_in()).into(),
                (key_infos.rank_out() + 1).into(),
                key_infos.size(),
            );

            ai_dft + vmp
        }
    }

    fn gglwe_product_dft<DR, A, K>(&self, res: &mut VecZnxDft<DR, BE>, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        DR: DataMut,
        A: VecZnxDftToRef<BE>,
        K: GGLWEPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let a: &VecZnxDft<&[u8], BE> = &a.to_ref();
        let key: &GGLWEPrepared<&[u8], BE> = &key.to_ref();

        let cols: usize = a.cols();
        let a_size: usize = a.size();
        let pmat: &VmpPMat<&[u8], BE> = &key.data;

        // If dsize == 1, then the digit decomposition is equal to Base2K and we can simply
        // can the vmp API.
        if key.dsize() == 1 {
            self.vmp_apply_dft_to_dft(res, a, pmat, scratch);
        // If dsize != 1, then the digit decomposition is k * Base2K with k > 1.
        // As such we need to perform a bivariate polynomial convolution in (X, Y) / (X^{N}+1) with Y = 2^-K
        // (instead of yn univariate one in X).
        //
        // Since the basis in Y is small (in practice degree 6-7 max), we perform it naiveley.
        // To do so, we group the different limbs of ai_dft by their respective degree in Y
        // which are multiples of the current digit.
        // For example if dsize = 3, with ai_dft = [a0, a1, a2, a3, a4, a5, a6],
        // we group them as [[a0, a3, a5], [a1, a4, a6], [a2, a5, 0]]
        // and evaluate sum(a_di * pmat * 2^{di*Base2k})
        } else {
            let dsize: usize = key.dsize().into();
            let dnum: usize = key.dnum().into();

            // We bound ai_dft size by the number of rows of the matrix
            let (mut ai_dft, scratch_1) = scratch.take_vec_znx_dft(self, cols, a_size.div_ceil(dsize).min(dnum));
            ai_dft.data_mut().fill(0);

            for di in 0..dsize {
                // Sets ai_dft size according to the current digit (if dsize does not divides a_size),
                // bounded by the number of rows (digits) in the prepared matrix.
                ai_dft.set_size(((a_size + di) / dsize).min(dnum));

                // Small optimization for dsize > 2
                // VMP produce some error e, and since we aggregate vmp * 2^{di * Base2k}, then
                // we also aggregate ei * 2^{di * Base2k}, with the largest error being ei * 2^{(dsize-1) * Base2k}.
                // As such we can ignore the last dsize-2 limbs safely of the sum of vmp products.
                // It is possible to further ignore the last dsize-1 limbs, but this introduce
                // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                // noise is kept with respect to the ideal functionality.
                res.set_size(pmat.size() - ((dsize - di) as isize - 2).max(0) as usize);

                for j in 0..cols {
                    self.vec_znx_dft_copy(dsize, dsize - di - 1, &mut ai_dft, j, a, j);
                }

                if di == 0 {
                    // res = pmat * ai_dft
                    self.vmp_apply_dft_to_dft(res, &ai_dft, pmat, scratch_1);
                } else {
                    // res = (pmat * ai_dft) * 2^{di * Base2k}
                    self.vmp_apply_dft_to_dft_add(res, &ai_dft, pmat, di, scratch_1);
                }
            }

            res.set_size(res.max_size());
        }
    }
}
