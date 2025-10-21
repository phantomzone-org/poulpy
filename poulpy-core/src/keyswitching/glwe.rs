use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, ScratchTakeBasic, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataViewMut, Module, Scratch, VecZnx, VecZnxBig, VecZnxDft, VmpPMat, ZnxInfos},
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

impl<BE: Backend> GLWEKeyswitch<BE> for Module<BE> where
    Self: Sized
        + ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAdd<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddSmallInplace<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
{
}

pub trait GLWEKeyswitch<BE: Backend>
where
    Self: Sized
        + ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAdd<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddSmallInplace<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes,
{
    fn glwe_keyswitch_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, key_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos,
    {
        let in_size: usize = a_infos
            .k()
            .div_ceil(key_infos.base2k())
            .div_ceil(key_infos.dsize().into()) as usize;
        let out_size: usize = res_infos.size();
        let ksk_size: usize = key_infos.size();
        let res_dft: usize = self.bytes_of_vec_znx_dft((key_infos.rank_out() + 1).into(), ksk_size); // TODO OPTIMIZE
        let ai_dft: usize = self.bytes_of_vec_znx_dft((key_infos.rank_in()).into(), in_size);
        let vmp: usize = self.vmp_apply_dft_to_dft_tmp_bytes(
            out_size,
            in_size,
            in_size,
            (key_infos.rank_in()).into(),
            (key_infos.rank_out() + 1).into(),
            ksk_size,
        ) + self.bytes_of_vec_znx_dft((key_infos.rank_in()).into(), in_size);
        let normalize_big: usize = self.vec_znx_big_normalize_tmp_bytes();
        if a_infos.base2k() == key_infos.base2k() {
            res_dft + ((ai_dft + vmp) | normalize_big)
        } else if key_infos.dsize() == 1 {
            // In this case, we only need one column, temporary, that we can drop once a_dft is computed.
            let normalize_conv: usize = VecZnx::bytes_of(self.n(), 1, in_size) + self.vec_znx_normalize_tmp_bytes();
            res_dft + (((ai_dft + normalize_conv) | vmp) | normalize_big)
        } else {
            // Since we stride over a to get a_dft when dsize > 1, we need to store the full columns of a with in the base conversion.
            let normalize_conv: usize = VecZnx::bytes_of(self.n(), (key_infos.rank_in()).into(), in_size);
            res_dft + ((ai_dft + normalize_conv + (self.vec_znx_normalize_tmp_bytes() | vmp)) | normalize_big)
        }
    }

    fn glwe_keyswitch<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        K: GGLWEPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
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
        let res_big: VecZnxBig<&mut [u8], BE> = keyswitch_internal(self, res_dft, a, b, scratch_1);
        (0..(res.rank() + 1).into()).for_each(|i| {
            self.vec_znx_big_normalize(
                basek_out,
                &mut res.data,
                i,
                base2k_out,
                &res_big,
                i,
                scratch_1,
            );
        })
    }

    fn glwe_keyswitch_inplace<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GGLWEPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GGLWEPrepared<&[u8], BE> = &key.to_ref();

        assert_eq!(
            res.rank(),
            a.rank_in(),
            "res.rank(): {} != a.rank_in(): {}",
            res.rank(),
            a.rank_in()
        );
        assert_eq!(
            res.rank(),
            a.rank_out(),
            "res.rank(): {} != b.rank_out(): {}",
            res.rank(),
            a.rank_out()
        );

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);

        let scrach_needed: usize = self.glwe_keyswitch_tmp_bytes(res, res, a);

        assert!(
            scratch.available() >= scrach_needed,
            "scratch.available()={} < glwe_keyswitch_tmp_bytes={scrach_needed}",
            scratch.available(),
        );

        let base2k_in: usize = res.base2k().into();
        let base2k_out: usize = a.base2k().into();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), a.size()); // Todo optimise
        let res_big: VecZnxBig<&mut [u8], BE> = keyswitch_internal(self, res_dft, res, a, scratch_1);
        (0..(res.rank() + 1).into()).for_each(|i| {
            self.vec_znx_big_normalize(
                base2k_in,
                &mut res.data,
                i,
                base2k_out,
                &res_big,
                i,
                scratch_1,
            );
        })
    }
}

impl GLWE<Vec<u8>> {}

impl<DataSelf: DataMut> GLWE<DataSelf> {}

pub(crate) fn keyswitch_internal<BE: Backend, M, DR, A, K>(
    module: &M,
    mut res: VecZnxDft<DR, BE>,
    a: &A,
    key: &K,
    scratch: &mut Scratch<BE>,
) -> VecZnxBig<DR, BE>
where
    DR: DataMut,
    A: GLWEToRef,
    K: GGLWEPreparedToRef<BE>,
    M: ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAdd<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddSmallInplace<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let a: &GLWE<&[u8]> = &a.to_ref();
    let key: &GGLWEPrepared<&[u8], BE> = &key.to_ref();

    let base2k_in: usize = a.base2k().into();
    let base2k_out: usize = key.base2k().into();
    let cols: usize = (a.rank() + 1).into();
    let a_size: usize = (a.size() * base2k_in).div_ceil(base2k_out);
    let pmat: &VmpPMat<&[u8], BE> = &key.data;

    if key.dsize() == 1 {
        let (mut ai_dft, scratch_1) = scratch.take_vec_znx_dft(module, cols - 1, a.size());

        if base2k_in == base2k_out {
            (0..cols - 1).for_each(|col_i| {
                module.vec_znx_dft_apply(1, 0, &mut ai_dft, col_i, a.data(), col_i + 1);
            });
        } else {
            let (mut a_conv, scratch_2) = scratch_1.take_vec_znx(module, 1, a_size);
            (0..cols - 1).for_each(|col_i| {
                module.vec_znx_normalize(
                    base2k_out,
                    &mut a_conv,
                    0,
                    base2k_in,
                    a.data(),
                    col_i + 1,
                    scratch_2,
                );
                module.vec_znx_dft_apply(1, 0, &mut ai_dft, col_i, &a_conv, 0);
            });
        }

        module.vmp_apply_dft_to_dft(&mut res, &ai_dft, pmat, scratch_1);
    } else {
        let dsize: usize = key.dsize().into();

        let (mut ai_dft, scratch_1) = scratch.take_vec_znx_dft(module, cols - 1, a_size.div_ceil(dsize));
        ai_dft.data_mut().fill(0);

        if base2k_in == base2k_out {
            for di in 0..dsize {
                ai_dft.set_size((a_size + di) / dsize);

                // Small optimization for dsize > 2
                // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
                // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(dsize-1) * B}.
                // As such we can ignore the last dsize-2 limbs safely of the sum of vmp products.
                // It is possible to further ignore the last dsize-1 limbs, but this introduce
                // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                // noise is kept with respect to the ideal functionality.
                res.set_size(pmat.size() - ((dsize - di) as isize - 2).max(0) as usize);

                for j in 0..cols - 1 {
                    module.vec_znx_dft_apply(dsize, dsize - di - 1, &mut ai_dft, j, a.data(), j + 1);
                }

                if di == 0 {
                    module.vmp_apply_dft_to_dft(&mut res, &ai_dft, pmat, scratch_1);
                } else {
                    module.vmp_apply_dft_to_dft_add(&mut res, &ai_dft, pmat, di, scratch_1);
                }
            }
        } else {
            let (mut a_conv, scratch_2) = scratch_1.take_vec_znx(module, cols - 1, a_size);
            for j in 0..cols - 1 {
                module.vec_znx_normalize(
                    base2k_out,
                    &mut a_conv,
                    j,
                    base2k_in,
                    a.data(),
                    j + 1,
                    scratch_2,
                );
            }

            for di in 0..dsize {
                ai_dft.set_size((a_size + di) / dsize);

                // Small optimization for dsize > 2
                // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
                // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(dsize-1) * B}.
                // As such we can ignore the last dsize-2 limbs safely of the sum of vmp products.
                // It is possible to further ignore the last dsize-1 limbs, but this introduce
                // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                // noise is kept with respect to the ideal functionality.
                res.set_size(pmat.size() - ((dsize - di) as isize - 2).max(0) as usize);

                for j in 0..cols - 1 {
                    module.vec_znx_dft_apply(dsize, dsize - di - 1, &mut ai_dft, j, &a_conv, j);
                }

                if di == 0 {
                    module.vmp_apply_dft_to_dft(&mut res, &ai_dft, pmat, scratch_2);
                } else {
                    module.vmp_apply_dft_to_dft_add(&mut res, &ai_dft, pmat, di, scratch_2);
                }
            }
        }

        res.set_size(res.max_size());
    }

    let mut res_big: VecZnxBig<DR, BE> = module.vec_znx_idft_apply_consume(res);
    module.vec_znx_big_add_small_inplace(&mut res_big, 0, a.data(), 0);
    res_big
}
