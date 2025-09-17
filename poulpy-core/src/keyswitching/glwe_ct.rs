use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftAllocBytes, VecZnxDftApply, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, DataViewMut, Module, Scratch, VecZnx, VecZnxBig, VecZnxDft, VmpPMat, ZnxInfos},
};

use crate::layouts::{GLWECiphertext, Infos, prepared::GGLWESwitchingKeyPrepared};

impl GLWECiphertext<Vec<u8>> {
    #[allow(clippy::too_many_arguments)]
    pub fn keyswitch_scratch_space<B: Backend>(
        module: &Module<B>,
        basek_out: usize,
        k_out: usize,
        basek_in: usize,
        k_in: usize,
        basek_ksk: usize,
        k_ksk: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        let in_size: usize = k_in.div_ceil(basek_ksk).div_ceil(digits);
        let out_size: usize = k_out.div_ceil(basek_out);
        let ksk_size: usize = k_ksk.div_ceil(basek_ksk);
        let res_dft: usize = module.vec_znx_dft_alloc_bytes(rank_out + 1, ksk_size); // TODO OPTIMIZE
        let ai_dft: usize = module.vec_znx_dft_alloc_bytes(rank_in, in_size);
        let vmp: usize = module.vmp_apply_dft_to_dft_tmp_bytes(out_size, in_size, in_size, rank_in, rank_out + 1, ksk_size)
            + module.vec_znx_dft_alloc_bytes(rank_in, in_size);
        let normalize_big: usize = module.vec_znx_big_normalize_tmp_bytes();
        if basek_in == basek_ksk {
            res_dft + ((ai_dft + vmp) | normalize_big)
        } else {
            if digits == 1 {
                // In this case, we only need one column, temporary, that we can drop once a_dft is computed.
                let normalize_conv: usize = VecZnx::alloc_bytes(module.n(), 1, in_size) + module.vec_znx_normalize_tmp_bytes();
                res_dft + ((ai_dft + normalize_conv | vmp) | normalize_big)
            } else {
                // Since we stride over a to get a_dft when digits > 1, we need to store the full columns of a with in the base conversion.
                let normalize_conv: usize = VecZnx::alloc_bytes(module.n(), rank_in, in_size);
                res_dft + ((ai_dft + normalize_conv + (module.vec_znx_normalize_tmp_bytes() | vmp)) | normalize_big)
            }
        }
    }

    pub fn keyswitch_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        basek_out: usize,
        k_out: usize,
        basek_ksk: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        Self::keyswitch_scratch_space(
            module, basek_out, k_out, basek_out, k_out, basek_ksk, k_ksk, digits, rank, rank,
        )
    }
}

impl<DataSelf: DataRef> GLWECiphertext<DataSelf> {
    #[allow(dead_code)]
    pub(crate) fn assert_keyswitch<B: Backend, DataLhs, DataRhs>(
        &self,
        module: &Module<B>,
        glwe: &GLWECiphertext<DataLhs>,
        gglwe: &GGLWESwitchingKeyPrepared<DataRhs, B>,
        scratch: &Scratch<B>,
    ) where
        DataLhs: DataRef,
        DataRhs: DataRef,
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
        Scratch<B>: ScratchAvailable,
    {
        let basek_out: usize = self.basek();
        let basek_in: usize = glwe.basek();
        let basek_ksk: usize = gglwe.basek();

        assert_eq!(
            glwe.rank(),
            gglwe.rank_in(),
            "glwe.rank(): {} != gglwe.rank_in(): {}",
            glwe.rank(),
            gglwe.rank_in()
        );
        assert_eq!(
            self.rank(),
            gglwe.rank_out(),
            "self.rank(): {} != gglwe.rank_out(): {}",
            self.rank(),
            gglwe.rank_out()
        );
        assert_eq!(gglwe.n(), self.n());
        assert_eq!(glwe.n(), self.n());

        let scrach_needed = GLWECiphertext::keyswitch_scratch_space(
            module,
            basek_out,
            self.k(),
            basek_in,
            glwe.k(),
            basek_ksk,
            gglwe.k(),
            gglwe.digits(),
            gglwe.rank_in(),
            gglwe.rank_out(),
        );

        assert!(
            scratch.available() >= scrach_needed,
            "scratch.available()={} < GLWECiphertext::keyswitch_scratch_space(
                    module,
                    self.basek(),
                    self.k(),
                    glwe.basek(),
                    glwe.k(),
                    gglwe.basek(),
                    gglwe.k(),
                    gglwe.digits(),
                    gglwe.rank_in(),
                    gglwe.rank_out(),
                )={scrach_needed}",
            scratch.available(),
        );
    }

    #[allow(dead_code)]
    pub(crate) fn assert_keyswitch_inplace<B: Backend, DataRhs>(
        &self,
        module: &Module<B>,
        gglwe: &GGLWESwitchingKeyPrepared<DataRhs, B>,
        scratch: &Scratch<B>,
    ) where
        DataRhs: DataRef,
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
        Scratch<B>: ScratchAvailable,
    {
        let basek_out: usize = self.basek();
        let basek_in: usize = self.basek();
        let basek_ksk: usize = gglwe.basek();

        assert_eq!(
            self.rank(),
            gglwe.rank_out(),
            "self.rank(): {} != gglwe.rank_out(): {}",
            self.rank(),
            gglwe.rank_out()
        );

        assert_eq!(gglwe.n(), self.n());

        let scrach_needed = GLWECiphertext::keyswitch_scratch_space(
            module,
            basek_out,
            self.k(),
            basek_in,
            self.k(),
            basek_ksk,
            gglwe.k(),
            gglwe.digits(),
            gglwe.rank_in(),
            gglwe.rank_out(),
        );

        assert!(
            scratch.available() >= scrach_needed,
            "scratch.available()={} < GLWECiphertext::keyswitch_scratch_space(
                    module,
                    basek_out,
                    self.k(),
                    basek_in,
                    self.k(),
                    basek_ksk,
                    gglwe.k(),
                    gglwe.digits(),
                    gglwe.rank_in(),
                    gglwe.rank_out(),
                )={scrach_needed}",
            scratch.available(),
        );
    }
}

impl<DataSelf: DataMut> GLWECiphertext<DataSelf> {
    pub fn keyswitch<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        glwe_in: &GLWECiphertext<DataLhs>,
        gglwe: &GGLWESwitchingKeyPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxNormalize<B>
            + VecZnxNormalizeTmpBytes,
        Scratch<B>: ScratchAvailable + TakeVecZnxDft<B> + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            self.assert_keyswitch(module, glwe_in, gglwe, scratch);
        }

        let basek_out: usize = self.basek();
        let basek_ksk: usize = gglwe.basek();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self.n(), self.cols(), gglwe.size()); // Todo optimise
        let res_big: VecZnxBig<_, B> = glwe_in.keyswitch_internal(module, res_dft, gglwe, scratch_1);
        (0..self.cols()).for_each(|i| {
            module.vec_znx_big_normalize(
                basek_out,
                &mut self.data,
                i,
                basek_ksk,
                &res_big,
                i,
                scratch_1,
            );
        })
    }

    pub fn keyswitch_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        gglwe: &GGLWESwitchingKeyPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDftTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxNormalize<B>
            + VecZnxNormalizeTmpBytes,
        Scratch<B>: ScratchAvailable + TakeVecZnxDft<B> + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            self.assert_keyswitch_inplace(module, gglwe, scratch);
        }

        let basek_in: usize = self.basek();
        let basek_ksk: usize = gglwe.basek();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self.n(), self.cols(), gglwe.size()); // Todo optimise
        let res_big: VecZnxBig<_, B> = self.keyswitch_internal(module, res_dft, gglwe, scratch_1);
        (0..self.cols()).for_each(|i| {
            module.vec_znx_big_normalize(
                basek_in,
                &mut self.data,
                i,
                basek_ksk,
                &res_big,
                i,
                scratch_1,
            );
        })
    }
}

impl<D: DataRef> GLWECiphertext<D> {
    pub(crate) fn keyswitch_internal<B: Backend, DataRes, DataKey>(
        &self,
        module: &Module<B>,
        res_dft: VecZnxDft<DataRes, B>,
        gglwe: &GGLWESwitchingKeyPrepared<DataKey, B>,
        scratch: &mut Scratch<B>,
    ) -> VecZnxBig<DataRes, B>
    where
        DataRes: DataMut,
        DataKey: DataRef,
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDftTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + TakeVecZnx,
    {
        if gglwe.digits() == 1 {
            return keyswitch_vmp_one_digit(
                module,
                self.basek(),
                gglwe.basek(),
                res_dft,
                &self.data,
                &gglwe.key.data,
                scratch,
            );
        }

        keyswitch_vmp_multiple_digits(
            module,
            self.basek(),
            gglwe.basek(),
            res_dft,
            &self.data,
            &gglwe.key.data,
            gglwe.digits(),
            scratch,
        )
    }
}

fn keyswitch_vmp_one_digit<B: Backend, DataRes, DataIn, DataVmp>(
    module: &Module<B>,
    basek_in: usize,
    basek_ksk: usize,
    mut res_dft: VecZnxDft<DataRes, B>,
    a: &VecZnx<DataIn>,
    mat: &VmpPMat<DataVmp, B>,
    scratch: &mut Scratch<B>,
) -> VecZnxBig<DataRes, B>
where
    DataRes: DataMut,
    DataIn: DataRef,
    DataVmp: DataRef,
    Module<B>: VecZnxDftAllocBytes
        + VecZnxDftApply<B>
        + VmpApplyDftToDft<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxNormalize<B>,
    Scratch<B>: TakeVecZnxDft<B> + TakeVecZnx,
{
    let cols: usize = a.cols();

    let a_size: usize = (a.size() * basek_in).div_ceil(basek_ksk);
    let (mut ai_dft, scratch_1) = scratch.take_vec_znx_dft(a.n(), cols - 1, a.size());

    if basek_in == basek_ksk {
        (0..cols - 1).for_each(|col_i| {
            module.vec_znx_dft_apply(1, 0, &mut ai_dft, col_i, a, col_i + 1);
        });
    } else {
        let (mut a_conv, scratch_2) = scratch_1.take_vec_znx(a.n(), 1, a_size);
        (0..cols - 1).for_each(|col_i| {
            module.vec_znx_normalize(basek_ksk, &mut a_conv, 0, basek_in, a, col_i + 1, scratch_2);
            module.vec_znx_dft_apply(1, 0, &mut ai_dft, col_i, &a_conv, 0);
        });
    }

    module.vmp_apply_dft_to_dft(&mut res_dft, &ai_dft, mat, scratch_1);
    let mut res_big: VecZnxBig<DataRes, B> = module.vec_znx_idft_apply_consume(res_dft);
    module.vec_znx_big_add_small_inplace(&mut res_big, 0, a, 0);
    res_big
}

fn keyswitch_vmp_multiple_digits<B: Backend, DataRes, DataIn, DataVmp>(
    module: &Module<B>,
    basek_in: usize,
    basek_ksk: usize,
    mut res_dft: VecZnxDft<DataRes, B>,
    a: &VecZnx<DataIn>,
    mat: &VmpPMat<DataVmp, B>,
    digits: usize,
    scratch: &mut Scratch<B>,
) -> VecZnxBig<DataRes, B>
where
    DataRes: DataMut,
    DataIn: DataRef,
    DataVmp: DataRef,
    Module<B>: VecZnxDftAllocBytes
        + VecZnxDftApply<B>
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxNormalize<B>,
    Scratch<B>: TakeVecZnxDft<B> + TakeVecZnx,
{
    let cols: usize = a.cols();
    let a_size: usize = (a.size() * basek_in).div_ceil(basek_ksk);
    let (mut ai_dft, scratch_1) = scratch.take_vec_znx_dft(a.n(), cols - 1, a_size.div_ceil(digits));
    ai_dft.data_mut().fill(0);

    if basek_in == basek_ksk {
        for di in 0..digits {
            ai_dft.set_size((a_size + di) / digits);

            // Small optimization for digits > 2
            // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
            // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(digits-1) * B}.
            // As such we can ignore the last digits-2 limbs safely of the sum of vmp products.
            // It is possible to further ignore the last digits-1 limbs, but this introduce
            // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
            // noise is kept with respect to the ideal functionality.
            res_dft.set_size(mat.size() - ((digits - di) as isize - 2).max(0) as usize);

            for j in 0..cols - 1 {
                module.vec_znx_dft_apply(digits, digits - di - 1, &mut ai_dft, j, a, j + 1);
            }

            if di == 0 {
                module.vmp_apply_dft_to_dft(&mut res_dft, &ai_dft, mat, scratch_1);
            } else {
                module.vmp_apply_dft_to_dft_add(&mut res_dft, &ai_dft, mat, di, scratch_1);
            }
        }
    } else {
        let (mut a_conv, scratch_2) = scratch_1.take_vec_znx(a.n(), cols - 1, a_size);
        for j in 0..cols - 1 {
            module.vec_znx_normalize(basek_ksk, &mut a_conv, j, basek_in, a, j + 1, scratch_2);
        }

        for di in 0..digits {
            ai_dft.set_size((a_size + di) / digits);

            // Small optimization for digits > 2
            // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
            // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(digits-1) * B}.
            // As such we can ignore the last digits-2 limbs safely of the sum of vmp products.
            // It is possible to further ignore the last digits-1 limbs, but this introduce
            // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
            // noise is kept with respect to the ideal functionality.
            res_dft.set_size(mat.size() - ((digits - di) as isize - 2).max(0) as usize);

            for j in 0..cols - 1 {
                module.vec_znx_dft_apply(digits, digits - di - 1, &mut ai_dft, j, &a_conv, j);
            }

            if di == 0 {
                module.vmp_apply_dft_to_dft(&mut res_dft, &ai_dft, mat, scratch_2);
            } else {
                module.vmp_apply_dft_to_dft_add(&mut res_dft, &ai_dft, mat, di, scratch_2);
            }
        }
    }

    res_dft.set_size(res_dft.max_size());
    let mut res_big: VecZnxBig<DataRes, B> = module.vec_znx_idft_apply_consume(res_dft);
    module.vec_znx_big_add_small_inplace(&mut res_big, 0, a, 0);
    res_big
}
