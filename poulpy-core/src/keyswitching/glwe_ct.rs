use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnxDft, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftAllocBytes, VecZnxDftApply, VecZnxIdftApplyConsume, VmpApplyDftToDft, VmpApplyDftToDftAdd,
        VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, DataViewMut, Module, Scratch, VecZnx, VecZnxBig, VecZnxDft, VmpPMat, ZnxInfos},
};

use crate::layouts::{GLWECiphertext, Infos, prepared::GGLWESwitchingKeyPrepared};

impl GLWECiphertext<Vec<u8>> {
    #[allow(clippy::too_many_arguments)]
    pub fn keyswitch_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_in: usize,
        k_ksk: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes,
    {
        let in_size: usize = k_in.div_ceil(basek).div_ceil(digits);
        let out_size: usize = k_out.div_ceil(basek);
        let ksk_size: usize = k_ksk.div_ceil(basek);
        let res_dft: usize = module.vec_znx_dft_alloc_bytes(rank_out + 1, ksk_size); // TODO OPTIMIZE
        let ai_dft: usize = module.vec_znx_dft_alloc_bytes(rank_in, in_size);
        let vmp: usize = module.vmp_apply_dft_to_dft_tmp_bytes(out_size, in_size, in_size, rank_in, rank_out + 1, ksk_size)
            + module.vec_znx_dft_alloc_bytes(rank_in, in_size);
        let normalize: usize = module.vec_znx_big_normalize_tmp_bytes();
        res_dft + ((ai_dft + vmp) | normalize)
    }

    pub fn keyswitch_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes,
    {
        Self::keyswitch_scratch_space(module, basek, k_out, k_out, k_ksk, digits, rank, rank)
    }
}

impl<DataSelf: DataRef> GLWECiphertext<DataSelf> {
    #[allow(dead_code)]
    pub(crate) fn assert_keyswitch<B: Backend, DataLhs, DataRhs>(
        &self,
        module: &Module<B>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &GGLWESwitchingKeyPrepared<DataRhs, B>,
        scratch: &Scratch<B>,
    ) where
        DataLhs: DataRef,
        DataRhs: DataRef,
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes,
        Scratch<B>: ScratchAvailable,
    {
        let basek: usize = self.basek();
        assert_eq!(
            lhs.rank(),
            rhs.rank_in(),
            "lhs.rank(): {} != rhs.rank_in(): {}",
            lhs.rank(),
            rhs.rank_in()
        );
        assert_eq!(
            self.rank(),
            rhs.rank_out(),
            "self.rank(): {} != rhs.rank_out(): {}",
            self.rank(),
            rhs.rank_out()
        );
        assert_eq!(self.basek(), basek);
        assert_eq!(lhs.basek(), basek);
        assert_eq!(rhs.n(), self.n());
        assert_eq!(lhs.n(), self.n());
        assert!(
            scratch.available()
                >= GLWECiphertext::keyswitch_scratch_space(
                    module,
                    self.basek(),
                    self.k(),
                    lhs.k(),
                    rhs.k(),
                    rhs.digits(),
                    rhs.rank_in(),
                    rhs.rank_out(),
                ),
            "scratch.available()={} < GLWECiphertext::keyswitch_scratch_space(
                    module,
                    self.basek(),
                    self.k(),
                    lhs.k(),
                    rhs.k(),
                    rhs.digits(),
                    rhs.rank_in(),
                    rhs.rank_out(),
                )={}",
            scratch.available(),
            GLWECiphertext::keyswitch_scratch_space(
                module,
                self.basek(),
                self.k(),
                lhs.k(),
                rhs.k(),
                rhs.digits(),
                rhs.rank_in(),
                rhs.rank_out(),
            )
        );
    }

    #[allow(dead_code)]
    pub(crate) fn assert_keyswitch_inplace<B: Backend, DataRhs>(
        &self,
        module: &Module<B>,
        rhs: &GGLWESwitchingKeyPrepared<DataRhs, B>,
        scratch: &Scratch<B>,
    ) where
        DataRhs: DataRef,
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes,
        Scratch<B>: ScratchAvailable,
    {
        let basek: usize = self.basek();
        assert_eq!(
            self.rank(),
            rhs.rank_out(),
            "self.rank(): {} != rhs.rank_out(): {}",
            self.rank(),
            rhs.rank_out()
        );
        assert_eq!(self.basek(), basek);
        assert_eq!(rhs.n(), self.n());
        assert!(
            scratch.available()
                >= GLWECiphertext::keyswitch_scratch_space(
                    module,
                    self.basek(),
                    self.k(),
                    self.k(),
                    rhs.k(),
                    rhs.digits(),
                    rhs.rank_in(),
                    rhs.rank_out(),
                ),
            "scratch.available()={} < GLWECiphertext::keyswitch_scratch_space(
                    module,
                    self.basek(),
                    self.k(),
                    self.k(),
                    rhs.k(),
                    rhs.digits(),
                    rhs.rank_in(),
                    rhs.rank_out(),
                )={}",
            scratch.available(),
            GLWECiphertext::keyswitch_scratch_space(
                module,
                self.basek(),
                self.k(),
                self.k(),
                rhs.k(),
                rhs.digits(),
                rhs.rank_in(),
                rhs.rank_out(),
            )
        );
    }
}

impl<DataSelf: DataMut> GLWECiphertext<DataSelf> {
    pub fn keyswitch<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &GGLWESwitchingKeyPrepared<DataRhs, B>,
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
            + VecZnxBigNormalize<B>,
        Scratch<B>: ScratchAvailable + TakeVecZnxDft<B>,
    {
        #[cfg(debug_assertions)]
        {
            self.assert_keyswitch(module, lhs, rhs, scratch);
        }
        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self.n(), self.cols(), rhs.size()); // Todo optimise
        let res_big: VecZnxBig<_, B> = lhs.keyswitch_internal(module, res_dft, rhs, scratch_1);
        (0..self.cols()).for_each(|i| {
            module.vec_znx_big_normalize(self.basek(), &mut self.data, i, &res_big, i, scratch_1);
        })
    }

    pub fn keyswitch_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGLWESwitchingKeyPrepared<DataRhs, B>,
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
            + VecZnxBigNormalize<B>,
        Scratch<B>: ScratchAvailable + TakeVecZnxDft<B>,
    {
        #[cfg(debug_assertions)]
        {
            self.assert_keyswitch_inplace(module, rhs, scratch);
        }
        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self.n(), self.cols(), rhs.size()); // Todo optimise
        let res_big: VecZnxBig<_, B> = self.keyswitch_internal(module, res_dft, rhs, scratch_1);
        (0..self.cols()).for_each(|i| {
            module.vec_znx_big_normalize(self.basek(), &mut self.data, i, &res_big, i, scratch_1);
        })
    }
}

impl<D: DataRef> GLWECiphertext<D> {
    pub(crate) fn keyswitch_internal<B: Backend, DataRes, DataKey>(
        &self,
        module: &Module<B>,
        res_dft: VecZnxDft<DataRes, B>,
        rhs: &GGLWESwitchingKeyPrepared<DataKey, B>,
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
            + VecZnxBigNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B>,
    {
        if rhs.digits() == 1 {
            return keyswitch_vmp_one_digit(module, res_dft, &self.data, &rhs.key.data, scratch);
        }

        keyswitch_vmp_multiple_digits(
            module,
            res_dft,
            &self.data,
            &rhs.key.data,
            rhs.digits(),
            scratch,
        )
    }
}

fn keyswitch_vmp_one_digit<B: Backend, DataRes, DataIn, DataVmp>(
    module: &Module<B>,
    mut res_dft: VecZnxDft<DataRes, B>,
    a: &VecZnx<DataIn>,
    mat: &VmpPMat<DataVmp, B>,
    scratch: &mut Scratch<B>,
) -> VecZnxBig<DataRes, B>
where
    DataRes: DataMut,
    DataIn: DataRef,
    DataVmp: DataRef,
    Module<B>:
        VecZnxDftAllocBytes + VecZnxDftApply<B> + VmpApplyDftToDft<B> + VecZnxIdftApplyConsume<B> + VecZnxBigAddSmallInplace<B>,
    Scratch<B>: TakeVecZnxDft<B>,
{
    let cols: usize = a.cols();
    let (mut ai_dft, scratch_1) = scratch.take_vec_znx_dft(a.n(), cols - 1, a.size());
    (0..cols - 1).for_each(|col_i| {
        module.vec_znx_dft_apply(1, 0, &mut ai_dft, col_i, a, col_i + 1);
    });
    module.vmp_apply_dft_to_dft(&mut res_dft, &ai_dft, mat, scratch_1);
    let mut res_big: VecZnxBig<DataRes, B> = module.vec_znx_idft_apply_consume(res_dft);
    module.vec_znx_big_add_small_inplace(&mut res_big, 0, a, 0);
    res_big
}

fn keyswitch_vmp_multiple_digits<B: Backend, DataRes, DataIn, DataVmp>(
    module: &Module<B>,
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
        + VecZnxBigAddSmallInplace<B>,
    Scratch<B>: TakeVecZnxDft<B>,
{
    let cols: usize = a.cols();
    let size: usize = a.size();
    let (mut ai_dft, scratch_1) = scratch.take_vec_znx_dft(a.n(), cols - 1, size.div_ceil(digits));

    ai_dft.data_mut().fill(0);

    (0..digits).for_each(|di| {
        ai_dft.set_size((size + di) / digits);

        // Small optimization for digits > 2
        // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
        // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(digits-1) * B}.
        // As such we can ignore the last digits-2 limbs safely of the sum of vmp products.
        // It is possible to further ignore the last digits-1 limbs, but this introduce
        // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
        // noise is kept with respect to the ideal functionality.
        res_dft.set_size(mat.size() - ((digits - di) as isize - 2).max(0) as usize);

        (0..cols - 1).for_each(|col_i| {
            module.vec_znx_dft_apply(digits, digits - di - 1, &mut ai_dft, col_i, a, col_i + 1);
        });

        if di == 0 {
            module.vmp_apply_dft_to_dft(&mut res_dft, &ai_dft, mat, scratch_1);
        } else {
            module.vmp_apply_dft_to_dft_add(&mut res_dft, &ai_dft, mat, di, scratch_1);
        }
    });

    res_dft.set_size(res_dft.max_size());
    let mut res_big: VecZnxBig<DataRes, B> = module.vec_znx_idft_apply_consume(res_dft);
    module.vec_znx_big_add_small_inplace(&mut res_big, 0, a, 0);
    res_big
}
