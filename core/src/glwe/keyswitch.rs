use backend::hal::{
    api::{
        DataViewMut, ScratchAvailable, TakeVecZnxDft, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftAllocBytes, VecZnxDftFromVecZnx, VecZnxDftToVecZnxBigConsume, VmpApply, VmpApplyAdd, VmpApplyTmpBytes, ZnxInfos,
    },
    layouts::{Backend, Module, Scratch, VecZnx, VecZnxBig, VecZnxDft, VmpPMat},
};

use crate::{GLWECiphertext, GLWESwitchingKeyExec, Infos};

pub trait GLWEKeyswitchFamily<B: Backend> = VecZnxDftAllocBytes
    + VmpApplyTmpBytes
    + VecZnxBigNormalizeTmpBytes
    + VmpApplyTmpBytes
    + VmpApply<B>
    + VmpApplyAdd<B>
    + VecZnxDftFromVecZnx<B>
    + VecZnxDftToVecZnxBigConsume<B>
    + VecZnxBigAddSmallInplace<B>
    + VecZnxBigNormalize<B>;

impl GLWECiphertext<Vec<u8>> {
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
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        let in_size: usize = k_in.div_ceil(basek).div_ceil(digits);
        let out_size: usize = k_out.div_ceil(basek);
        let ksk_size: usize = k_ksk.div_ceil(basek);
        let res_dft: usize = module.vec_znx_dft_alloc_bytes(rank_out + 1, ksk_size); // TODO OPTIMIZE
        let ai_dft: usize = module.vec_znx_dft_alloc_bytes(rank_in, in_size);
        let vmp: usize = module.vmp_apply_tmp_bytes(out_size, in_size, in_size, rank_in, rank_out + 1, ksk_size)
            + module.vec_znx_dft_alloc_bytes(rank_in, in_size);
        let normalize: usize = module.vec_znx_big_normalize_tmp_bytes(module.n());
        return res_dft + ((ai_dft + vmp) | normalize);
    }

    pub fn keyswitch_from_fourier_scratch_space<B: Backend>(
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
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        Self::keyswitch_scratch_space(module, basek, k_out, k_in, k_ksk, digits, rank_in, rank_out)
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
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        Self::keyswitch_scratch_space(module, basek, k_out, k_out, k_ksk, digits, rank, rank)
    }
}

impl<DataSelf: AsRef<[u8]>> GLWECiphertext<DataSelf> {
    pub(crate) fn assert_keyswitch<B: Backend, DataLhs, DataRhs>(
        &self,
        module: &Module<B>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &GLWESwitchingKeyExec<DataRhs, B>,
        scratch: &Scratch<B>,
    ) where
        Module<B>: GLWEKeyswitchFamily<B>,
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
        assert_eq!(rhs.n(), module.n());
        assert_eq!(self.n(), module.n());
        assert_eq!(lhs.n(), module.n());
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
                )
        );
    }
}

impl<DataSelf: AsRef<[u8]> + AsMut<[u8]>> GLWECiphertext<DataSelf> {
    pub fn keyswitch<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &GLWESwitchingKeyExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEKeyswitchFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        #[cfg(debug_assertions)]
        {
            self.assert_keyswitch(module, lhs, rhs, scratch);
        }
        let (res_dft, scratch1) = scratch.take_vec_znx_dft(module, self.cols(), rhs.size()); // Todo optimise
        let res_big: VecZnxBig<_, B> = keyswitch(module, res_dft, lhs, rhs, scratch1);
        (0..self.cols()).for_each(|i| {
            module.vec_znx_big_normalize(self.basek(), &mut self.data, i, &res_big, i, scratch1);
        })
    }

    pub fn keyswitch_inplace<DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GLWESwitchingKeyExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEKeyswitchFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        unsafe {
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            self.keyswitch(&module, &*self_ptr, rhs, scratch);
        }
    }
}

pub(crate) fn keyswitch<B: Backend, DataRes, DataIn, DataKey>(
    module: &Module<B>,
    res_dft: VecZnxDft<DataRes, B>,
    lhs: &GLWECiphertext<DataIn>,
    rhs: &GLWESwitchingKeyExec<DataKey, B>,
    scratch: &mut Scratch<B>,
) -> VecZnxBig<DataRes, B>
where
    DataRes: AsRef<[u8]> + AsMut<[u8]>,
    DataIn: AsRef<[u8]>,
    DataKey: AsRef<[u8]>,
    Module<B>: GLWEKeyswitchFamily<B>,
    Scratch<B>: TakeVecZnxDft<B>,
{
    if rhs.digits() == 1 {
        return keyswitch_vmp_one_digit(module, res_dft, &lhs.data, &rhs.key.data, scratch);
    }

    keyswitch_vmp_multiple_digits(
        module,
        res_dft,
        &lhs.data,
        &rhs.key.data,
        rhs.digits(),
        scratch,
    )
}

fn keyswitch_vmp_one_digit<B: Backend, DataRes, DataIn, DataVmp>(
    module: &Module<B>,
    mut res_dft: VecZnxDft<DataRes, B>,
    a: &VecZnx<DataIn>,
    mat: &VmpPMat<DataVmp, B>,
    scratch: &mut Scratch<B>,
) -> VecZnxBig<DataRes, B>
where
    DataRes: AsRef<[u8]> + AsMut<[u8]>,
    DataIn: AsRef<[u8]>,
    DataVmp: AsRef<[u8]>,
    Module<B>:
        VecZnxDftAllocBytes + VecZnxDftFromVecZnx<B> + VmpApply<B> + VecZnxDftToVecZnxBigConsume<B> + VecZnxBigAddSmallInplace<B>,
    Scratch<B>: TakeVecZnxDft<B>,
{
    let cols: usize = a.cols();
    let (mut ai_dft, scratch1) = scratch.take_vec_znx_dft(module, cols - 1, a.size());
    (0..cols - 1).for_each(|col_i| {
        module.vec_znx_dft_from_vec_znx(1, 0, &mut ai_dft, col_i, a, col_i + 1);
    });
    module.vmp_apply(&mut res_dft, &ai_dft, mat, scratch1);
    let mut res_big: VecZnxBig<DataRes, B> = module.vec_znx_dft_to_vec_znx_big_consume(res_dft);
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
    DataRes: AsRef<[u8]> + AsMut<[u8]>,
    DataIn: AsRef<[u8]>,
    DataVmp: AsRef<[u8]>,
    Module<B>: VecZnxDftAllocBytes
        + VecZnxDftFromVecZnx<B>
        + VmpApply<B>
        + VmpApplyAdd<B>
        + VecZnxDftToVecZnxBigConsume<B>
        + VecZnxBigAddSmallInplace<B>,
    Scratch<B>: TakeVecZnxDft<B>,
{
    let cols: usize = a.cols();
    let size: usize = a.size();
    let (mut ai_dft, scratch1) = scratch.take_vec_znx_dft(module, cols - 1, (size + digits - 1) / digits);

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
            module.vec_znx_dft_from_vec_znx(digits, digits - di - 1, &mut ai_dft, col_i, a, col_i + 1);
        });

        if di == 0 {
            module.vmp_apply(&mut res_dft, &ai_dft, mat, scratch1);
        } else {
            module.vmp_apply_add(&mut res_dft, &ai_dft, mat, di, scratch1);
        }
    });

    res_dft.set_size(res_dft.max_size());
    let mut res_big: VecZnxBig<DataRes, B> = module.vec_znx_dft_to_vec_znx_big_consume(res_dft);
    module.vec_znx_big_add_small_inplace(&mut res_big, 0, a, 0);
    res_big
}
