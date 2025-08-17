use poulpy_backend::hal::{
    api::{
        ScratchAvailable, TakeVecZnxDft, VecZnxAutomorphismInplace, VecZnxBigAddSmallInplace, VecZnxBigAutomorphismInplace,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallAInplace, VecZnxBigSubSmallBInplace,
        VecZnxDftAllocBytes, VecZnxDftFromVecZnx, VecZnxDftToVecZnxBigConsume, VmpApply, VmpApplyAdd, VmpApplyTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, VecZnxBig},
};

use crate::layouts::{GLWECiphertext, Infos, prepared::GGLWEAutomorphismKeyPrepared};

impl GLWECiphertext<Vec<u8>> {
    #[allow(clippy::too_many_arguments)]
    pub fn automorphism_scratch_space<B: Backend>(
        module: &Module<B>,
        n: usize,
        basek: usize,
        k_out: usize,
        k_in: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyTmpBytes + VecZnxBigNormalizeTmpBytes,
    {
        Self::keyswitch_scratch_space(module, n, basek, k_out, k_in, k_ksk, digits, rank, rank)
    }

    pub fn automorphism_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        n: usize,
        basek: usize,
        k_out: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyTmpBytes + VecZnxBigNormalizeTmpBytes,
    {
        Self::keyswitch_inplace_scratch_space(module, n, basek, k_out, k_ksk, digits, rank)
    }
}

impl<DataSelf: DataMut> GLWECiphertext<DataSelf> {
    pub fn automorphism<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &GGLWEAutomorphismKeyPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxAutomorphismInplace,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        self.keyswitch(module, lhs, &rhs.key, scratch);
        (0..self.rank() + 1).for_each(|i| {
            module.vec_znx_automorphism_inplace(rhs.p(), &mut self.data, i);
        })
    }

    pub fn automorphism_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGLWEAutomorphismKeyPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxAutomorphismInplace,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        self.keyswitch_inplace(module, &rhs.key, scratch);
        (0..self.rank() + 1).for_each(|i| {
            module.vec_znx_automorphism_inplace(rhs.p(), &mut self.data, i);
        })
    }

    pub fn automorphism_add<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &GGLWEAutomorphismKeyPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxBigAutomorphismInplace<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        #[cfg(debug_assertions)]
        {
            self.assert_keyswitch(module, lhs, &rhs.key, scratch);
        }
        let (res_dft, scratch1) = scratch.take_vec_znx_dft(self.n(), self.cols(), rhs.size()); // TODO: optimise size
        let mut res_big: VecZnxBig<_, B> = lhs.keyswitch_internal(module, res_dft, &rhs.key, scratch1);
        (0..self.cols()).for_each(|i| {
            module.vec_znx_big_automorphism_inplace(rhs.p(), &mut res_big, i);
            module.vec_znx_big_add_small_inplace(&mut res_big, i, &lhs.data, i);
            module.vec_znx_big_normalize(self.basek(), &mut self.data, i, &res_big, i, scratch1);
        })
    }

    pub fn automorphism_add_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGLWEAutomorphismKeyPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxBigAutomorphismInplace<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        unsafe {
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            self.automorphism_add(module, &*self_ptr, rhs, scratch);
        }
    }

    pub fn automorphism_sub_ab<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &GGLWEAutomorphismKeyPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxBigSubSmallAInplace<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        #[cfg(debug_assertions)]
        {
            self.assert_keyswitch(module, lhs, &rhs.key, scratch);
        }
        let (res_dft, scratch1) = scratch.take_vec_znx_dft(self.n(), self.cols(), rhs.size()); // TODO: optimise size
        let mut res_big: VecZnxBig<_, B> = lhs.keyswitch_internal(module, res_dft, &rhs.key, scratch1);
        (0..self.cols()).for_each(|i| {
            module.vec_znx_big_automorphism_inplace(rhs.p(), &mut res_big, i);
            module.vec_znx_big_sub_small_a_inplace(&mut res_big, i, &lhs.data, i);
            module.vec_znx_big_normalize(self.basek(), &mut self.data, i, &res_big, i, scratch1);
        })
    }

    pub fn automorphism_sub_ab_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGLWEAutomorphismKeyPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxBigSubSmallAInplace<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        unsafe {
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            self.automorphism_sub_ab(module, &*self_ptr, rhs, scratch);
        }
    }

    pub fn automorphism_sub_ba<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &GGLWEAutomorphismKeyPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxBigSubSmallBInplace<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        #[cfg(debug_assertions)]
        {
            self.assert_keyswitch(module, lhs, &rhs.key, scratch);
        }
        let (res_dft, scratch1) = scratch.take_vec_znx_dft(self.n(), self.cols(), rhs.size()); // TODO: optimise size
        let mut res_big: VecZnxBig<_, B> = lhs.keyswitch_internal(module, res_dft, &rhs.key, scratch1);
        (0..self.cols()).for_each(|i| {
            module.vec_znx_big_automorphism_inplace(rhs.p(), &mut res_big, i);
            module.vec_znx_big_sub_small_b_inplace(&mut res_big, i, &lhs.data, i);
            module.vec_znx_big_normalize(self.basek(), &mut self.data, i, &res_big, i, scratch1);
        })
    }

    pub fn automorphism_sub_ba_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGLWEAutomorphismKeyPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxBigSubSmallBInplace<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        unsafe {
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            self.automorphism_sub_ba(module, &*self_ptr, rhs, scratch);
        }
    }
}
