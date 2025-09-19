use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxAutomorphismInplace, VecZnxBigAddSmallInplace,
        VecZnxBigAutomorphismInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallAInplace,
        VecZnxBigSubSmallBInplace, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxIdftApplyConsume, VecZnxNormalize,
        VecZnxNormalizeTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, VecZnxBig},
};

use crate::layouts::{GGLWELayoutInfos, GLWECiphertext, GLWEInfos, LWEInfos, prepared::GGLWEAutomorphismKeyPrepared};

impl GLWECiphertext<Vec<u8>> {
    pub fn automorphism_scratch_space<B: Backend, OUT, IN, KEY>(
        module: &Module<B>,
        out_infos: &OUT,
        in_infos: &IN,
        key_infos: &KEY,
    ) -> usize
    where
        OUT: GLWEInfos,
        IN: GLWEInfos,
        KEY: GGLWELayoutInfos,
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        Self::keyswitch_scratch_space(module, out_infos, in_infos, key_infos)
    }

    pub fn automorphism_inplace_scratch_space<B: Backend, OUT, KEY>(module: &Module<B>, out_infos: &OUT, key_infos: &KEY) -> usize
    where
        OUT: GLWEInfos,
        KEY: GGLWELayoutInfos,
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        Self::keyswitch_inplace_scratch_space(module, out_infos, key_infos)
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
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxAutomorphismInplace<B>
            + VecZnxNormalize<B>
            + VecZnxNormalizeTmpBytes,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        self.keyswitch(module, lhs, &rhs.key, scratch);
        (0..(self.rank() + 1).into()).for_each(|i| {
            module.vec_znx_automorphism_inplace(rhs.p(), &mut self.data, i, scratch);
        })
    }

    pub fn automorphism_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGLWEAutomorphismKeyPrepared<DataRhs, B>,
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
            + VecZnxAutomorphismInplace<B>
            + VecZnxNormalize<B>
            + VecZnxNormalizeTmpBytes,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        self.keyswitch_inplace(module, &rhs.key, scratch);
        (0..(self.rank() + 1).into()).for_each(|i| {
            module.vec_znx_automorphism_inplace(rhs.p(), &mut self.data, i, scratch);
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
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            self.assert_keyswitch(module, lhs, &rhs.key, scratch);
        }
        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self.n().into(), (self.rank() + 1).into(), rhs.size()); // TODO: optimise size
        let mut res_big: VecZnxBig<_, B> = lhs.keyswitch_internal(module, res_dft, &rhs.key, scratch_1);
        (0..(self.rank() + 1).into()).for_each(|i| {
            module.vec_znx_big_automorphism_inplace(rhs.p(), &mut res_big, i, scratch_1);
            module.vec_znx_big_add_small_inplace(&mut res_big, i, &lhs.data, i);
            module.vec_znx_big_normalize(
                self.base2k().into(),
                &mut self.data,
                i,
                rhs.base2k().into(),
                &res_big,
                i,
                scratch_1,
            );
        })
    }

    pub fn automorphism_add_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGLWEAutomorphismKeyPrepared<DataRhs, B>,
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
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            self.assert_keyswitch_inplace(module, &rhs.key, scratch);
        }
        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self.n().into(), (self.rank() + 1).into(), rhs.size()); // TODO: optimise size
        let mut res_big: VecZnxBig<_, B> = self.keyswitch_internal(module, res_dft, &rhs.key, scratch_1);
        (0..(self.rank() + 1).into()).for_each(|i| {
            module.vec_znx_big_automorphism_inplace(rhs.p(), &mut res_big, i, scratch_1);
            module.vec_znx_big_add_small_inplace(&mut res_big, i, &self.data, i);
            module.vec_znx_big_normalize(
                self.base2k().into(),
                &mut self.data,
                i,
                rhs.base2k().into(),
                &res_big,
                i,
                scratch_1,
            );
        })
    }

    pub fn automorphism_sub_ab<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &GGLWEAutomorphismKeyPrepared<DataRhs, B>,
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
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxBigSubSmallAInplace<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            self.assert_keyswitch(module, lhs, &rhs.key, scratch);
        }
        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self.n().into(), (self.rank() + 1).into(), rhs.size()); // TODO: optimise size
        let mut res_big: VecZnxBig<_, B> = lhs.keyswitch_internal(module, res_dft, &rhs.key, scratch_1);
        (0..(self.rank() + 1).into()).for_each(|i| {
            module.vec_znx_big_automorphism_inplace(rhs.p(), &mut res_big, i, scratch_1);
            module.vec_znx_big_sub_small_a_inplace(&mut res_big, i, &lhs.data, i);
            module.vec_znx_big_normalize(
                self.base2k().into(),
                &mut self.data,
                i,
                rhs.base2k().into(),
                &res_big,
                i,
                scratch_1,
            );
        })
    }

    pub fn automorphism_sub_ab_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGLWEAutomorphismKeyPrepared<DataRhs, B>,
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
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxBigSubSmallAInplace<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            self.assert_keyswitch_inplace(module, &rhs.key, scratch);
        }
        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self.n().into(), (self.rank() + 1).into(), rhs.size()); // TODO: optimise size
        let mut res_big: VecZnxBig<_, B> = self.keyswitch_internal(module, res_dft, &rhs.key, scratch_1);
        (0..(self.rank() + 1).into()).for_each(|i| {
            module.vec_znx_big_automorphism_inplace(rhs.p(), &mut res_big, i, scratch_1);
            module.vec_znx_big_sub_small_a_inplace(&mut res_big, i, &self.data, i);
            module.vec_znx_big_normalize(
                self.base2k().into(),
                &mut self.data,
                i,
                rhs.base2k().into(),
                &res_big,
                i,
                scratch_1,
            );
        })
    }

    pub fn automorphism_sub_ba<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &GGLWEAutomorphismKeyPrepared<DataRhs, B>,
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
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxBigSubSmallBInplace<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            self.assert_keyswitch(module, lhs, &rhs.key, scratch);
        }
        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self.n().into(), (self.rank() + 1).into(), rhs.size()); // TODO: optimise size
        let mut res_big: VecZnxBig<_, B> = lhs.keyswitch_internal(module, res_dft, &rhs.key, scratch_1);
        (0..(self.rank() + 1).into()).for_each(|i| {
            module.vec_znx_big_automorphism_inplace(rhs.p(), &mut res_big, i, scratch_1);
            module.vec_znx_big_sub_small_b_inplace(&mut res_big, i, &lhs.data, i);
            module.vec_znx_big_normalize(
                self.base2k().into(),
                &mut self.data,
                i,
                rhs.base2k().into(),
                &res_big,
                i,
                scratch_1,
            );
        })
    }

    pub fn automorphism_sub_ba_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGLWEAutomorphismKeyPrepared<DataRhs, B>,
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
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxBigSubSmallBInplace<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            self.assert_keyswitch_inplace(module, &rhs.key, scratch);
        }
        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self.n().into(), (self.rank() + 1).into(), rhs.size()); // TODO: optimise size
        let mut res_big: VecZnxBig<_, B> = self.keyswitch_internal(module, res_dft, &rhs.key, scratch_1);
        (0..(self.rank() + 1).into()).for_each(|i| {
            module.vec_znx_big_automorphism_inplace(rhs.p(), &mut res_big, i, scratch_1);
            module.vec_znx_big_sub_small_b_inplace(&mut res_big, i, &self.data, i);
            module.vec_znx_big_normalize(
                self.base2k().into(),
                &mut self.data,
                i,
                rhs.base2k().into(),
                &res_big,
                i,
                scratch_1,
            );
        })
    }
}
