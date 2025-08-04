use backend::hal::{
    api::{
        ScratchAvailable, TakeVecZnxDft, VecZnxAutomorphismInplace, VecZnxBigAutomorphismInplace, VecZnxBigSubSmallAInplace,
        VecZnxBigSubSmallBInplace,
    },
    layouts::{Backend, Module, Scratch, VecZnxBig},
};

use crate::{AutomorphismKeyExec, GLWECiphertext, GLWEKeyswitchFamily, Infos, glwe::keyswitch::keyswitch};

impl GLWECiphertext<Vec<u8>> {
    pub fn automorphism_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_in: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        Self::keyswitch_scratch_space(module, basek, k_out, k_in, k_ksk, digits, rank, rank)
    }

    pub fn automorphism_inplace_scratch_space<B: Backend>(
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
        Self::keyswitch_inplace_scratch_space(module, basek, k_out, k_ksk, digits, rank)
    }
}

impl<DataSelf: AsRef<[u8]> + AsMut<[u8]>> GLWECiphertext<DataSelf> {
    pub fn automorphism<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &AutomorphismKeyExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEKeyswitchFamily<B> + VecZnxAutomorphismInplace,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        self.keyswitch(module, lhs, &rhs.key, scratch);
        (0..self.rank() + 1).for_each(|i| {
            module.vec_znx_automorphism_inplace(rhs.p(), &mut self.data, i);
        })
    }

    pub fn automorphism_inplace<DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &AutomorphismKeyExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEKeyswitchFamily<B> + VecZnxAutomorphismInplace,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        self.keyswitch_inplace(module, &rhs.key, scratch);
        (0..self.rank() + 1).for_each(|i| {
            module.vec_znx_automorphism_inplace(rhs.p(), &mut self.data, i);
        })
    }

    pub fn automorphism_add<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &AutomorphismKeyExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEKeyswitchFamily<B> + VecZnxBigAutomorphismInplace<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        #[cfg(debug_assertions)]
        {
            self.assert_keyswitch(module, lhs, &rhs.key, scratch);
        }
        let (res_dft, scratch1) = scratch.take_vec_znx_dft(module, self.cols(), rhs.size()); // TODO: optimise size
        let mut res_big: VecZnxBig<_, B> = keyswitch(module, res_dft, lhs, &rhs.key, scratch1);
        (0..self.cols()).for_each(|i| {
            module.vec_znx_big_automorphism_inplace(rhs.p(), &mut res_big, i);
            module.vec_znx_big_add_small_inplace(&mut res_big, i, &lhs.data, i);
            module.vec_znx_big_normalize(self.basek(), &mut self.data, i, &res_big, i, scratch1);
        })
    }

    pub fn automorphism_add_inplace<DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &AutomorphismKeyExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEKeyswitchFamily<B> + VecZnxBigAutomorphismInplace<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        unsafe {
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            self.automorphism_add(module, &*self_ptr, rhs, scratch);
        }
    }

    pub fn automorphism_sub_ab<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &AutomorphismKeyExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEKeyswitchFamily<B> + VecZnxBigAutomorphismInplace<B> + VecZnxBigSubSmallAInplace<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        #[cfg(debug_assertions)]
        {
            self.assert_keyswitch(module, lhs, &rhs.key, scratch);
        }
        let (res_dft, scratch1) = scratch.take_vec_znx_dft(module, self.cols(), rhs.size()); // TODO: optimise size
        let mut res_big: VecZnxBig<_, B> = keyswitch(module, res_dft, lhs, &rhs.key, scratch1);
        (0..self.cols()).for_each(|i| {
            module.vec_znx_big_automorphism_inplace(rhs.p(), &mut res_big, i);
            module.vec_znx_big_sub_small_a_inplace(&mut res_big, i, &lhs.data, i);
            module.vec_znx_big_normalize(self.basek(), &mut self.data, i, &res_big, i, scratch1);
        })
    }

    pub fn automorphism_sub_ab_inplace<DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &AutomorphismKeyExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEKeyswitchFamily<B> + VecZnxBigAutomorphismInplace<B> + VecZnxBigSubSmallAInplace<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        unsafe {
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            self.automorphism_sub_ab(module, &*self_ptr, rhs, scratch);
        }
    }

    pub fn automorphism_sub_ba<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &AutomorphismKeyExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEKeyswitchFamily<B> + VecZnxBigAutomorphismInplace<B> + VecZnxBigSubSmallBInplace<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        #[cfg(debug_assertions)]
        {
            self.assert_keyswitch(module, lhs, &rhs.key, scratch);
        }
        let (res_dft, scratch1) = scratch.take_vec_znx_dft(module, self.cols(), rhs.size()); // TODO: optimise size
        let mut res_big: VecZnxBig<_, B> = keyswitch(module, res_dft, lhs, &rhs.key, scratch1);
        (0..self.cols()).for_each(|i| {
            module.vec_znx_big_automorphism_inplace(rhs.p(), &mut res_big, i);
            module.vec_znx_big_sub_small_b_inplace(&mut res_big, i, &lhs.data, i);
            module.vec_znx_big_normalize(self.basek(), &mut self.data, i, &res_big, i, scratch1);
        })
    }

    pub fn automorphism_sub_ba_inplace<DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &AutomorphismKeyExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEKeyswitchFamily<B> + VecZnxBigAutomorphismInplace<B> + VecZnxBigSubSmallBInplace<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        unsafe {
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            self.automorphism_sub_ba(module, &*self_ptr, rhs, scratch);
        }
    }
}
