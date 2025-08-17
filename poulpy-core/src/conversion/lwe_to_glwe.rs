use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnxDft, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftAllocBytes, VecZnxDftFromVecZnx, VecZnxDftToVecZnxBigConsume, VmpApply, VmpApplyAdd, VmpApplyTmpBytes, ZnxView,
        ZnxViewMut, ZnxZero,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    TakeGLWECt,
    layouts::{GLWECiphertext, Infos, LWECiphertext, prepared::LWEToGLWESwitchingKeyPrepared},
};

impl GLWECiphertext<Vec<u8>> {
    pub fn from_lwe_scratch_space<B: Backend>(
        module: &Module<B>,
        n: usize,
        basek: usize,
        k_lwe: usize,
        k_glwe: usize,
        k_ksk: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyTmpBytes + VecZnxBigNormalizeTmpBytes,
    {
        GLWECiphertext::keyswitch_scratch_space(module, n, basek, k_glwe, k_lwe, k_ksk, 1, 1, rank)
            + GLWECiphertext::bytes_of(n, basek, k_lwe, 1)
    }
}

impl<D: DataMut> GLWECiphertext<D> {
    pub fn from_lwe<DLwe, DKsk, B: Backend>(
        &mut self,
        module: &Module<B>,
        lwe: &LWECiphertext<DLwe>,
        ksk: &LWEToGLWESwitchingKeyPrepared<DKsk, B>,
        scratch: &mut Scratch<B>,
    ) where
        DLwe: DataRef,
        DKsk: DataRef,
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>,
        Scratch<B>: ScratchAvailable + TakeVecZnxDft<B> + TakeGLWECt,
    {
        #[cfg(debug_assertions)]
        {
            assert!(lwe.n() <= self.n());
            assert_eq!(self.basek(), self.basek());
        }

        let (mut glwe, scratch1) = scratch.take_glwe_ct(ksk.n(), lwe.basek(), lwe.k(), 1);
        glwe.data.zero();

        let n_lwe: usize = lwe.n();

        (0..lwe.size()).for_each(|i| {
            let data_lwe: &[i64] = lwe.data.at(0, i);
            glwe.data.at_mut(0, i)[0] = data_lwe[0];
            glwe.data.at_mut(1, i)[..n_lwe].copy_from_slice(&data_lwe[1..]);
        });

        self.keyswitch(module, &glwe, &ksk.0, scratch1);
    }
}
