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
    layouts::{GLWECiphertext, Infos, LWECiphertext, prepared::GLWEToLWESwitchingKeyPrepared},
};

impl LWECiphertext<Vec<u8>> {
    pub fn from_glwe_scratch_space<B: Backend>(
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
        GLWECiphertext::bytes_of(n, basek, k_lwe, 1)
            + GLWECiphertext::keyswitch_scratch_space(module, n, basek, k_lwe, k_glwe, k_ksk, 1, rank, 1)
    }
}

impl<DLwe: DataMut> LWECiphertext<DLwe> {
    pub fn sample_extract<DGlwe: DataRef>(&mut self, a: &GLWECiphertext<DGlwe>) {
        #[cfg(debug_assertions)]
        {
            assert!(self.n() <= a.n());
        }

        let min_size: usize = self.size().min(a.size());
        let n: usize = self.n();

        self.data.zero();
        (0..min_size).for_each(|i| {
            let data_lwe: &mut [i64] = self.data.at_mut(0, i);
            data_lwe[0] = a.data.at(0, i)[0];
            data_lwe[1..].copy_from_slice(&a.data.at(1, i)[..n]);
        });
    }

    pub fn from_glwe<DGlwe, DKs, B: Backend>(
        &mut self,
        module: &Module<B>,
        a: &GLWECiphertext<DGlwe>,
        ks: &GLWEToLWESwitchingKeyPrepared<DKs, B>,
        scratch: &mut Scratch<B>,
    ) where
        DGlwe: DataRef,
        DKs: DataRef,
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
            assert_eq!(self.basek(), a.basek());
            assert_eq!(a.n(), ks.n());
        }
        let (mut tmp_glwe, scratch1) = scratch.take_glwe_ct(a.n(), a.basek(), self.k(), 1);
        tmp_glwe.keyswitch(module, a, &ks.0, scratch1);
        self.sample_extract(&tmp_glwe);
    }
}
