use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftAllocBytes, VecZnxDftApply, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxView, ZnxViewMut, ZnxZero},
};

use crate::{
    TakeGLWECt,
    layouts::{GLWECiphertext, Infos, LWECiphertext, prepared::GLWEToLWESwitchingKeyPrepared},
};

impl LWECiphertext<Vec<u8>> {
    pub fn from_glwe_scratch_space<B: Backend>(
        module: &Module<B>,
        basek_lwe: usize,
        k_lwe: usize,
        basek_glwe: usize,
        k_glwe: usize,
        basek_ksk: usize,
        k_ksk: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        GLWECiphertext::bytes_of(module.n(), basek_lwe, k_lwe, 1)
            + GLWECiphertext::keyswitch_scratch_space(
                module, basek_lwe, k_lwe, basek_glwe, k_glwe, basek_ksk, k_ksk, 1, rank, 1,
            )
    }
}

impl<DLwe: DataMut> LWECiphertext<DLwe> {
    pub fn sample_extract<DGlwe: DataRef>(&mut self, a: &GLWECiphertext<DGlwe>) {
        #[cfg(debug_assertions)]
        {
            assert!(self.n() <= a.n());
            assert!(self.basek() == a.basek());
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
        Scratch<B>: ScratchAvailable + TakeVecZnxDft<B> + TakeGLWECt + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(ks.n(), module.n());
            assert!(self.n() <= module.n());
        }
        let (mut tmp_glwe, scratch_1) = scratch.take_glwe_ct(module.n(), self.basek(), self.k(), 1);
        tmp_glwe.keyswitch(module, a, &ks.0, scratch_1);
        self.sample_extract(&tmp_glwe);
    }
}
