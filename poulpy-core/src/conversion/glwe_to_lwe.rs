use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxView, ZnxViewMut, ZnxZero},
};

use crate::{
    TakeGLWE,
    layouts::{GGLWEInfos, GLWE, GLWEInfos, GLWELayout, LWE, LWEInfos, Rank, prepared::GLWEToLWESwitchingKeyPrepared},
};

impl LWE<Vec<u8>> {
    pub fn from_glwe_scratch_space<B: Backend, OUT, IN, KEY>(
        module: &Module<B>,
        lwe_infos: &OUT,
        glwe_infos: &IN,
        key_infos: &KEY,
    ) -> usize
    where
        OUT: LWEInfos,
        IN: GLWEInfos,
        KEY: GGLWEInfos,
        Module<B>: VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        let glwe_layout: GLWELayout = GLWELayout {
            n: module.n().into(),
            base2k: lwe_infos.base2k(),
            k: lwe_infos.k(),
            rank: Rank(1),
        };

        GLWE::bytes_of(
            module.n().into(),
            lwe_infos.base2k(),
            lwe_infos.k(),
            1u32.into(),
        ) + GLWE::keyswitch_scratch_space(module, &glwe_layout, glwe_infos, key_infos)
    }
}

impl<DLwe: DataMut> LWE<DLwe> {
    pub fn sample_extract<DGlwe: DataRef>(&mut self, a: &GLWE<DGlwe>) {
        #[cfg(debug_assertions)]
        {
            assert!(self.n() <= a.n());
            assert!(self.base2k() == a.base2k());
        }

        let min_size: usize = self.size().min(a.size());
        let n: usize = self.n().into();

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
        a: &GLWE<DGlwe>,
        ks: &GLWEToLWESwitchingKeyPrepared<DKs, B>,
        scratch: &mut Scratch<B>,
    ) where
        DGlwe: DataRef,
        DKs: DataRef,
        Module<B>: VecZnxDftBytesOf
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
        Scratch<B>: ScratchAvailable + TakeVecZnxDft<B> + TakeGLWE + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n() as u32);
            assert_eq!(ks.n(), module.n() as u32);
            assert!(self.n() <= module.n() as u32);
        }

        let glwe_layout: GLWELayout = GLWELayout {
            n: module.n().into(),
            base2k: self.base2k(),
            k: self.k(),
            rank: Rank(1),
        };

        let (mut tmp_glwe, scratch_1) = scratch.take_glwe_ct(&glwe_layout);
        tmp_glwe.keyswitch(module, a, &ks.0, scratch_1);
        self.sample_extract(&tmp_glwe);
    }
}
