use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxCopy, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxView, ZnxViewMut, ZnxZero},
};

use crate::{
    TakeGLWECt,
    layouts::{GGLWEInfos, GLWE, GLWELayout, LWE, LWEInfos, Rank, TorusPrecision, prepared::LWESwitchingKeyPrepared},
};

impl LWE<Vec<u8>> {
    pub fn keyswitch_scratch_space<B: Backend, OUT, IN, KEY>(
        module: &Module<B>,
        out_infos: &OUT,
        in_infos: &IN,
        key_infos: &KEY,
    ) -> usize
    where
        OUT: LWEInfos,
        IN: LWEInfos,
        KEY: GGLWEInfos,
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
            + VecZnxNormalizeTmpBytes,
    {
        let max_k: TorusPrecision = in_infos.k().max(out_infos.k());

        let glwe_in_infos: GLWELayout = GLWELayout {
            n: module.n().into(),
            base2k: in_infos.base2k(),
            k: max_k,
            rank: Rank(1),
        };

        let glwe_out_infos: GLWELayout = GLWELayout {
            n: module.n().into(),
            base2k: out_infos.base2k(),
            k: max_k,
            rank: Rank(1),
        };

        let glwe_in: usize = GLWE::bytes_of(&glwe_in_infos);
        let glwe_out: usize = GLWE::bytes_of(&glwe_out_infos);
        let ks: usize = GLWE::keyswitch_scratch_space(module, &glwe_out_infos, &glwe_in_infos, key_infos);

        glwe_in + glwe_out + ks
    }
}

impl<DLwe: DataMut> LWE<DLwe> {
    pub fn keyswitch<A, DKs, B: Backend>(
        &mut self,
        module: &Module<B>,
        a: &LWE<A>,
        ksk: &LWESwitchingKeyPrepared<DKs, B>,
        scratch: &mut Scratch<B>,
    ) where
        A: DataRef,
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
            + VecZnxNormalizeTmpBytes
            + VecZnxCopy,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert!(self.n() <= module.n() as u32);
            assert!(a.n() <= module.n() as u32);
            assert!(scratch.available() >= LWE::keyswitch_scratch_space(module, self, a, ksk));
        }

        let max_k: TorusPrecision = self.k().max(a.k());

        let a_size: usize = a.k().div_ceil(ksk.base2k()) as usize;

        let (mut glwe_in, scratch_1) = scratch.take_glwe_ct(&GLWELayout {
            n: ksk.n(),
            base2k: a.base2k(),
            k: max_k,
            rank: Rank(1),
        });
        glwe_in.data.zero();

        let (mut glwe_out, scratch_1) = scratch_1.take_glwe_ct(&GLWELayout {
            n: ksk.n(),
            base2k: self.base2k(),
            k: max_k,
            rank: Rank(1),
        });

        let n_lwe: usize = a.n().into();

        for i in 0..a_size {
            let data_lwe: &[i64] = a.data.at(0, i);
            glwe_in.data.at_mut(0, i)[0] = data_lwe[0];
            glwe_in.data.at_mut(1, i)[..n_lwe].copy_from_slice(&data_lwe[1..]);
        }

        glwe_out.keyswitch(module, &glwe_in, &ksk.0, scratch_1);
        self.sample_extract(&glwe_out);
    }
}
