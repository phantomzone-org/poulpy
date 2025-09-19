use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxCopy, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, VecZnx, ZnxView, ZnxViewMut, ZnxZero},
};

use crate::{
    TakeGLWECt,
    layouts::{
        GGLWELayoutInfos, GLWECiphertext, GLWECiphertextLayout, LWECiphertext, LWEInfos, Rank, TorusPrecision,
        prepared::LWESwitchingKeyPrepared,
    },
};

impl LWECiphertext<Vec<u8>> {
    pub fn keyswitch_scratch_space<B: Backend, OUT, IN, KEY>(
        module: &Module<B>,
        out_infos: &OUT,
        in_infos: &IN,
        key_infos: &KEY,
    ) -> usize
    where
        OUT: LWEInfos,
        IN: LWEInfos,
        KEY: GGLWELayoutInfos,
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
        let ct: usize = GLWECiphertext::alloc_bytes_with(
            module.n().into(),
            key_infos.base2k(),
            out_infos.k().max(in_infos.k()),
            1usize.into(),
        );

        let glwe_layout: GLWECiphertextLayout = GLWECiphertextLayout {
            n: module.n().into(),
            base2k: out_infos.base2k(),
            k: out_infos.k(),
            rank: Rank(1),
        };

        let ks: usize = GLWECiphertext::keyswitch_inplace_scratch_space(module, &glwe_layout, key_infos);

        if in_infos.base2k() == key_infos.base2k() {
            ct + ks
        } else {
            let a_conv = VecZnx::alloc_bytes(module.n(), 1, in_infos.size()) + module.vec_znx_normalize_tmp_bytes();
            ct + a_conv + ks
        }
    }
}

impl<DLwe: DataMut> LWECiphertext<DLwe> {
    pub fn keyswitch<A, DKs, B: Backend>(
        &mut self,
        module: &Module<B>,
        a: &LWECiphertext<A>,
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
        }

        let max_k: TorusPrecision = self.k().max(a.k());

        let a_size: usize = a.k().div_ceil(ksk.base2k()) as usize;

        let (mut glwe_in, scratch_1) = scratch.take_glwe_ct(&GLWECiphertextLayout {
            n: ksk.n(),
            base2k: a.base2k(),
            k: max_k,
            rank: Rank(1),
        });
        glwe_in.data.zero();
        let (mut glwe_out, scratch_1) = scratch_1.take_glwe_ct(&GLWECiphertextLayout {
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
