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
    layouts::{GLWECiphertext, Infos, LWECiphertext, prepared::LWESwitchingKeyPrepared},
};

impl LWECiphertext<Vec<u8>> {
    pub fn keyswitch_scratch_space<B: Backend>(
        module: &Module<B>,
        basek_out: usize,
        k_lwe_out: usize,
        basek_in: usize,
        k_lwe_in: usize,
        basek_ksk: usize,
        k_ksk: usize,
    ) -> usize
    where
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
        let ct: usize = GLWECiphertext::bytes_of(module.n(), basek_ksk, k_lwe_out.max(k_lwe_in), 1);
        let ks: usize = GLWECiphertext::keyswitch_inplace_scratch_space(module, basek_out, k_lwe_out, basek_ksk, k_ksk, 1, 1);

        if basek_in == basek_ksk {
            ct + ks
        } else {
            let a_conv = VecZnx::alloc_bytes(module.n(), 1, k_lwe_in.div_ceil(basek_in)) + module.vec_znx_normalize_tmp_bytes();
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
            assert!(self.n() <= module.n());
            assert!(a.n() <= module.n());
            assert_eq!(self.basek(), a.basek());
        }

        let max_k: usize = self.k().max(a.k());
        let basek_in: usize = a.basek();
        let basek_out: usize = self.basek();

        let a_size: usize = a.k().div_ceil(ksk.basek());

        let (mut glwe_in, scratch_1) = scratch.take_glwe_ct(ksk.n(), basek_in, max_k, 1);
        glwe_in.data.zero();
        let (mut glwe_out, scratch_1) = scratch_1.take_glwe_ct(ksk.n(), basek_out, max_k, 1);

        let n_lwe: usize = a.n();

        for i in 0..a_size {
            let data_lwe: &[i64] = a.data.at(0, i);
            glwe_in.data.at_mut(0, i)[0] = data_lwe[0];
            glwe_in.data.at_mut(1, i)[..n_lwe].copy_from_slice(&data_lwe[1..]);
        }

        glwe_out.keyswitch(module, &glwe_in, &ksk.0, scratch_1);
        self.sample_extract(&glwe_out);
    }
}
