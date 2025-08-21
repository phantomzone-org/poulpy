use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftAllocBytes, VecZnxDftFromVecZnx, VecZnxDftToVecZnxBigConsume, VmpApply, VmpApplyAdd, VmpApplyTmpBytes, ZnxView,
        ZnxViewMut, ZnxZero,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    TakeGLWECt,
    layouts::{GLWECiphertext, Infos, LWECiphertext, prepared::LWESwitchingKeyPrepared},
};

impl LWECiphertext<Vec<u8>> {
    pub fn keyswitch_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_lwe_out: usize,
        k_lwe_in: usize,
        k_ksk: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>,
    {
        GLWECiphertext::bytes_of(module.n(), basek, k_lwe_out.max(k_lwe_in), 1)
            + GLWECiphertext::keyswitch_inplace_scratch_space(module, basek, k_lwe_out, k_ksk, 1, 1)
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
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert!(self.n() <= module.n());
            assert!(a.n() <= module.n());
            assert_eq!(self.basek(), a.basek());
        }

        let max_k: usize = self.k().max(a.k());
        let basek: usize = self.basek();

        let (mut glwe, scratch1) = scratch.take_glwe_ct(ksk.n(), basek, max_k, 1);
        glwe.data.zero();

        let n_lwe: usize = a.n();

        (0..a.size()).for_each(|i| {
            let data_lwe: &[i64] = a.data.at(0, i);
            glwe.data.at_mut(0, i)[0] = data_lwe[0];
            glwe.data.at_mut(1, i)[..n_lwe].copy_from_slice(&data_lwe[1..]);
        });

        glwe.keyswitch_inplace(module, &ksk.0, scratch1);

        self.sample_extract(&glwe);
    }
}
