use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxCopy, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, VecZnx, ZnxView, ZnxViewMut, ZnxZero},
};

use crate::{
    layouts::{prepared::LWESwitchingKeyPrepared, GGLWEMetadata, GLWECiphertext, GLWEMetadata, Infos, LWECiphertext, LWEMetadata}, TakeGLWECt
};

impl LWECiphertext<Vec<u8>> {
    pub fn keyswitch_scratch_space<B: Backend>(
        module: &Module<B>,
        out_metadata: LWEMetadata,
        in_metadata: LWEMetadata,
        key_metadata: GGLWEMetadata,
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
        let ct: usize = GLWECiphertext::bytes_of(
            module.n(),
            key_metadata.basek,
            out_metadata.k.max(in_metadata.basek),
            1,
        );
        let ks: usize = GLWECiphertext::keyswitch_inplace_scratch_space(module, out_metadata.as_glwe(), key_metadata);

        if in_metadata.basek == key_metadata.basek {
            ct + ks
        } else {
            let a_conv = VecZnx::alloc_bytes(module.n(), 1, in_metadata.k.div_ceil(in_metadata.basek))
                + module.vec_znx_normalize_tmp_bytes();
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
        }

        let max_k: usize = self.k().max(a.k());
        let basek_in: usize = a.basek();
        let basek_out: usize = self.basek();

        let a_size: usize = a.k().div_ceil(ksk.basek());

        let (mut glwe_in, scratch_1) = scratch.take_glwe_ct(ksk.n(), GLWEMetadata{basek: basek_in, k: max_k, rank: 1});
        glwe_in.data.zero();
        let (mut glwe_out, scratch_1) = scratch_1.take_glwe_ct(ksk.n(), GLWEMetadata{basek: basek_out, k: max_k, rank: 1});

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
