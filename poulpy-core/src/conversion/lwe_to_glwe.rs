use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftAllocBytes, VecZnxDftApply, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, VecZnx, ZnxView, ZnxViewMut, ZnxZero},
};

use crate::{
    TakeGLWECt,
    layouts::{GGLWEMetadata, GLWECiphertext, GLWEMetadata, Infos, LWECiphertext, prepared::LWEToGLWESwitchingKeyPrepared},
};

impl GLWECiphertext<Vec<u8>> {
    pub fn from_lwe_scratch_space<B: Backend>(
        module: &Module<B>,
        glwe_metadata: GLWEMetadata,
        lwe_metadata: GLWEMetadata,
        key_metadata: GGLWEMetadata,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        let ct: usize = GLWECiphertext::bytes_of(
            module.n(),
            key_metadata.basek,
            lwe_metadata.k.max(glwe_metadata.k),
            1,
        );
        let ks: usize = GLWECiphertext::keyswitch_inplace_scratch_space(module, glwe_metadata, key_metadata);
        if lwe_metadata.basek == key_metadata.basek {
            ct + ks
        } else {
            let a_conv = VecZnx::alloc_bytes(module.n(), 1, lwe_metadata.k.div_ceil(lwe_metadata.basek))
                + module.vec_znx_normalize_tmp_bytes();
            ct + a_conv + ks
        }
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
            assert_eq!(self.n(), module.n());
            assert_eq!(ksk.n(), module.n());
            assert!(lwe.n() <= module.n());
        }

        let (mut glwe, scratch_1) = scratch.take_glwe_ct(ksk.n(), ksk.basek(), lwe.k(), 1);
        glwe.data.zero();

        let n_lwe: usize = lwe.n();

        if lwe.basek() == ksk.basek() {
            for i in 0..lwe.size() {
                let data_lwe: &[i64] = lwe.data.at(0, i);
                glwe.data.at_mut(0, i)[0] = data_lwe[0];
                glwe.data.at_mut(1, i)[..n_lwe].copy_from_slice(&data_lwe[1..]);
            }
        } else {
            let (mut a_conv, scratch_2) = scratch_1.take_vec_znx(module.n(), 1, lwe.size());
            a_conv.zero();
            for j in 0..lwe.size() {
                let data_lwe: &[i64] = lwe.data.at(0, j);
                a_conv.at_mut(0, j)[0] = data_lwe[0]
            }

            module.vec_znx_normalize(
                ksk.basek(),
                &mut glwe.data,
                0,
                lwe.basek(),
                &a_conv,
                0,
                scratch_2,
            );

            a_conv.zero();
            for j in 0..lwe.size() {
                let data_lwe: &[i64] = lwe.data.at(0, j);
                a_conv.at_mut(0, j)[..n_lwe].copy_from_slice(&data_lwe[1..]);
            }

            module.vec_znx_normalize(
                ksk.basek(),
                &mut glwe.data,
                1,
                lwe.basek(),
                &a_conv,
                0,
                scratch_2,
            );
        }

        self.keyswitch(module, &glwe, &ksk.0, scratch_1);
    }
}
