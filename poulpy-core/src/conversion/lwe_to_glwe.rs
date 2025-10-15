use poulpy_hal::{
    api::{
        ScratchAvailable, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftApply,
        VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, VecZnx, ZnxView, ZnxViewMut, ZnxZero},
};

use crate::layouts::{GGLWEInfos, GLWE, GLWEInfos, GLWELayout, LWE, LWEInfos, prepared::LWEToGLWESwitchingKeyPrepared};

impl GLWE<Vec<u8>> {
    pub fn from_lwe_scratch_space<B: Backend, OUT, IN, KEY>(
        module: &Module<B>,
        glwe_infos: &OUT,
        lwe_infos: &IN,
        key_infos: &KEY,
    ) -> usize
    where
        OUT: GLWEInfos,
        IN: LWEInfos,
        KEY: GGLWEInfos,
        Module<B>: VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        let ct: usize = GLWE::bytes_of(
            module.n().into(),
            key_infos.base2k(),
            lwe_infos.k().max(glwe_infos.k()),
            1u32.into(),
        );
        let ks: usize = GLWE::keyswitch_inplace_scratch_space(module, glwe_infos, key_infos);
        if lwe_infos.base2k() == key_infos.base2k() {
            ct + ks
        } else {
            let a_conv = VecZnx::bytes_of(module.n(), 1, lwe_infos.size()) + module.vec_znx_normalize_tmp_bytes();
            ct + a_conv + ks
        }
    }
}

impl<D: DataMut> GLWE<D> {
    pub fn from_lwe<DLwe, DKsk, B: Backend>(
        &mut self,
        module: &Module<B>,
        lwe: &LWE<DLwe>,
        ksk: &LWEToGLWESwitchingKeyPrepared<DKsk, B>,
        scratch: &mut Scratch<B>,
    ) where
        DLwe: DataRef,
        DKsk: DataRef,
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
        Scratch<B>: ScratchAvailable,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), module.n() as u32);
            assert_eq!(ksk.n(), module.n() as u32);
            assert!(lwe.n() <= module.n() as u32);
        }

        let (mut glwe, scratch_1) = scratch.take_glwe_ct(&GLWELayout {
            n: ksk.n(),
            base2k: ksk.base2k(),
            k: lwe.k(),
            rank: 1u32.into(),
        });
        glwe.data.zero();

        let n_lwe: usize = lwe.n().into();

        if lwe.base2k() == ksk.base2k() {
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
                ksk.base2k().into(),
                &mut glwe.data,
                0,
                lwe.base2k().into(),
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
                ksk.base2k().into(),
                &mut glwe.data,
                1,
                lwe.base2k().into(),
                &a_conv,
                0,
                scratch_2,
            );
        }

        self.keyswitch(module, &glwe, &ksk.0, scratch_1);
    }
}
