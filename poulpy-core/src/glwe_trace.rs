use std::collections::HashMap;

use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxBigAddSmallInplace, VecZnxBigAutomorphismInplace, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxIdftApplyConsume, VecZnxNormalize,
        VecZnxNormalizeTmpBytes, VecZnxRshInplace, VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, VecZnx},
};

use crate::{
    TakeGLWECt,
    layouts::{Base2K, GGLWEInfos, GLWECiphertext, GLWECiphertextLayout, GLWEInfos, LWEInfos, prepared::AutomorphismKeyPrepared},
    operations::GLWEOperations,
};

impl GLWECiphertext<Vec<u8>> {
    pub fn trace_galois_elements<B: Backend>(module: &Module<B>) -> Vec<i64> {
        let mut gal_els: Vec<i64> = Vec::new();
        (0..module.log_n()).for_each(|i| {
            if i == 0 {
                gal_els.push(-1);
            } else {
                gal_els.push(module.galois_element(1 << (i - 1)));
            }
        });
        gal_els
    }

    pub fn trace_scratch_space<B: Backend, OUT, IN, KEY>(
        module: &Module<B>,
        out_infos: &OUT,
        in_infos: &IN,
        key_infos: &KEY,
    ) -> usize
    where
        OUT: GLWEInfos,
        IN: GLWEInfos,
        KEY: GGLWEInfos,
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        let trace: usize = Self::automorphism_inplace_scratch_space(module, out_infos, key_infos);
        if in_infos.base2k() != key_infos.base2k() {
            let glwe_conv: usize = VecZnx::alloc_bytes(
                module.n(),
                (key_infos.rank_out() + 1).into(),
                out_infos.k().min(in_infos.k()).div_ceil(key_infos.base2k()) as usize,
            ) + module.vec_znx_normalize_tmp_bytes();
            return glwe_conv + trace;
        }

        trace
    }

    pub fn trace_inplace_scratch_space<B: Backend, OUT, KEY>(module: &Module<B>, out_infos: &OUT, key_infos: &KEY) -> usize
    where
        OUT: GLWEInfos,
        KEY: GGLWEInfos,
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        Self::trace_scratch_space(module, out_infos, out_infos, key_infos)
    }
}

impl<DataSelf: DataMut> GLWECiphertext<DataSelf> {
    pub fn trace<DataLhs: DataRef, DataAK: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        start: usize,
        end: usize,
        lhs: &GLWECiphertext<DataLhs>,
        auto_keys: &HashMap<i64, AutomorphismKeyPrepared<DataAK, B>>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxRshInplace<B>
            + VecZnxCopy
            + VecZnxNormalizeTmpBytes
            + VecZnxNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        self.copy(module, lhs);
        self.trace_inplace(module, start, end, auto_keys, scratch);
    }

    pub fn trace_inplace<DataAK: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        start: usize,
        end: usize,
        auto_keys: &HashMap<i64, AutomorphismKeyPrepared<DataAK, B>>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxRshInplace<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        let basek_ksk: Base2K = auto_keys
            .get(auto_keys.keys().next().unwrap())
            .unwrap()
            .base2k();

        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), module.n() as u32);
            assert!(start < end);
            assert!(end <= module.log_n());
            for key in auto_keys.values() {
                assert_eq!(key.n(), module.n() as u32);
                assert_eq!(key.base2k(), basek_ksk);
                assert_eq!(key.rank_in(), self.rank());
                assert_eq!(key.rank_out(), self.rank());
            }
        }

        if self.base2k() != basek_ksk {
            let (mut self_conv, scratch_1) = scratch.take_glwe_ct(&GLWECiphertextLayout {
                n: module.n().into(),
                base2k: basek_ksk,
                k: self.k(),
                rank: self.rank(),
            });

            for j in 0..(self.rank() + 1).into() {
                module.vec_znx_normalize(
                    basek_ksk.into(),
                    &mut self_conv.data,
                    j,
                    basek_ksk.into(),
                    &self.data,
                    j,
                    scratch_1,
                );
            }

            for i in start..end {
                self_conv.rsh(module, 1, scratch_1);

                let p: i64 = if i == 0 {
                    -1
                } else {
                    module.galois_element(1 << (i - 1))
                };

                if let Some(key) = auto_keys.get(&p) {
                    self_conv.automorphism_add_inplace(module, key, scratch_1);
                } else {
                    panic!("auto_keys[{p}] is empty")
                }
            }

            for j in 0..(self.rank() + 1).into() {
                module.vec_znx_normalize(
                    self.base2k().into(),
                    &mut self.data,
                    j,
                    basek_ksk.into(),
                    &self_conv.data,
                    j,
                    scratch_1,
                );
            }
        } else {
            for i in start..end {
                self.rsh(module, 1, scratch);

                let p: i64 = if i == 0 {
                    -1
                } else {
                    module.galois_element(1 << (i - 1))
                };

                if let Some(key) = auto_keys.get(&p) {
                    self.automorphism_add_inplace(module, key, scratch);
                } else {
                    panic!("auto_keys[{p}] is empty")
                }
            }
        }
    }
}
