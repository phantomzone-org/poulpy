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
    layouts::{GLWECiphertext, Infos, prepared::GGLWEAutomorphismKeyPrepared},
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

    #[allow(clippy::too_many_arguments)]
    pub fn trace_scratch_space<B: Backend>(
        module: &Module<B>,
        basek_out: usize,
        k_out: usize,
        basek_in: usize,
        k_in: usize,
        basek_ksk: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        let trace = Self::automorphism_inplace_scratch_space(
            module,
            basek_out,
            k_out.min(k_in),
            basek_ksk,
            k_ksk,
            digits,
            rank,
        );
        if basek_in != basek_ksk {
            let glwe_conv: usize = VecZnx::alloc_bytes(module.n(), rank + 1, k_out.min(k_in).div_ceil(basek_ksk))
                + module.vec_znx_normalize_tmp_bytes();
            return glwe_conv + trace;
        }

        trace
    }

    pub fn trace_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        basek_out: usize,
        k_out: usize,
        basek_ksk: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        Self::trace_scratch_space(
            module, basek_out, k_out, basek_out, k_out, basek_ksk, k_ksk, digits, rank,
        )
    }
}

impl<DataSelf: DataMut> GLWECiphertext<DataSelf> {
    pub fn trace<DataLhs: DataRef, DataAK: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        start: usize,
        end: usize,
        lhs: &GLWECiphertext<DataLhs>,
        auto_keys: &HashMap<i64, GGLWEAutomorphismKeyPrepared<DataAK, B>>,
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
        auto_keys: &HashMap<i64, GGLWEAutomorphismKeyPrepared<DataAK, B>>,
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
        let basek_ksk: usize = auto_keys
            .get(auto_keys.keys().next().unwrap())
            .unwrap()
            .basek();

        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), module.n());
            assert!(start < end);
            assert!(end <= module.log_n());
            for (_, key) in auto_keys {
                assert_eq!(key.n(), module.n());
                assert_eq!(key.basek(), basek_ksk);
                assert_eq!(key.rank_in(), self.rank());
                assert_eq!(key.rank_out(), self.rank());
            }
        }

        if self.basek() != basek_ksk {
            let (mut self_conv, scratch_1) = scratch.take_glwe_ct(module.n(), basek_ksk, self.k(), self.rank());

            for j in 0..self.cols() {
                module.vec_znx_normalize(
                    basek_ksk,
                    &mut self_conv.data,
                    j,
                    basek_ksk,
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

            for j in 0..self.cols() {
                module.vec_znx_normalize(
                    self.basek(),
                    &mut self.data,
                    j,
                    basek_ksk,
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
