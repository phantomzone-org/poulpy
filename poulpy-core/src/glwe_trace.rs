use std::collections::HashMap;

use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnxDft, VecZnxBigAddSmallInplace, VecZnxBigAutomorphismInplace, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftAllocBytes, VecZnxDftFromVecZnx, VecZnxDftToVecZnxBigConsume,
        VecZnxRshInplace, VmpApply, VmpApplyAdd, VmpApplyTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    layouts::{GLWECiphertext, prepared::GGLWEAutomorphismKeyPrepared},
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
        basek: usize,
        out_k: usize,
        in_k: usize,
        ksk_k: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyTmpBytes + VecZnxBigNormalizeTmpBytes,
    {
        Self::automorphism_inplace_scratch_space(module, basek, out_k.min(in_k), ksk_k, digits, rank)
    }

    pub fn trace_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        out_k: usize,
        ksk_k: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyTmpBytes + VecZnxBigNormalizeTmpBytes,
    {
        Self::automorphism_inplace_scratch_space(module, basek, out_k, ksk_k, digits, rank)
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
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxRshInplace
            + VecZnxCopy,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
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
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxRshInplace,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        (start..end).for_each(|i| {
            self.rsh(module, 1);

            let p: i64 = if i == 0 {
                -1
            } else {
                module.galois_element(1 << (i - 1))
            };

            if let Some(key) = auto_keys.get(&p) {
                self.automorphism_add_inplace(module, key, scratch);
            } else {
                panic!("auto_keys[{}] is empty", p)
            }
        });
    }
}
