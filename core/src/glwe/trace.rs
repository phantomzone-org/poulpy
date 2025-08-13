use std::collections::HashMap;

use backend::hal::{
    api::{ScratchAvailable, TakeVecZnxDft, VecZnxBigAutomorphismInplace, VecZnxCopy, VecZnxRshInplace},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    AutomorphismKeyExec, GLWECiphertext, GLWECiphertextToMut, GLWECiphertextToRef, GLWEKeyswitchFamily, GLWEOps, Infos,
    SetMetaData,
};

pub trait GLWETraceFamily<B: Backend> = GLWEKeyswitchFamily<B> + VecZnxCopy + VecZnxRshInplace + VecZnxBigAutomorphismInplace<B>;

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

    pub fn trace_scratch_space<B: Backend>(
        module: &Module<B>,
        n: usize,
        basek: usize,
        out_k: usize,
        in_k: usize,
        ksk_k: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        Self::automorphism_inplace_scratch_space(module, n, basek, out_k.min(in_k), ksk_k, digits, rank)
    }

    pub fn trace_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        n: usize,
        basek: usize,
        out_k: usize,
        ksk_k: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        Self::automorphism_inplace_scratch_space(module, n, basek, out_k, ksk_k, digits, rank)
    }
}

impl<DataSelf: DataMut> GLWECiphertext<DataSelf>
where
    GLWECiphertext<DataSelf>: GLWECiphertextToMut + Infos + SetMetaData,
{
    pub fn trace<DataLhs: DataRef, DataAK: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        start: usize,
        end: usize,
        lhs: &GLWECiphertext<DataLhs>,
        auto_keys: &HashMap<i64, AutomorphismKeyExec<DataAK, B>>,
        scratch: &mut Scratch<B>,
    ) where
        GLWECiphertext<DataLhs>: GLWECiphertextToRef + Infos + VecZnxRshInplace,
        Module<B>: GLWETraceFamily<B>,
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
        auto_keys: &HashMap<i64, AutomorphismKeyExec<DataAK, B>>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWETraceFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        (start..end).for_each(|i| {
            self.rsh(module, 1);

            let p: i64;
            if i == 0 {
                p = -1;
            } else {
                p = module.galois_element(1 << (i - 1));
            }

            if let Some(key) = auto_keys.get(&p) {
                self.automorphism_add_inplace(module, key, scratch);
            } else {
                panic!("auto_keys[{}] is empty", p)
            }
        });
    }
}
