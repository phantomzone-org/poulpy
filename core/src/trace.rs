use std::collections::HashMap;

use backend::{FFT64, Module, Scratch};

use crate::{AutomorphismKey, GLWECiphertext, GLWECiphertextToMut, GLWECiphertextToRef, GLWEOps, Infos, SetMetaData};

impl GLWECiphertext<Vec<u8>> {
    pub fn trace_galois_elements(module: &Module<FFT64>) -> Vec<i64> {
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

    pub fn trace_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        out_k: usize,
        in_k: usize,
        atk_k: usize,
        rank: usize,
    ) -> usize {
        Self::automorphism_inplace_scratch_space(module, basek, out_k.min(in_k), atk_k, rank)
    }

    pub fn trace_inplace_scratch_space(module: &Module<FFT64>, basek: usize, out_k: usize, atk_k: usize, rank: usize) -> usize {
        Self::automorphism_inplace_scratch_space(module, basek, out_k, atk_k, rank)
    }
}

impl<DataSelf: AsRef<[u8]> + AsMut<[u8]>> GLWECiphertext<DataSelf>
where
    GLWECiphertext<DataSelf>: GLWECiphertextToMut + Infos + SetMetaData,
{
    pub fn trace<DataLhs: AsRef<[u8]>, DataAK: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        start: usize,
        end: usize,
        lhs: &GLWECiphertext<DataLhs>,
        auto_keys: &HashMap<i64, AutomorphismKey<DataAK, FFT64>>,
        scratch: &mut Scratch,
    ) where
        GLWECiphertext<DataLhs>: GLWECiphertextToRef + Infos,
    {
        self.copy(module, lhs);
        self.trace_inplace(module, start, end, auto_keys, scratch);
    }

    pub fn trace_inplace<DataAK: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        start: usize,
        end: usize,
        auto_keys: &HashMap<i64, AutomorphismKey<DataAK, FFT64>>,
        scratch: &mut Scratch,
    ) {
        (start..end).for_each(|i| {
            self.rsh(1, scratch);

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
