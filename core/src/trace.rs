use std::collections::HashMap;

use backend::{FFT64, MatZnxDft, MatZnxDftToRef, Module, Scratch, VecZnx, VecZnxToMut, VecZnxToRef};

use crate::{automorphism::AutomorphismKey, glwe_ciphertext::GLWECiphertext};

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
        out_size: usize,
        in_size: usize,
        autokey_size: usize,
        rank: usize,
    ) -> usize {
        Self::automorphism_inplace_scratch_space(module, out_size.max(in_size), rank, autokey_size)
    }

    pub fn trace_inplace_scratch_space(module: &Module<FFT64>, out_size: usize, autokey_size: usize, rank: usize) -> usize {
        Self::automorphism_inplace_scratch_space(module, out_size, rank, autokey_size)
    }
}

impl<DataSelf> GLWECiphertext<DataSelf>
where
    VecZnx<DataSelf>: VecZnxToMut + VecZnxToRef,
{
    pub fn trace<DataLhs, DataAK>(
        &mut self,
        module: &Module<FFT64>,
        start: usize,
        end: usize,
        lhs: &GLWECiphertext<DataLhs>,
        auto_keys: &HashMap<i64, AutomorphismKey<DataAK, FFT64>>,
        scratch: &mut Scratch,
    ) where
        VecZnx<DataLhs>: VecZnxToRef,
        MatZnxDft<DataAK, FFT64>: MatZnxDftToRef<FFT64>,
    {
        self.copy(lhs);
        self.trace_inplace(module, start, end, auto_keys, scratch);
    }

    pub fn trace_inplace<DataAK>(
        &mut self,
        module: &Module<FFT64>,
        start: usize,
        end: usize,
        auto_keys: &HashMap<i64, AutomorphismKey<DataAK, FFT64>>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataAK, FFT64>: MatZnxDftToRef<FFT64>,
    {
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
